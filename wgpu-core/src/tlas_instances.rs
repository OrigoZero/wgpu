//! The instances a TLAS holds between its builds.
//!
//! A [`TlasPackage`] names only the instances written since the TLAS's last
//! build, so the TLAS keeps every instance it was given: the bottom-level
//! structure it places, and the bytes the backend reads for it. A build
//! writes into the instance buffer only the instances that changed since the
//! instance buffer of a submitted build held them, and the cost of a build on
//! the CPU follows what changed rather than how many instances the TLAS holds.
//!
//! While the instances the TLAS holds are one run from index 0, instance `i`
//! lives at position `i` of the instance buffer and a build copies the changed
//! runs alone. A TLAS holding gaps is laid out packed, and each of its builds
//! writes the whole buffer.
//!
//! [`TlasPackage`]: crate::ray_tracing::TlasPackage

use alloc::{sync::Arc, vec, vec::Vec};
use core::ops::Range;

use crate::{
    ray_tracing::{TlasBuild, TlasShape},
    resource::Blas,
    FastHashMap,
};

/// What a submitted build of a TLAS did, as [`TlasInstances::submitted`]
/// reads it.
pub(crate) struct SubmittedBuild<'a> {
    pub shape: Option<TlasShape>,
    pub updated: bool,
    pub uploaded: &'a [(u32, u32)],
    pub wrote_whole: bool,
    pub encode_index: u64,
}

impl<'a> From<&'a TlasBuild> for SubmittedBuild<'a> {
    fn from(build: &'a TlasBuild) -> Self {
        Self {
            shape: build.shape,
            updated: build.updated,
            uploaded: &build.uploaded,
            wrote_whole: build.wrote_whole,
            encode_index: build.encode_index,
        }
    }
}

/// One copy into the instance buffer: bytes `src` of the staged bytes, to
/// offset `dst` of the instance buffer.
pub(crate) struct InstanceCopy {
    pub src: Range<usize>,
    pub dst: u64,
}

/// What a build writes into the instance buffer.
pub(crate) struct InstanceUpload {
    pub bytes: Vec<u8>,
    pub copies: Vec<InstanceCopy>,
    /// The instances written, each with its version.
    pub uploaded: Vec<(u32, u32)>,
    /// Whether the build writes the whole instance buffer.
    pub whole: bool,
}

/// The instances of one TLAS. `B` is the bottom-level structure an instance
/// places.
#[derive(Debug)]
pub(crate) struct TlasInstances<B = Blas> {
    stride: usize,
    slots: Vec<Option<Arc<B>>>,
    bytes: Vec<u8>,
    versions: Vec<u32>,
    /// The instances the instance buffer of a submitted build does not hold
    /// as they stand.
    pending: Vec<u32>,
    pending_mark: Vec<bool>,
    /// When set, the next build writes the whole instance buffer: the encoded
    /// build count it was asked at.
    whole_since: Option<u64>,
    active: u32,
    end: u32,
    /// The bottom-level structures the instances place, each with how many
    /// instances place it.
    refs: FastHashMap<usize, (Arc<B>, u32)>,
    /// Moves whenever an instance comes, goes or places another structure.
    structure_version: u64,
    submitted: Option<TlasShape>,
    encoded: Option<TlasShape>,
    encode_count: u64,
    submitted_encode: u64,
}

impl<B> TlasInstances<B> {
    pub(crate) fn new(max_instances: u32, stride: u32) -> Self {
        let max = max_instances as usize;
        let mut slots = Vec::with_capacity(max);
        slots.resize_with(max, || None);
        Self {
            stride: stride as usize,
            slots,
            bytes: vec![0; max * stride as usize],
            versions: vec![0; max],
            pending: Vec::new(),
            pending_mark: vec![false; max],
            whole_since: None,
            active: 0,
            end: 0,
            refs: FastHashMap::default(),
            structure_version: 0,
            submitted: None,
            encoded: None,
            encode_count: 0,
            submitted_encode: 0,
        }
    }

    pub(crate) fn active(&self) -> u32 {
        self.active
    }

    /// Hold `instance` at `index`: the structure it places and the bytes the
    /// backend reads for it, or nothing.
    pub(crate) fn set(&mut self, index: u32, instance: Option<(Arc<B>, &[u8])>) {
        let i = index as usize;
        let old = self.slots[i].take();
        let same_structure = match (&old, &instance) {
            (Some(a), Some((b, _))) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        if !same_structure {
            self.structure_version = self.structure_version.wrapping_add(1);
            if let Some(old) = &old {
                let key = Arc::as_ptr(old) as usize;
                if let Some(entry) = self.refs.get_mut(&key) {
                    entry.1 -= 1;
                    if entry.1 == 0 {
                        self.refs.remove(&key);
                    }
                }
            }
            if let Some((blas, _)) = &instance {
                self.refs
                    .entry(Arc::as_ptr(blas) as usize)
                    .or_insert_with(|| (blas.clone(), 0))
                    .1 += 1;
            }
        }
        match (old.is_some(), instance.is_some()) {
            (false, true) => self.active += 1,
            (true, false) => self.active -= 1,
            _ => {}
        }
        match instance {
            Some((blas, bytes)) => {
                self.bytes[i * self.stride..(i + 1) * self.stride].copy_from_slice(bytes);
                self.slots[i] = Some(blas);
                self.end = self.end.max(index + 1);
            }
            None => {
                while self.end > 0 && self.slots[self.end as usize - 1].is_none() {
                    self.end -= 1;
                }
            }
        }
        self.versions[i] = self.versions[i].wrapping_add(1);
        if !self.pending_mark[i] {
            self.pending_mark[i] = true;
            self.pending.push(index);
        }
    }

    /// The structures the instances place, each once.
    pub(crate) fn dependencies(&self) -> impl Iterator<Item = &Arc<B>> {
        self.refs.values().map(|(blas, _)| blas)
    }

    /// Start encoding a build: which build it is, the shape it builds, and
    /// whether it can update the structure the submitted build left.
    pub(crate) fn begin_encode(&mut self, prefer_update: bool) -> (u64, TlasShape, bool) {
        self.encode_count += 1;
        let shape = TlasShape {
            instance_count: self.active,
            structure_version: self.structure_version,
        };
        let update = prefer_update
            && self.active > 0
            && self.submitted == Some(shape)
            && self.encoded == Some(shape);
        self.encoded = Some(shape);
        (self.encode_count, shape, update)
    }

    /// What the build being encoded writes into the instance buffer.
    ///
    /// A TLAS holding gaps is written packed, which places its instances away
    /// from their indices, so the build after it writes the whole buffer
    /// again.
    pub(crate) fn upload(&mut self) -> InstanceUpload {
        let stride = self.stride;
        let gaps = self.active != self.end;
        let mut out = InstanceUpload {
            bytes: Vec::new(),
            copies: Vec::new(),
            uploaded: self
                .pending
                .iter()
                .map(|&slot| (slot, self.versions[slot as usize]))
                .collect(),
            whole: gaps || self.whole_since.is_some(),
        };
        if gaps {
            self.whole_since = Some(self.encode_count);
        }
        if out.whole {
            out.bytes.reserve(self.active as usize * stride);
            for (i, slot) in self.slots[..self.end as usize].iter().enumerate() {
                if slot.is_some() {
                    out.bytes
                        .extend_from_slice(&self.bytes[i * stride..(i + 1) * stride]);
                }
            }
            if !out.bytes.is_empty() {
                out.copies.push(InstanceCopy {
                    src: 0..out.bytes.len(),
                    dst: 0,
                });
            }
            return out;
        }
        let mut changed: Vec<u32> = self
            .pending
            .iter()
            .copied()
            .filter(|&slot| slot < self.end)
            .collect();
        changed.sort_unstable();
        let mut k = 0;
        while k < changed.len() {
            let first = changed[k] as usize;
            let mut last = first;
            k += 1;
            while k < changed.len() && changed[k] as usize == last + 1 {
                last += 1;
                k += 1;
            }
            let src_start = out.bytes.len();
            out.bytes
                .extend_from_slice(&self.bytes[first * stride..(last + 1) * stride]);
            out.copies.push(InstanceCopy {
                src: src_start..out.bytes.len(),
                dst: (first * stride) as u64,
            });
        }
        out
    }

    /// Ask the next build to write the whole instance buffer.
    pub(crate) fn write_whole(&mut self) {
        self.whole_since.get_or_insert(self.encode_count);
    }

    /// A build of this TLAS was submitted. Answers `false` when it updates a
    /// structure whose submitted build had another shape.
    pub(crate) fn submitted(&mut self, build: SubmittedBuild<'_>) -> bool {
        if build.updated && self.submitted != build.shape {
            return false;
        }
        self.submitted = build.shape;
        if build.encode_index < self.submitted_encode {
            self.write_whole();
        } else {
            self.submitted_encode = build.encode_index;
        }
        for &(slot, version) in build.uploaded {
            if self.versions[slot as usize] == version {
                self.pending_mark[slot as usize] = false;
            }
        }
        let marks = &self.pending_mark;
        self.pending.retain(|&slot| marks[slot as usize]);
        if build.wrote_whole && self.whole_since.is_some_and(|since| build.encode_index > since) {
            self.whole_since = None;
        }
        true
    }

    /// A build was marked done without the instances: the instance buffer
    /// holds what the caller wrote, so the next build writes it whole.
    pub(crate) fn marked_built(&mut self) {
        self.submitted = None;
        self.encoded = None;
        self.write_whole();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(v: u8) -> [u8; 4] {
        [v; 4]
    }

    /// Encode a build and submit it at once, as a frame does.
    fn build(t: &mut TlasInstances<u32>, prefer_update: bool) -> (InstanceUpload, bool) {
        let (encode_index, shape, update) = t.begin_encode(prefer_update);
        let upload = t.upload();
        assert!(t.submitted(SubmittedBuild {
            shape: Some(shape),
            updated: update,
            uploaded: &upload.uploaded,
            wrote_whole: upload.whole,
            encode_index,
        }));
        (upload, update)
    }

    #[test]
    fn a_build_writes_only_the_instances_changed_since_the_last() {
        let blas = Arc::new(0u32);
        let mut t = TlasInstances::<u32>::new(8, 4);
        for i in 0..6 {
            t.set(i, Some((blas.clone(), &bytes(i as u8))));
        }
        let (first, _) = build(&mut t, true);
        assert_eq!(first.bytes.len(), 24, "the first build writes every instance");

        t.set(1, Some((blas.clone(), &bytes(10))));
        t.set(2, Some((blas.clone(), &bytes(11))));
        t.set(4, Some((blas.clone(), &bytes(12))));
        let (second, updated) = build(&mut t, true);
        assert!(updated, "moved transforms over the same structures update");
        assert_eq!(second.bytes, [10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12]);
        let dsts: Vec<u64> = second.copies.iter().map(|c| c.dst).collect();
        assert_eq!(dsts, [4, 16], "two runs: instances 1-2 and instance 4");

        let (third, _) = build(&mut t, true);
        assert!(third.copies.is_empty(), "a build with nothing changed writes nothing");
    }

    #[test]
    fn another_structure_or_count_builds_afresh() {
        let a = Arc::new(0u32);
        let b = Arc::new(1u32);
        let mut t = TlasInstances::<u32>::new(4, 4);
        t.set(0, Some((a.clone(), &bytes(0))));
        t.set(1, Some((a.clone(), &bytes(1))));
        assert!(!build(&mut t, true).1, "the first build builds");
        t.set(1, Some((b.clone(), &bytes(1))));
        assert!(!build(&mut t, true).1, "an instance placing another structure builds");
        t.set(2, Some((b.clone(), &bytes(2))));
        assert!(!build(&mut t, true).1, "another instance count builds");
        t.set(2, Some((b.clone(), &bytes(3))));
        assert!(build(&mut t, true).1, "a moved instance updates");
        t.set(2, Some((b.clone(), &bytes(4))));
        assert!(!build(&mut t, false).1, "a structure that prefers builds builds");
        assert_eq!(t.dependencies().count(), 2);
    }

    #[test]
    fn a_tlas_with_gaps_is_written_packed_and_whole() {
        let a = Arc::new(0u32);
        let mut t = TlasInstances::<u32>::new(4, 4);
        t.set(0, Some((a.clone(), &bytes(0))));
        t.set(2, Some((a.clone(), &bytes(2))));
        let (packed, _) = build(&mut t, true);
        assert!(packed.whole);
        assert_eq!(packed.bytes, [0, 0, 0, 0, 2, 2, 2, 2]);
        assert_eq!(t.active(), 2);

        // Filling the gap puts every instance back at its index, which the
        // packed buffer did not hold: the build writes the whole buffer.
        t.set(1, Some((a.clone(), &bytes(1))));
        let (filled, _) = build(&mut t, true);
        assert!(filled.whole);
        assert_eq!(filled.bytes.len(), 12);

        t.set(1, Some((a.clone(), &bytes(5))));
        let (after, _) = build(&mut t, true);
        assert!(!after.whole, "a TLAS with no gap writes what changed");
        assert_eq!(after.bytes, [5, 5, 5, 5]);
    }

    #[test]
    fn instances_a_build_never_submitted_wrote_are_written_again() {
        let a = Arc::new(0u32);
        let mut t = TlasInstances::<u32>::new(4, 4);
        t.set(0, Some((a.clone(), &bytes(0))));
        build(&mut t, true);
        t.set(0, Some((a.clone(), &bytes(1))));
        // A build encoded and never submitted.
        let _ = t.begin_encode(true);
        let _ = t.upload();
        let (next, _) = build(&mut t, true);
        assert_eq!(next.bytes, [1, 1, 1, 1], "the instance is still owed to the buffer");
    }

    #[test]
    fn an_update_over_another_shape_is_refused_at_submission() {
        let a = Arc::new(0u32);
        let mut t = TlasInstances::<u32>::new(4, 4);
        t.set(0, Some((a.clone(), &bytes(0))));
        build(&mut t, true);
        let stale = TlasShape {
            instance_count: 7,
            structure_version: 99,
        };
        assert!(!t.submitted(SubmittedBuild {
            shape: Some(stale),
            updated: true,
            uploaded: &[],
            wrote_whole: false,
            encode_index: 9,
        }));
    }
}
