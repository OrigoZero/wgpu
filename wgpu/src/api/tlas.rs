use crate::{api::blas::TlasInstance, dispatch, util::Mutex};
use crate::{BindingResource, Label};
use alloc::{vec, vec::Vec};
#[cfg(wgpu_core)]
use core::ops::Deref;
use core::ops::{Index, IndexMut, Range};
use wgt::WasmNotSendSync;

/// Descriptor to create top level acceleration structures.
pub type CreateTlasDescriptor<'a> = wgt::CreateTlasDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(CreateTlasDescriptor<'_>: Send, Sync);

#[derive(Debug)]
/// Top Level Acceleration Structure (TLAS).
///
/// A TLAS contains a series of [TLAS instances], which are a reference to
/// a BLAS and a transformation matrix placing the geometry in the world.
///
/// A TLAS also contains an extra set of TLAS instances in a device readable form, you cant interact
/// directly with these, instead you have to build the TLAS with [TLAS instances].
///
/// A build hands the device only the instances written since the TLAS was
/// last built, so a build costs the CPU what changed rather than how many
/// instances the TLAS holds.
///
/// [TLAS instances]: TlasInstance
pub struct Tlas {
    pub(crate) inner: dispatch::DispatchTlas,
    pub(crate) instances: Vec<Option<TlasInstance>>,
    pub(crate) lowest_unmodified: u32,
    pub(crate) modified: Mutex<ModifiedInstances>,
}
static_assertions::assert_impl_all!(Tlas: WasmNotSendSync);

/// The instances of a [`Tlas`] written since its last build.
#[derive(Debug, Default)]
pub(crate) struct ModifiedInstances {
    marked: Vec<bool>,
    indices: Vec<u32>,
}

impl ModifiedInstances {
    pub(crate) fn with_capacity(count: usize) -> Self {
        Self {
            marked: vec![false; count],
            indices: Vec::new(),
        }
    }

    fn mark(&mut self, range: Range<usize>) {
        if self.marked.len() < range.end {
            self.marked.resize(range.end, false);
        }
        for index in range {
            if !self.marked[index] {
                self.marked[index] = true;
                self.indices.push(index as u32);
            }
        }
    }

    /// The instances written since the last call, in ascending order.
    pub(crate) fn take(&mut self) -> Vec<u32> {
        let mut indices = core::mem::take(&mut self.indices);
        for &index in &indices {
            self.marked[index as usize] = false;
        }
        indices.sort_unstable();
        indices
    }
}

impl Clone for Tlas {
    /// A clone names the same acceleration structure, and its first build
    /// hands the device every instance the clone holds.
    fn clone(&self) -> Self {
        let mut modified = ModifiedInstances::with_capacity(self.instances.len());
        modified.mark(0..self.instances.len());
        Self {
            inner: self.inner.clone(),
            instances: self.instances.clone(),
            lowest_unmodified: self.lowest_unmodified,
            modified: Mutex::new(modified),
        }
    }
}

crate::cmp::impl_eq_ord_hash_proxy!(Tlas => .inner);

impl Tlas {
    /// Get the [`wgpu_hal`] acceleration structure from this `Tlas`.
    ///
    /// Find the Api struct corresponding to the active backend in [`wgpu_hal::api`],
    /// and pass that struct to the to the `A` type parameter.
    ///
    /// Returns a guard that dereferences to the type of the hal backend
    /// which implements [`A::AccelerationStructure`].
    ///
    /// # Types
    ///
    /// The returned type depends on the backend:
    ///
    #[doc = crate::macros::hal_type_vulkan!("AccelerationStructure")]
    #[doc = crate::macros::hal_type_metal!("AccelerationStructure")]
    #[doc = crate::macros::hal_type_dx12!("AccelerationStructure")]
    #[doc = crate::macros::hal_type_gles!("AccelerationStructure")]
    ///
    /// # Deadlocks
    ///
    /// - The returned guard holds a read-lock on a device-local "destruction"
    ///   lock, which will cause all calls to `destroy` to block until the
    ///   guard is released.
    ///
    /// # Errors
    ///
    /// This method will return None if:
    /// - The acceleration structure is not from the backend specified by `A`.
    /// - The acceleration structure is from the `webgpu` or `custom` backend.
    ///
    /// # Safety
    ///
    /// - The returned resource must not be destroyed unless the guard
    ///   is the last reference to it and it is not in use by the GPU.
    ///   The guard and handle may be dropped at any time however.
    /// - All the safety requirements of wgpu-hal must be upheld.
    ///
    /// [`A::AccelerationStructure`]: hal::Api::AccelerationStructure
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal<A: hal::Api>(
        &mut self,
    ) -> Option<impl Deref<Target = A::AccelerationStructure>> {
        let tlas = self.inner.as_core_opt()?;
        unsafe { tlas.context.tlas_as_hal::<A>(tlas) }
    }

    /// Returns custom implementation of Tlas (if custom backend and is internally T)
    #[cfg(custom)]
    pub fn as_custom<T: crate::custom::TlasInterface>(&self) -> Option<&T> {
        self.inner.as_custom()
    }
    /// Returns the index of the first instance that has not been modified since the last build.
    /// Custom backends use this to perform partial TLAS updates.
    #[cfg(custom)]
    pub fn lowest_unmodified(&self) -> u32 {
        self.lowest_unmodified
    }

    /// Get a reference to all instances.
    pub fn get(&self) -> &[Option<TlasInstance>] {
        &self.instances
    }

    /// Make the next build of this TLAS build it from its instances rather
    /// than update it, whatever its [`AccelerationStructureUpdateMode`].
    ///
    /// An update keeps the structure's hierarchy and moves its bounds, so a
    /// structure updated over many frames of motion traces slower than a
    /// fresh build; rebuilding every so often restores it.
    ///
    /// [`AccelerationStructureUpdateMode`]: crate::AccelerationStructureUpdateMode
    pub fn request_rebuild(&self) {
        self.inner.request_rebuild();
    }

    /// The instances written since the last build, which the build hands to
    /// the device.
    #[cfg_attr(not(wgpu_core), expect(dead_code))]
    pub(crate) fn take_modified(&self) -> Vec<u32> {
        self.modified.lock().take()
    }

    /// Get a mutable slice to a range of instances.
    /// Returns None if the range is out of bounds.
    /// All elements from the lowest accessed index up are marked as modified.
    // this recommendation is not useful yet, but is likely to be when ability to update arrives or possible optimisations for building get implemented.
    /// For best performance it is recommended to prefer access to low elements and modify higher elements as little as possible.
    /// This can be done by ordering instances from the most to the least used. It is recommended
    /// to use [`Self::index_mut`] unless the option if out of bounds is required
    pub fn get_mut_slice(&mut self, range: Range<usize>) -> Option<&mut [Option<TlasInstance>]> {
        if range.end > self.instances.len() {
            return None;
        }
        if range.end as u32 > self.lowest_unmodified {
            self.lowest_unmodified = range.end as u32;
        }
        self.modified.lock().mark(range.clone());
        Some(&mut self.instances[range])
    }

    /// Get a single mutable reference to an instance.
    /// Returns None if the range is out of bounds.
    /// All elements from the lowest accessed index up are marked as modified.
    // this recommendation is not useful yet, but is likely to be when ability to update arrives or possible optimisations for building get implemented.
    /// For best performance it is recommended to prefer access to low elements and modify higher elements as little as possible.
    /// This can be done by ordering instances from the most to the least used. It is recommended
    /// to use [`Self::index_mut`] unless the option if out of bounds is required
    pub fn get_mut_single(&mut self, index: usize) -> Option<&mut Option<TlasInstance>> {
        if index >= self.instances.len() {
            return None;
        }
        if index as u32 + 1 > self.lowest_unmodified {
            self.lowest_unmodified = index as u32 + 1;
        }
        self.modified.lock().mark(index..index + 1);
        Some(&mut self.instances[index])
    }

    /// Get the binding resource for the underling acceleration structure, to be used when creating a [`BindGroup`]
    ///
    /// [`BindGroup`]: super::BindGroup
    pub fn as_binding(&self) -> BindingResource<'_> {
        BindingResource::AccelerationStructure(self)
    }
}

impl Index<usize> for Tlas {
    type Output = Option<TlasInstance>;

    fn index(&self, index: usize) -> &Self::Output {
        self.instances.index(index)
    }
}

impl Index<Range<usize>> for Tlas {
    type Output = [Option<TlasInstance>];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        self.instances.index(index)
    }
}

impl IndexMut<usize> for Tlas {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let idx = self.instances.index_mut(index);
        if index as u32 + 1 > self.lowest_unmodified {
            self.lowest_unmodified = index as u32 + 1;
        }
        self.modified.lock().mark(index..index + 1);
        idx
    }
}

impl IndexMut<Range<usize>> for Tlas {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        let idx = self.instances.index_mut(index.clone());
        if index.end > self.lowest_unmodified as usize {
            self.lowest_unmodified = index.end as u32;
        }
        self.modified.lock().mark(index);
        idx
    }
}
