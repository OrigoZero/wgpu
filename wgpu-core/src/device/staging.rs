//! The staging memory a queue's writes are copied out of.
//!
//! `Queue::write_buffer` and `Queue::write_texture` copy the caller's bytes
//! into host-visible memory and record a copy from there into the
//! destination, on the queue's [`PendingWrites`] encoder. The writes made
//! between two submissions share a [`StagingChunk`]: each write takes the
//! next aligned range of the chunk, and a write that does not fit in what is
//! left of it closes the chunk and opens another. The submission that copies
//! out of a chunk retires it, and once that submission is done the chunk's
//! buffer goes back to the device's [`StagingPool`], from which the next chunk
//! is taken. So a frame's small writes cost no buffer creation each: the
//! device creates staging buffers while the load it carries grows, and reuses
//! them from then on.
//!
//! A write larger than a whole chunk is staged in a buffer of its own,
//! created for it and destroyed once its submission is done.
//!
//! Every other staging buffer the device maps for one copy (the one
//! `Queue::create_staging_buffer` hands out for `Queue::write_buffer_with`,
//! the initial contents of a buffer mapped at creation, the metadata of an
//! indirect-draw validation pass) comes from the same pool when its bytes fit
//! in a chunk: a buffer of the power of two its bytes round up to, back in the
//! pool once the submission that copies out of it is done
//! ([`StagingBuffer::new`]).
//!
//! [`StagingBuffer::new`]: crate::resource::StagingBuffer::new
//!
//! [`PendingWrites`]: crate::device::queue::PendingWrites

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::ptr::NonNull;

use crate::{
    command::TrimmedStack,
    device::{Device, DeviceError},
    hal_label,
    lock::{rank, Mutex},
    resource::FlushedStagingBuffer,
};

/// Size in bytes of each staging chunk, and the largest write a chunk holds.
pub(crate) const STAGING_CHUNK_SIZE: u64 = 256 << 10;

/// Size in bytes of the smallest pooled staging buffer.
const SMALLEST_STAGING_CLASS_SIZE: u64 = 4 << 10;

/// How many size classes a [`StagingPool`] keeps buffers in: one per power of
/// two from [`SMALLEST_STAGING_CLASS_SIZE`] to [`STAGING_CHUNK_SIZE`].
const STAGING_CLASS_COUNT: usize = (STAGING_CHUNK_SIZE.trailing_zeros()
    - SMALLEST_STAGING_CLASS_SIZE.trailing_zeros()) as usize
    + 1;

/// How many acquisitions of one size class make one trimming window of a
/// [`StagingPool`].
///
/// At the end of each window the pool destroys the free buffers of the class
/// no acquisition in the window reached, so the class settles at the most
/// buffers the device had in flight at once during the window.
pub(crate) const STAGING_POOL_TRIM_WINDOW: usize = 1024;

/// The most free buffers a [`StagingPool`] keeps in one size class. A buffer
/// released to a class holding this many is destroyed, so a burst of uploads
/// (a scene load) hands its staging memory back as its submissions retire.
pub(crate) const STAGING_POOL_CAPACITY: usize = 32;

/// The size class of a pooled staging buffer: the power of two its bytes
/// round up to, from [`SMALLEST_STAGING_CLASS_SIZE`] to [`STAGING_CHUNK_SIZE`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StagingClass(u8);

impl StagingClass {
    /// The class of a whole staging chunk.
    pub(crate) const CHUNK: Self = Self((STAGING_CLASS_COUNT - 1) as u8);

    /// The class a staging buffer of `size` bytes is pooled in, or `None`
    /// when `size` is larger than a chunk.
    pub(crate) fn of(size: u64) -> Option<Self> {
        if size > STAGING_CHUNK_SIZE {
            return None;
        }
        let rounded = size.max(SMALLEST_STAGING_CLASS_SIZE).next_power_of_two();
        Some(Self(
            (rounded.trailing_zeros() - SMALLEST_STAGING_CLASS_SIZE.trailing_zeros()) as u8,
        ))
    }

    /// Size in bytes of the buffers of this class.
    pub(crate) fn size(self) -> u64 {
        SMALLEST_STAGING_CLASS_SIZE << self.0
    }
}

/// The free staging buffers of a device, each an unmapped buffer that no
/// submission still reads, kept by [`StagingClass`]: the queue's staging
/// chunks in the largest class, and the staging buffers mapped for one copy in
/// the class their bytes round up to.
pub(crate) struct StagingPool {
    free: Mutex<[TrimmedStack<Box<dyn hal::DynBuffer>>; STAGING_CLASS_COUNT]>,
}

impl StagingPool {
    pub(crate) fn new() -> Self {
        Self {
            free: Mutex::new(
                rank::DEVICE_STAGING_POOL,
                core::array::from_fn(|_| TrimmedStack::new(STAGING_POOL_TRIM_WINDOW)),
            ),
        }
    }

    /// A free buffer of `class`, taken from the pool or, when the pool holds
    /// none, created on `device`.
    pub(crate) fn acquire(
        &self,
        device: &Device,
        class: StagingClass,
    ) -> Result<Box<dyn hal::DynBuffer>, DeviceError> {
        let (raw, idle) = self.free.lock()[class.0 as usize].pop();
        let destroyed = idle.len() as isize;
        for raw in idle {
            unsafe { device.raw().destroy_buffer(raw) };
        }
        device.counters.pooled_staging_buffers.sub(destroyed);
        if let Some(raw) = raw {
            device.counters.pooled_staging_buffers.sub(1);
            return Ok(raw);
        }
        let desc = hal::BufferDescriptor {
            label: hal_label(
                Some("(wgpu internal) Pooled staging"),
                device.instance_flags,
            ),
            size: class.size(),
            usage: wgt::BufferUses::MAP_WRITE | wgt::BufferUses::COPY_SRC,
            memory_flags: hal::MemoryFlags::TRANSIENT,
        };
        let raw =
            unsafe { device.raw().create_buffer(&desc) }.map_err(|e| device.handle_hal_error(e))?;
        device.counters.staging_buffer_creations.add(1);
        Ok(raw)
    }

    /// Put `raw`, a buffer of `class` unmapped and read by no pending
    /// submission, back in the pool, or destroy it when its class is full.
    pub(crate) fn release(
        &self,
        device: &Device,
        raw: Box<dyn hal::DynBuffer>,
        class: StagingClass,
    ) {
        let mut free = self.free.lock();
        let stack = &mut free[class.0 as usize];
        if stack.len() < STAGING_POOL_CAPACITY {
            stack.push(raw);
            drop(free);
            device.counters.pooled_staging_buffers.add(1);
        } else {
            drop(free);
            unsafe { device.raw().destroy_buffer(raw) };
        }
    }

    /// Destroy every buffer the pool holds. The device calls this as it is
    /// dropped, when nothing else holds one.
    pub(crate) fn dispose(&self, device: &dyn hal::DynDevice) {
        let held: Vec<_> = self
            .free
            .lock()
            .iter_mut()
            .flat_map(|stack| stack.drain())
            .collect();
        for raw in held {
            unsafe { device.destroy_buffer(raw) };
        }
    }
}

/// The chunk of staging memory the queue's pending writes are filling: a
/// pool buffer, mapped, of which the first `used` bytes hold staged writes.
#[derive(Debug)]
pub(crate) struct StagingChunk {
    /// The chunk's buffer, taken out when the chunk is retired.
    raw: Option<Box<dyn hal::DynBuffer>>,
    device: Arc<Device>,
    ptr: NonNull<u8>,
    is_coherent: bool,
    used: u64,
}

#[cfg(send_sync)]
unsafe impl Send for StagingChunk {}
#[cfg(send_sync)]
unsafe impl Sync for StagingChunk {}

impl StagingChunk {
    /// Take a chunk from `device`'s pool and map it.
    pub(crate) fn acquire(device: &Arc<Device>) -> Result<Self, DeviceError> {
        let raw = device.staging_pool.acquire(device, StagingClass::CHUNK)?;
        let mapping = match unsafe { device.raw().map_buffer(raw.as_ref(), 0..STAGING_CHUNK_SIZE) }
        {
            Ok(mapping) => mapping,
            Err(e) => {
                device
                    .staging_pool
                    .release(device, raw, StagingClass::CHUNK);
                return Err(device.handle_hal_error(e));
            }
        };
        Ok(Self {
            raw: Some(raw),
            device: device.clone(),
            ptr: mapping.ptr,
            is_coherent: mapping.is_coherent,
            used: 0,
        })
    }

    /// The chunk's buffer.
    pub(crate) fn raw(&self) -> &dyn hal::DynBuffer {
        self.raw
            .as_deref()
            .expect("an open staging chunk holds its buffer")
    }

    /// Whether any write was staged in the chunk.
    pub(crate) fn is_used(&self) -> bool {
        self.used != 0
    }

    /// Take the next `size` bytes of the chunk at an offset aligned to
    /// `alignment`, returning that offset and the bytes, or `None` when the
    /// chunk has no such range left.
    pub(crate) fn allocate(&mut self, size: u64, alignment: u64) -> Option<(u64, &mut [u8])> {
        let offset = wgt::math::align_to(self.used, alignment);
        let end = offset.checked_add(size)?;
        if end > STAGING_CHUNK_SIZE {
            return None;
        }
        self.used = end;
        // SAFETY: The chunk is mapped for its whole size, `offset..end` lies
        // inside it, and no other range handed out overlaps this one.
        let bytes = unsafe {
            core::slice::from_raw_parts_mut(self.ptr.as_ptr().add(offset as usize), size as usize)
        };
        Some((offset, bytes))
    }

    /// Make the staged bytes visible to the device and unmap the chunk,
    /// handing it over to the submission that copies out of it. Once that
    /// submission is done the returned buffer is dropped, which puts the chunk
    /// back in the device's pool.
    pub(crate) fn retire(mut self) -> FlushedStagingBuffer {
        let raw = self
            .raw
            .take()
            .expect("an open staging chunk holds its buffer");
        let device = self.device.raw();
        if !self.is_coherent && self.used != 0 {
            #[allow(clippy::single_range_in_vec_init)]
            unsafe {
                device.flush_mapped_ranges(raw.as_ref(), &[0..self.used])
            };
        }
        unsafe { device.unmap_buffer(raw.as_ref()) };
        FlushedStagingBuffer::pooled(
            raw,
            self.device.clone(),
            STAGING_CHUNK_SIZE,
            StagingClass::CHUNK,
        )
    }
}

impl Drop for StagingChunk {
    /// A chunk dropped while open (the queue dropped between two
    /// submissions) was copied from by no submission: it goes back to the
    /// pool unmapped.
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            unsafe { self.device.raw().unmap_buffer(raw.as_ref()) };
            self.device
                .staging_pool
                .release(&self.device, raw, StagingClass::CHUNK);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{StagingClass, SMALLEST_STAGING_CLASS_SIZE, STAGING_CHUNK_SIZE};

    #[test]
    fn a_size_takes_the_smallest_class_that_holds_it() {
        assert_eq!(
            StagingClass::of(1).map(StagingClass::size),
            Some(SMALLEST_STAGING_CLASS_SIZE)
        );
        assert_eq!(
            StagingClass::of(4 << 10).map(StagingClass::size),
            Some(4 << 10)
        );
        assert_eq!(
            StagingClass::of((4 << 10) + 4).map(StagingClass::size),
            Some(8 << 10)
        );
        assert_eq!(
            StagingClass::of(100_000).map(StagingClass::size),
            Some(128 << 10)
        );
        assert_eq!(
            StagingClass::of(STAGING_CHUNK_SIZE),
            Some(StagingClass::CHUNK)
        );
        assert_eq!(StagingClass::CHUNK.size(), STAGING_CHUNK_SIZE);
    }

    #[test]
    fn a_size_past_a_chunk_has_no_class() {
        assert_eq!(StagingClass::of(STAGING_CHUNK_SIZE + 4), None);
    }
}
