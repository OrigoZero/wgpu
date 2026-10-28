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
//! indirect-draw validation pass) is a whole chunk from the same pool when its
//! bytes fit in one, back in the pool once the submission that copies out of
//! it is done ([`StagingBuffer::new`]).
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

/// How many chunk acquisitions make one trimming window of a
/// [`StagingPool`].
///
/// At the end of each window the pool destroys the free chunks no acquisition
/// in the window reached, so the pool settles at the most chunks the device
/// had in flight at once during the window.
pub(crate) const STAGING_POOL_TRIM_WINDOW: usize = 1024;

/// The most free chunks a [`StagingPool`] keeps. A chunk released to a pool
/// holding this many is destroyed, so a burst of uploads (a scene load) hands
/// its staging memory back as its submissions retire.
pub(crate) const STAGING_POOL_CAPACITY: usize = 16;

/// The free staging chunks of a device, each an unmapped buffer of
/// [`STAGING_CHUNK_SIZE`] bytes that no submission still reads.
pub(crate) struct StagingPool {
    free: Mutex<TrimmedStack<Box<dyn hal::DynBuffer>>>,
}

impl StagingPool {
    pub(crate) fn new() -> Self {
        Self {
            free: Mutex::new(
                rank::DEVICE_STAGING_POOL,
                TrimmedStack::new(STAGING_POOL_TRIM_WINDOW),
            ),
        }
    }

    /// A free chunk buffer, taken from the pool or, when the pool is empty,
    /// created on `device`.
    pub(crate) fn acquire(&self, device: &Device) -> Result<Box<dyn hal::DynBuffer>, DeviceError> {
        let (raw, idle) = self.free.lock().pop();
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
            label: hal_label(Some("(wgpu internal) Staging chunk"), device.instance_flags),
            size: STAGING_CHUNK_SIZE,
            usage: wgt::BufferUses::MAP_WRITE | wgt::BufferUses::COPY_SRC,
            memory_flags: hal::MemoryFlags::TRANSIENT,
        };
        let raw =
            unsafe { device.raw().create_buffer(&desc) }.map_err(|e| device.handle_hal_error(e))?;
        device.counters.staging_buffer_creations.add(1);
        Ok(raw)
    }

    /// Put the chunk buffer `raw`, unmapped and read by no pending
    /// submission, back in the pool, or destroy it when the pool is full.
    pub(crate) fn release(&self, device: &Device, raw: Box<dyn hal::DynBuffer>) {
        let mut free = self.free.lock();
        if free.len() < STAGING_POOL_CAPACITY {
            free.push(raw);
            drop(free);
            device.counters.pooled_staging_buffers.add(1);
        } else {
            drop(free);
            unsafe { device.raw().destroy_buffer(raw) };
        }
    }

    /// Destroy every chunk the pool holds. The device calls this as it is
    /// dropped, when nothing else holds a chunk.
    pub(crate) fn dispose(&self, device: &dyn hal::DynDevice) {
        let held: Vec<_> = self.free.lock().drain();
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
        let raw = device.staging_pool.acquire(device)?;
        let mapping = match unsafe { device.raw().map_buffer(raw.as_ref(), 0..STAGING_CHUNK_SIZE) }
        {
            Ok(mapping) => mapping,
            Err(e) => {
                device.staging_pool.release(device, raw);
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
        FlushedStagingBuffer::pooled(raw, self.device.clone(), STAGING_CHUNK_SIZE)
    }
}

impl Drop for StagingChunk {
    /// A chunk dropped while open (the queue dropped between two
    /// submissions) was copied from by no submission: it goes back to the
    /// pool unmapped.
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            unsafe { self.device.raw().unmap_buffer(raw.as_ref()) };
            self.device.staging_pool.release(&self.device, raw);
        }
    }
}
