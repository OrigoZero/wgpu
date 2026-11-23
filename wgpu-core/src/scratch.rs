//! The scratch memory acceleration structure builds use.
//!
//! A build takes a scratch buffer from the device's [`ScratchPool`] and holds
//! it until the submission that runs the build is done; the buffer then goes
//! back to the pool for the next build. So a frame that builds or updates its
//! structures costs no buffer creation once the pool holds what the builds in
//! flight need. A pooled buffer no build has taken for
//! [`SCRATCH_POOL_IDLE_ACQUISITIONS`] acquisitions is destroyed, so the scratch
//! a burst of builds (a scene load) needed goes back to the device.

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::mem::ManuallyDrop;

use wgt::BufferUses;

use crate::device::{Device, DeviceError};
use crate::{hal_label, resource_log};

/// How many acquisitions a pooled scratch buffer may go untaken before the
/// pool destroys it.
pub(crate) const SCRATCH_POOL_IDLE_ACQUISITIONS: u32 = 64;

/// The most free scratch buffers a [`ScratchPool`] keeps.
pub(crate) const SCRATCH_POOL_CAPACITY: usize = 8;

#[derive(Debug)]
struct FreeScratch {
    raw: Box<dyn hal::DynBuffer>,
    size: u64,
    idle: u32,
}

/// The free scratch buffers of a device.
#[derive(Debug, Default)]
pub(crate) struct ScratchPool {
    free: Vec<FreeScratch>,
}

impl ScratchPool {
    /// The smallest free buffer of at least `size` bytes, if the pool holds
    /// one. Every other free buffer goes one acquisition idler, and one idle
    /// for too long is destroyed.
    fn acquire(
        &mut self,
        device: &dyn hal::DynDevice,
        size: u64,
    ) -> Option<(Box<dyn hal::DynBuffer>, u64)> {
        let best = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, f)| f.size >= size)
            .min_by_key(|(_, f)| f.size)
            .map(|(i, _)| i);
        let taken = best.map(|i| self.free.swap_remove(i));
        let mut i = 0;
        while i < self.free.len() {
            self.free[i].idle += 1;
            if self.free[i].idle > SCRATCH_POOL_IDLE_ACQUISITIONS {
                let f = self.free.swap_remove(i);
                unsafe { device.destroy_buffer(f.raw) };
            } else {
                i += 1;
            }
        }
        taken.map(|f| (f.raw, f.size))
    }

    fn release(&mut self, device: &dyn hal::DynDevice, raw: Box<dyn hal::DynBuffer>, size: u64) {
        if self.free.len() >= SCRATCH_POOL_CAPACITY {
            let smallest = self
                .free
                .iter()
                .enumerate()
                .min_by_key(|(_, f)| f.size)
                .map(|(i, f)| (i, f.size));
            match smallest {
                Some((i, smallest)) if smallest < size => {
                    let f = self.free.swap_remove(i);
                    unsafe { device.destroy_buffer(f.raw) };
                }
                _ => {
                    unsafe { device.destroy_buffer(raw) };
                    return;
                }
            }
        }
        self.free.push(FreeScratch { raw, size, idle: 0 });
    }

    /// Destroy every free buffer.
    pub(crate) fn dispose(&mut self, device: &dyn hal::DynDevice) {
        for f in self.free.drain(..) {
            unsafe { device.destroy_buffer(f.raw) };
        }
    }
}

#[derive(Debug)]
pub struct ScratchBuffer {
    raw: ManuallyDrop<Box<dyn hal::DynBuffer>>,
    size: u64,
    device: Arc<Device>,
}

impl ScratchBuffer {
    /// A scratch buffer of at least `size` bytes: one from the device's pool
    /// when it holds one, otherwise a buffer created for it, sized to the
    /// power of two `size` rounds up to so a slowly growing need creates few.
    pub(crate) fn new(device: &Arc<Device>, size: wgt::BufferSize) -> Result<Self, DeviceError> {
        let pooled = device.scratch_pool.lock().acquire(device.raw(), size.get());
        let (raw, size) = match pooled {
            Some(pooled) => pooled,
            None => {
                let size = size.get().checked_next_power_of_two().unwrap_or(size.get());
                let raw = unsafe {
                    device
                        .raw()
                        .create_buffer(&hal::BufferDescriptor {
                            label: hal_label(Some("(wgpu) scratch buffer"), device.instance_flags),
                            size,
                            usage: BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
                            memory_flags: hal::MemoryFlags::empty(),
                        })
                        .map_err(DeviceError::from_hal)?
                };
                (raw, size)
            }
        };
        Ok(Self {
            raw: ManuallyDrop::new(raw),
            size,
            device: device.clone(),
        })
    }
    pub(crate) fn raw(&self) -> &dyn hal::DynBuffer {
        self.raw.as_ref()
    }
}

impl Drop for ScratchBuffer {
    fn drop(&mut self) {
        resource_log!("Release raw ScratchBuffer");
        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        self.device
            .scratch_pool
            .lock()
            .release(self.device.raw(), raw, self.size);
    }
}
