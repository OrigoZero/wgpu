use alloc::{boxed::Box, vec::Vec};

use crate::lock::{rank, Mutex};

/// How many acquisitions make one trimming window of a [`CommandAllocator`].
///
/// At the end of each window the allocator destroys the free encoders that no
/// acquisition in the window reached, so the pool settles at the most
/// encoders the device had in flight at once during the window rather than
/// at the most it ever had.
pub(crate) const COMMAND_ALLOCATOR_TRIM_WINDOW: usize = 1024;

/// A pool of free [`wgpu_hal::CommandEncoder`]s, owned by a `Device`.
///
/// Each encoder in this list is in the "closed" state.
///
/// Since a raw [`CommandEncoder`][ce] is itself a pool for allocating
/// raw [`CommandBuffer`][cb]s, this is a pool of pools.
///
/// A burst of encoders in flight at once (a scene load, a batch of uploads)
/// makes the pool create as many encoders as the burst held, and each of them
/// keeps the command buffers it allocated. The pool destroys the ones that sit
/// unused for a whole [`COMMAND_ALLOCATOR_TRIM_WINDOW`] of acquisitions, so
/// what it holds follows the load the device has now.
///
/// [`wgpu_hal::CommandEncoder`]: hal::CommandEncoder
/// [ce]: hal::CommandEncoder
/// [cb]: hal::Api::CommandBuffer
pub(crate) struct CommandAllocator {
    free_encoders: Mutex<TrimmedStack<Box<dyn hal::DynCommandEncoder>>>,
}

impl CommandAllocator {
    pub(crate) fn new() -> Self {
        Self {
            free_encoders: Mutex::new(
                rank::COMMAND_ALLOCATOR_FREE_ENCODERS,
                TrimmedStack::new(COMMAND_ALLOCATOR_TRIM_WINDOW),
            ),
        }
    }

    /// Return a fresh [`wgpu_hal::CommandEncoder`] in the "closed" state.
    ///
    /// If we have free encoders in the pool, take one of those. Otherwise,
    /// create a new one on `device`.
    ///
    /// [`wgpu_hal::CommandEncoder`]: hal::CommandEncoder
    pub(crate) fn acquire_encoder(
        &self,
        device: &dyn hal::DynDevice,
        queue: &dyn hal::DynQueue,
    ) -> Result<Box<dyn hal::DynCommandEncoder>, hal::DeviceError> {
        let (encoder, idle) = self.free_encoders.lock().pop();
        // Dropping a free hal encoder destroys its command pool and every
        // command buffer allocated from it. None of them is pending: an
        // encoder returns to the free list only once its submission is done.
        drop(idle);
        match encoder {
            Some(encoder) => Ok(encoder),
            None => unsafe {
                let hal_desc = hal::CommandEncoderDescriptor { label: None, queue };
                device.create_command_encoder(&hal_desc)
            },
        }
    }

    /// Add `encoder` back to the free pool.
    pub(crate) fn release_encoder(&self, encoder: Box<dyn hal::DynCommandEncoder>) {
        self.free_encoders.lock().push(encoder);
    }
}

/// A last-in, first-out free list that gives back what it holds idle.
///
/// The stack counts the pops it serves in windows of `window` pops. Pushes and
/// pops only ever touch its top, so the fewest items it held at any moment of
/// a window (its low-water mark) is how many items at its bottom no pop in that
/// window reached. The pop that closes a window hands those items back to the
/// caller to drop, and the next window starts from what is left.
///
/// After a window, the stack holds at most the most items that were out of it
/// at once during that window, plus what was pushed since. A burst that took
/// many items out once leaves its surplus behind for one window at most.
#[derive(Debug)]
pub(crate) struct TrimmedStack<T> {
    items: Vec<T>,
    window: usize,
    pops: usize,
    low_water: usize,
}

impl<T> TrimmedStack<T> {
    pub(crate) const fn new(window: usize) -> Self {
        Self {
            items: Vec::new(),
            window,
            pops: 0,
            low_water: 0,
        }
    }

    /// Put `item` back on top.
    pub(crate) fn push(&mut self, item: T) {
        self.items.push(item);
    }

    /// Take the top item, if any, and, when this pop closes a window, the
    /// items the window left idle at the bottom.
    pub(crate) fn pop(&mut self) -> (Option<T>, Vec<T>) {
        let item = self.items.pop();
        self.low_water = self.low_water.min(self.items.len());
        self.pops += 1;
        if self.pops < self.window {
            return (item, Vec::new());
        }
        let idle: Vec<T> = self.items.drain(..self.low_water).collect();
        self.pops = 0;
        self.low_water = self.items.len();
        (item, idle)
    }

    /// How many items the stack holds.
    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }

    /// Take every item out of the stack.
    pub(crate) fn drain(&mut self) -> Vec<T> {
        self.low_water = 0;
        core::mem::take(&mut self.items)
    }
}

#[cfg(test)]
mod tests {
    use super::TrimmedStack;
    use alloc::vec::Vec;

    /// Take `n` items out at once, making the ones the stack lacks, then put
    /// them all back: `n` encoders in flight together.
    fn in_flight(stack: &mut TrimmedStack<u32>, n: u32, made: &mut u32, dropped: &mut Vec<u32>) {
        let mut out = Vec::new();
        for _ in 0..n {
            let (item, idle) = stack.pop();
            dropped.extend(idle);
            out.push(item.unwrap_or_else(|| {
                *made += 1;
                *made
            }));
        }
        for item in out {
            stack.push(item);
        }
    }

    #[test]
    fn a_burst_is_given_back_once_a_window_passes_without_one() {
        let window = 64;
        let mut stack = TrimmedStack::new(window);
        let (mut made, mut dropped) = (0, Vec::new());

        in_flight(&mut stack, 500, &mut made, &mut dropped);
        assert_eq!(stack.len(), 500, "the burst's items are all free after it");

        for _ in 0..(2 * window) {
            in_flight(&mut stack, 3, &mut made, &mut dropped);
        }
        assert_eq!(stack.len(), 3, "steady work three at a time holds three");
        assert_eq!(
            dropped.len(),
            497,
            "the burst's surplus was handed back to be dropped"
        );
        assert_eq!(
            made, 500,
            "the steady work after the burst made nothing new"
        );
    }

    #[test]
    fn repeated_bursts_hold_the_stack_at_the_steady_load() {
        let window = 64;
        let mut stack = TrimmedStack::new(window);
        let (mut made, mut dropped) = (0, Vec::new());

        let mut held = Vec::new();
        for _ in 0..20 {
            in_flight(&mut stack, 200, &mut made, &mut dropped);
            for _ in 0..(2 * window) {
                in_flight(&mut stack, 2, &mut made, &mut dropped);
            }
            held.push(stack.len());
        }
        assert!(
            held.iter().all(|&n| n == 2),
            "every round settles at two: {held:?}"
        );
        assert_eq!(
            stack.len() + dropped.len(),
            made as usize,
            "every item made is either held or handed back"
        );
    }

    #[test]
    fn items_in_use_every_window_are_kept() {
        let window = 16;
        let mut stack = TrimmedStack::new(window);
        let (mut made, mut dropped) = (0, Vec::new());
        for _ in 0..(10 * window) {
            in_flight(&mut stack, 8, &mut made, &mut dropped);
        }
        assert_eq!(made, 8, "eight at a time needs eight, made once");
        assert!(
            dropped.is_empty(),
            "nothing in steady use is given back: {dropped:?}"
        );
        assert_eq!(stack.len(), 8);
    }
}
