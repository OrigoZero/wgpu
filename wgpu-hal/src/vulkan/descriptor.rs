use alloc::vec::Vec;
use arrayvec::ArrayVec;
use ash::vk;
use hashbrown::{HashMap, HashSet};

/// The descriptor pool calls [`DescriptorAllocator`] makes on a device.
///
/// `ash::Device` answers them with the driver's own calls; the allocator's
/// tests answer them the way a driver does, including the failures a driver
/// returns from a pool that still has room.
pub trait PoolDevice {
    /// `vkCreateDescriptorPool`.
    unsafe fn create_pool(
        &self,
        info: &vk::DescriptorPoolCreateInfo<'_>,
    ) -> Result<vk::DescriptorPool, vk::Result>;
    /// `vkDestroyDescriptorPool`.
    unsafe fn destroy_pool(&self, pool: vk::DescriptorPool);
    /// `vkResetDescriptorPool`, which frees every set the pool holds.
    unsafe fn reset_pool(&self, pool: vk::DescriptorPool) -> Result<(), vk::Result>;
    /// `vkAllocateDescriptorSets` of the one set `info` names.
    unsafe fn allocate_set(
        &self,
        info: &vk::DescriptorSetAllocateInfo<'_>,
    ) -> Result<vk::DescriptorSet, vk::Result>;
    /// `vkFreeDescriptorSets` of one set.
    unsafe fn free_set(
        &self,
        pool: vk::DescriptorPool,
        set: vk::DescriptorSet,
    ) -> Result<(), vk::Result>;
}

impl PoolDevice for ash::Device {
    unsafe fn create_pool(
        &self,
        info: &vk::DescriptorPoolCreateInfo<'_>,
    ) -> Result<vk::DescriptorPool, vk::Result> {
        unsafe { self.create_descriptor_pool(info, None) }
    }

    unsafe fn destroy_pool(&self, pool: vk::DescriptorPool) {
        unsafe { self.destroy_descriptor_pool(pool, None) }
    }

    unsafe fn reset_pool(&self, pool: vk::DescriptorPool) -> Result<(), vk::Result> {
        unsafe { self.reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty()) }
    }

    unsafe fn allocate_set(
        &self,
        info: &vk::DescriptorSetAllocateInfo<'_>,
    ) -> Result<vk::DescriptorSet, vk::Result> {
        unsafe { self.allocate_descriptor_sets(info) }.map(|sets| sets[0])
    }

    unsafe fn free_set(
        &self,
        pool: vk::DescriptorPool,
        set: vk::DescriptorSet,
    ) -> Result<(), vk::Result> {
        unsafe { self.free_descriptor_sets(pool, core::slice::from_ref(&set)) }
    }
}

const POOL_MIN_SETS: u32 = 64;
const POOL_MAX_SETS: u32 = 512;

/// The dynamic counts are tracked separately since dynamic textures/buffers
/// have distinct descriptor types compared to their non-dynamic counterparts.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct DescriptorCounts {
    pub sampler: u32,
    pub sampled_image: u32,
    pub storage_image: u32,
    pub uniform_buffer: u32,
    pub uniform_buffer_dynamic: u32,
    pub storage_buffer: u32,
    pub storage_buffer_dynamic: u32,
    pub acceleration_structure: u32,
}

impl DescriptorCounts {
    fn total(&self) -> u32 {
        self.sampler
            + self.sampled_image
            + self.storage_image
            + self.uniform_buffer
            + self.uniform_buffer_dynamic
            + self.storage_buffer
            + self.storage_buffer_dynamic
            + self.acceleration_structure
    }
}

#[derive(Debug)]
pub struct DescriptorSet {
    raw: vk::DescriptorSet,
    bucket_key: BucketKey,
    pool_index: usize,
}

impl DescriptorSet {
    pub fn raw(&self) -> vk::DescriptorSet {
        self.raw
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct BucketKey {
    counts: DescriptorCounts,
    update_after_bind: bool,
}

struct Pool {
    raw: vk::DescriptorPool,
    capacity: u32,
    available: u32,
    /// Set when the driver refused to allocate from this pool despite it
    /// having free capacity, making `alloc` skip it until the pool is empty
    /// again and can be reset.
    fragmented: bool,
}

/// Keeps track of all pools created with this bucket's [`BucketKey`].
#[derive(Default)]
struct Bucket {
    /// We keep track of all descriptor set layouts that might use this bucket.
    /// If the set becomes empty, the bucket is destroyed.
    layouts: HashSet<vk::DescriptorSetLayout>,
    pools: Vec<Pool>,
    available_sets: u32,
    allocated_sets: u32,
}

impl Bucket {
    fn create_pool<D: PoolDevice>(
        &mut self,
        device: &D,
        key: &BucketKey,
        capacity_hint: u32,
    ) -> Result<(usize, &mut Pool), crate::DeviceError> {
        let index = self.pools.len();
        let pool = create_descriptor_pool(device, key, capacity_hint)?;
        self.available_sets += pool.capacity;
        self.pools.push(pool);
        Ok((index, self.pools.last_mut().unwrap()))
    }
}

pub struct DescriptorAllocator {
    buckets: HashMap<BucketKey, Bucket>,
    max_update_after_bind_descriptors_in_all_pools: u32,
    update_after_bind_descriptors_in_all_pools: u32,
}

impl super::BindGroupLayout {
    fn bucket_key(&self) -> BucketKey {
        let update_after_bind = self.contains_binding_arrays;
        let counts = self.desc_count.clone();
        BucketKey {
            counts,
            update_after_bind,
        }
    }
}

impl DescriptorAllocator {
    pub fn new(max_update_after_bind_descriptors_in_all_pools: u32) -> Self {
        DescriptorAllocator {
            buckets: HashMap::default(),
            max_update_after_bind_descriptors_in_all_pools,
            update_after_bind_descriptors_in_all_pools: 0,
        }
    }

    pub fn register_layout<D: PoolDevice>(
        &mut self,
        device: &D,
        layout: &super::BindGroupLayout,
    ) -> Result<(), crate::DeviceError> {
        let key = layout.bucket_key();
        let bucket = match self.buckets.entry(key) {
            hashbrown::hash_map::Entry::Occupied(occupied_entry) => occupied_entry.into_mut(),
            hashbrown::hash_map::Entry::Vacant(vacant_entry) => {
                let mut bucket = Bucket::default();
                // Create at least 1 pool upfront instead of doing this when
                // creating the first bind group.
                bucket.create_pool(device, vacant_entry.key(), POOL_MIN_SETS)?;
                vacant_entry.insert(bucket)
            }
        };

        assert!(bucket.layouts.insert(layout.raw));

        Ok(())
    }

    pub fn unregister_layout<D: PoolDevice>(
        &mut self,
        device: &D,
        layout: &super::BindGroupLayout,
    ) {
        let key = layout.bucket_key();
        let bucket = self.buckets.get_mut(&key).unwrap();

        assert!(bucket.layouts.remove(&layout.raw));

        if bucket.layouts.is_empty() {
            // Remove the bucket and destroy any remaining pools.
            let bucket = self.buckets.remove(&key).unwrap();
            for pool in bucket.pools {
                assert_eq!(
                    pool.available, pool.capacity,
                    "pool is not empty, at least one DescriptorSet has not been freed"
                );
                unsafe { device.destroy_pool(pool.raw) };
            }
        }
    }

    pub unsafe fn alloc<D: PoolDevice>(
        &mut self,
        device: &D,
        layout: &super::BindGroupLayout,
    ) -> Result<DescriptorSet, crate::DeviceError> {
        let update_after_bind = layout.contains_binding_arrays;
        let total_descriptors = layout.desc_count.total();

        if update_after_bind
            && self.max_update_after_bind_descriptors_in_all_pools
                - self.update_after_bind_descriptors_in_all_pools
                < total_descriptors
        {
            return Err(crate::DeviceError::OutOfMemory);
        }

        let key = layout.bucket_key();
        let bucket = self.buckets.get_mut(&key).unwrap();
        let capacity_hint = bucket.allocated_sets;

        // Prefer smaller/older/fuller pools for new allocations to prevent
        // fragmentation and possibly fragmentation of hardware resources
        // (VK_ERROR_FRAGMENTATION)
        let mut candidate = bucket
            .pools
            .iter()
            .position(|pool| !pool.fragmented && pool.available != 0);

        let (raw, pool_index) = loop {
            let index = match candidate {
                Some(index) => index,
                None => bucket.create_pool(device, &key, capacity_hint)?.0,
            };
            let created_pool = candidate.is_none();

            let vk_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(bucket.pools[index].raw)
                .set_layouts(core::slice::from_ref(&layout.raw));

            // From https://docs.vulkan.org/refpages/latest/refpages/source/VkDescriptorPoolCreateInfo.html#_description
            // > if all sets allocated from the pool since it was created or
            // > most recently reset use the same number of descriptors
            // > (of each type) and the requested allocation also uses that
            // > same number of descriptors (of each type), then
            // > fragmentation must not cause an allocation failure
            //
            // These pools satisfy that condition, but some drivers (e.g.
            // Mali) fail such allocations anyway once a pool has served
            // a number of allocations; freeing sets does not make the
            // pool usable again, only resetting it does.
            //
            // Recover instead of failing: reset a pool that has no live sets,
            // otherwise mark the pool `fragmented` so that allocations skip
            // it. A fragmented pool is reset once its last set is freed.
            //
            // See https://github.com/gfx-rs/wgpu/pull/10264
            match unsafe { device.allocate_set(&vk_info) } {
                Ok(set) => break (set, index),
                Err(vk::Result::ERROR_FRAGMENTED_POOL | vk::Result::ERROR_OUT_OF_POOL_MEMORY) => {
                    if created_pool {
                        // A fresh pool failing to allocate even a single set leaves
                        // no pool left to fall back to, so drop it and give up.
                        let pool = bucket.pools.pop().unwrap();
                        bucket.available_sets -= pool.capacity;
                        unsafe { device.destroy_pool(pool.raw) };
                        log::error!(
                            "descriptor set allocation from a fresh pool failed unexpectedly"
                        );
                        return Err(crate::DeviceError::OutOfMemory);
                    }

                    // Resetting a pool restores the spec guarantee that
                    // fragmentation cannot cause an allocation failure, but it
                    // implicitly frees all of the pool's sets, so only reset a
                    // pool that has no live sets.
                    let pool = &bucket.pools[index];
                    if pool.available == pool.capacity {
                        let reset_result = unsafe { device.reset_pool(pool.raw) };
                        if reset_result.is_ok() {
                            match unsafe { device.allocate_set(&vk_info) } {
                                Ok(set) => break (set, index),
                                Err(
                                    vk::Result::ERROR_FRAGMENTED_POOL
                                    | vk::Result::ERROR_OUT_OF_POOL_MEMORY,
                                ) => {}
                                Err(err) => return Err(super::map_host_device_oom_err(err)),
                            }
                        }
                    }

                    let available = bucket.pools[index].available;
                    let capacity = bucket.pools[index].capacity;
                    bucket.pools[index].fragmented = true;
                    bucket.available_sets -= available;
                    log::debug!(
                        "Retiring fragmented descriptor pool ({available}/{capacity} sets free), retrying on another pool"
                    );
                    candidate = bucket
                        .pools
                        .iter()
                        .position(|pool| !pool.fragmented && pool.available != 0);
                }
                Err(err) => return Err(super::map_host_device_oom_err(err)),
            }
        };

        let pool = &mut bucket.pools[pool_index];
        pool.available -= 1;
        bucket.available_sets -= 1;
        bucket.allocated_sets += 1;
        if update_after_bind {
            self.update_after_bind_descriptors_in_all_pools += total_descriptors;
        }

        Ok(DescriptorSet {
            raw,
            bucket_key: key,
            pool_index,
        })
    }

    pub unsafe fn free<D: PoolDevice>(&mut self, device: &D, set: DescriptorSet) {
        let bucket = self.buckets.get_mut(&set.bucket_key).unwrap();
        let pool = bucket.pools.get_mut(set.pool_index).unwrap();

        let result = unsafe { device.free_set(pool.raw, set.raw()) };
        if let Err(err) = result {
            // vkFreeDescriptorSets is documented to return:
            // - VK_ERROR_UNKNOWN
            // - VK_ERROR_VALIDATION_FAILED (we shouldn't encounter this one)
            // wgpu-hal doesn't currently report errors in destroy functions.
            // Panic here for now. It might be ok to ignore the error but the
            // proper solution would probably be to lose the device.
            panic!("vkFreeDescriptorSets error: {err}, please report this error");
        }

        pool.available += 1;

        if !pool.fragmented {
            bucket.available_sets += 1;
        } else if pool.available == pool.capacity {
            // A fragmented pool's free sets are unusable until it is reset, so
            // they do not count towards `available_sets`.
            // On Mali, freeing sets does not make such a pool usable again;
            // resetting it is what does, and that is only safe once no sets are live.
            let reset_result = unsafe { device.reset_pool(pool.raw) };
            if reset_result.is_ok() {
                pool.fragmented = false;
                bucket.available_sets += pool.capacity;
            }
        }
        bucket.allocated_sets -= 1;
        if set.bucket_key.update_after_bind {
            self.update_after_bind_descriptors_in_all_pools -= set.bucket_key.counts.total();
        }

        // Do not immediately destroy empty pools since they might have
        // been recently created.
        // Destroy a pool only if it's empty and we have 1/4th its capacity
        // in other pools.
        // Note that this logic will never destroy the last pool.
        let last = bucket.pools.last().unwrap();
        let last_available = last.available;
        let last_capacity = last.capacity;
        // Only a non-fragmented pool contributes its capacity to
        // `available_sets`.
        let last_contribution = if last.fragmented { 0 } else { last_capacity };
        if last_available == last_capacity
            && bucket.available_sets - last_contribution > last_capacity / 4
        {
            let pool = bucket.pools.pop().unwrap();
            unsafe { device.destroy_pool(pool.raw) };
            bucket.available_sets -= last_contribution;
        }
    }
}

impl Drop for DescriptorAllocator {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            assert!(
                self.buckets.is_empty(),
                "buckets are not empty, at least one BGL has not been unregistered"
            )
        }
    }
}

fn create_descriptor_pool<D: PoolDevice>(
    device: &D,
    key: &BucketKey,
    capacity_hint: u32,
) -> Result<Pool, crate::DeviceError> {
    let counts = &key.counts;

    const NR_OF_DESCRIPTOR_TYPES: usize = 8;

    use vk::DescriptorType as Dt;
    let counts: [_; NR_OF_DESCRIPTOR_TYPES] = [
        (Dt::SAMPLER, counts.sampler),
        (Dt::SAMPLED_IMAGE, counts.sampled_image),
        (Dt::STORAGE_IMAGE, counts.storage_image),
        (Dt::UNIFORM_BUFFER, counts.uniform_buffer),
        (Dt::UNIFORM_BUFFER_DYNAMIC, counts.uniform_buffer_dynamic),
        (Dt::STORAGE_BUFFER, counts.storage_buffer),
        (Dt::STORAGE_BUFFER_DYNAMIC, counts.storage_buffer_dynamic),
        (
            Dt::ACCELERATION_STRUCTURE_KHR,
            counts.acceleration_structure,
        ),
    ];

    // bounded doubling growth strategy
    let mut capacity = capacity_hint
        .clamp(POOL_MIN_SETS, POOL_MAX_SETS)
        .next_power_of_two();

    // Lower capacity to avoid overflowing `count * capacity` calculations
    // below. The resulting `capacity` might not be a power of 2 but that's
    // ok.
    for (_, count) in counts {
        capacity = (u32::MAX / count.max(1)).min(capacity);
    }

    let pool_sizes = counts
        .into_iter()
        .filter(|&(_, count)| count != 0)
        .map(|(ty, count)| vk::DescriptorPoolSize {
            ty,
            descriptor_count: count * capacity,
        })
        .collect::<ArrayVec<_, NR_OF_DESCRIPTOR_TYPES>>();

    // Prior to Vulkan 1.3.215 `poolSizeCount` was required to be non-zero, so
    // add a single small sampler entry if `pool_sizes` would otherwise be
    // empty.
    let mut pool_sizes = pool_sizes;
    if pool_sizes.is_empty() {
        pool_sizes.push(vk::DescriptorPoolSize {
            ty: Dt::SAMPLER,
            descriptor_count: 1,
        });
    }

    let mut flags = vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET;
    if key.update_after_bind {
        flags |= vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND;
    };

    let vk_info = vk::DescriptorPoolCreateInfo::default()
        .flags(flags)
        .max_sets(capacity)
        .pool_sizes(&pool_sizes);

    let raw = unsafe { device.create_pool(&vk_info) }
        .map_err(super::map_host_device_oom_and_fragmentation_err)?;

    Ok(Pool {
        raw,
        capacity,
        available: capacity,
        fragmented: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::collections::VecDeque;
    use ash::vk::Handle as _;
    use core::cell::{Cell, RefCell};

    /// A pool the way a driver that reuses no slot until a reset keeps one:
    /// it hands out its `max_sets` slots in order, a freed set's slot stays
    /// spent, and once every slot was handed out it answers
    /// `ERROR_FRAGMENTED_POOL` while it holds fewer than `max_sets` live sets
    /// (the Mali driver of the Galaxy Tab S11 answers this way after a pool
    /// has served its sets, though the sets all have one shape).
    struct BumpPool {
        max_sets: u32,
        handed_out: u32,
        live: HashSet<u64>,
    }

    #[derive(Default)]
    struct BumpDriver {
        pools: RefCell<HashMap<u64, BumpPool>>,
        next_handle: Cell<u64>,
        /// Every allocation fails with `ERROR_FRAGMENTED_POOL`.
        refuse_all: bool,
        fragmented_answers: Cell<u32>,
        resets: Cell<u32>,
        created: Cell<u32>,
        destroyed: Cell<u32>,
    }

    impl BumpDriver {
        fn handle(&self) -> u64 {
            self.next_handle.set(self.next_handle.get() + 1);
            self.next_handle.get()
        }

        fn live_sets(&self) -> usize {
            self.pools.borrow().values().map(|p| p.live.len()).sum()
        }
    }

    impl PoolDevice for BumpDriver {
        unsafe fn create_pool(
            &self,
            info: &vk::DescriptorPoolCreateInfo<'_>,
        ) -> Result<vk::DescriptorPool, vk::Result> {
            assert!(
                info.pool_size_count > 0,
                "a pool states at least one pool size"
            );
            let raw = self.handle();
            self.pools.borrow_mut().insert(
                raw,
                BumpPool {
                    max_sets: info.max_sets,
                    handed_out: 0,
                    live: HashSet::default(),
                },
            );
            self.created.set(self.created.get() + 1);
            Ok(vk::DescriptorPool::from_raw(raw))
        }

        unsafe fn destroy_pool(&self, pool: vk::DescriptorPool) {
            let pool = self
                .pools
                .borrow_mut()
                .remove(&pool.as_raw())
                .expect("a live pool");
            assert!(pool.live.is_empty(), "a pool is destroyed with live sets");
            self.destroyed.set(self.destroyed.get() + 1);
        }

        unsafe fn reset_pool(&self, pool: vk::DescriptorPool) -> Result<(), vk::Result> {
            let mut pools = self.pools.borrow_mut();
            let pool = pools.get_mut(&pool.as_raw()).expect("a live pool");
            assert!(
                pool.live.is_empty(),
                "a pool is reset while a set it holds is in use"
            );
            pool.handed_out = 0;
            self.resets.set(self.resets.get() + 1);
            Ok(())
        }

        unsafe fn allocate_set(
            &self,
            info: &vk::DescriptorSetAllocateInfo<'_>,
        ) -> Result<vk::DescriptorSet, vk::Result> {
            let raw = self.handle();
            let mut pools = self.pools.borrow_mut();
            let pool = pools
                .get_mut(&info.descriptor_pool.as_raw())
                .expect("a live pool");
            if pool.live.len() as u32 == pool.max_sets {
                return Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY);
            }
            if self.refuse_all || pool.handed_out == pool.max_sets {
                self.fragmented_answers
                    .set(self.fragmented_answers.get() + 1);
                return Err(vk::Result::ERROR_FRAGMENTED_POOL);
            }
            pool.handed_out += 1;
            pool.live.insert(raw);
            Ok(vk::DescriptorSet::from_raw(raw))
        }

        unsafe fn free_set(
            &self,
            pool: vk::DescriptorPool,
            set: vk::DescriptorSet,
        ) -> Result<(), vk::Result> {
            let mut pools = self.pools.borrow_mut();
            let pool = pools.get_mut(&pool.as_raw()).expect("a live pool");
            assert!(
                pool.live.remove(&set.as_raw()),
                "a set is freed from a pool that does not hold it"
            );
            Ok(())
        }
    }

    fn storage_buffer_layout(raw: u64) -> super::super::BindGroupLayout {
        super::super::BindGroupLayout {
            raw: vk::DescriptorSetLayout::from_raw(raw),
            desc_count: DescriptorCounts {
                storage_buffer: 1,
                ..Default::default()
            },
            entries: Box::new([]),
            binding_map: Vec::new(),
            contains_binding_arrays: false,
        }
    }

    fn pools_of(allocator: &DescriptorAllocator, layout: &super::super::BindGroupLayout) -> usize {
        allocator.buckets[&layout.bucket_key()].pools.len()
    }

    /// Bind groups made and dropped frame after frame, a window of them live
    /// at once, keep allocating on a driver whose pools reuse no slot until
    /// a reset, and the pools they take stay bounded.
    #[test]
    fn churn_on_pools_that_reuse_no_slot_until_reset_keeps_allocating() {
        let driver = BumpDriver::default();
        let mut allocator = DescriptorAllocator::new(0);
        let layout = storage_buffer_layout(1);
        allocator.register_layout(&driver, &layout).unwrap();

        const LIVE: usize = 48;
        let mut live = VecDeque::new();
        let mut most_pools = 0;
        for _ in 0..20_000 {
            let set = unsafe { allocator.alloc(&driver, &layout) }
                .expect("a set is allocated while sets are freed as fast as they are made");
            live.push_back(set);
            if live.len() > LIVE {
                unsafe { allocator.free(&driver, live.pop_front().unwrap()) };
            }
            most_pools = most_pools.max(pools_of(&allocator, &layout));
        }

        assert!(
            driver.fragmented_answers.get() > 0,
            "the driver answered ERROR_FRAGMENTED_POOL"
        );
        assert!(
            driver.resets.get() > 0,
            "a spent pool is reset once it is empty"
        );
        assert!(most_pools <= 4, "the pools stay bounded, held {most_pools}");
        assert_eq!(driver.live_sets(), LIVE);

        for set in live.drain(..) {
            unsafe { allocator.free(&driver, set) };
        }
        allocator.unregister_layout(&driver, &layout);
        assert_eq!(
            driver.created.get(),
            driver.destroyed.get(),
            "every pool is destroyed"
        );
    }

    /// A set that stays live keeps its pool from being reset, and allocation
    /// goes on in other pools.
    #[test]
    fn a_long_lived_set_does_not_stop_allocation() {
        let driver = BumpDriver::default();
        let mut allocator = DescriptorAllocator::new(0);
        let layout = storage_buffer_layout(1);
        allocator.register_layout(&driver, &layout).unwrap();

        let pinned = unsafe { allocator.alloc(&driver, &layout) }.unwrap();
        let mut most_pools = 0;
        for _ in 0..10_000 {
            let set = unsafe { allocator.alloc(&driver, &layout) }
                .expect("a set is allocated beside a long-lived one");
            unsafe { allocator.free(&driver, set) };
            most_pools = most_pools.max(pools_of(&allocator, &layout));
        }
        assert!(most_pools <= 3, "the pools stay bounded, held {most_pools}");

        unsafe { allocator.free(&driver, pinned) };
        allocator.unregister_layout(&driver, &layout);
        assert_eq!(
            driver.created.get(),
            driver.destroyed.get(),
            "every pool is destroyed"
        );
        assert_eq!(driver.live_sets(), 0);
    }

    /// A driver that refuses even a fresh pool's first set answers an error
    /// to the caller, and the allocator is left able to unregister.
    #[test]
    fn a_pool_that_refuses_its_first_set_answers_an_error() {
        let driver = BumpDriver {
            refuse_all: true,
            ..Default::default()
        };
        let mut allocator = DescriptorAllocator::new(0);
        let layout = storage_buffer_layout(1);
        allocator.register_layout(&driver, &layout).unwrap();

        for _ in 0..2 {
            let result = unsafe { allocator.alloc(&driver, &layout) };
            assert!(matches!(result, Err(crate::DeviceError::OutOfMemory)));
        }
        assert_eq!(pools_of(&allocator, &layout), 1);

        allocator.unregister_layout(&driver, &layout);
        assert_eq!(
            driver.created.get(),
            driver.destroyed.get(),
            "every pool is destroyed"
        );
    }

    /// A layout with no bindings still makes a pool the driver accepts.
    #[test]
    fn an_empty_layout_makes_a_pool_with_a_pool_size() {
        let driver = BumpDriver::default();
        let mut allocator = DescriptorAllocator::new(0);
        let layout = super::super::BindGroupLayout {
            desc_count: DescriptorCounts::default(),
            ..storage_buffer_layout(2)
        };
        allocator.register_layout(&driver, &layout).unwrap();
        let set = unsafe { allocator.alloc(&driver, &layout) }.unwrap();
        unsafe { allocator.free(&driver, set) };
        allocator.unregister_layout(&driver, &layout);
        assert_eq!(driver.created.get(), driver.destroyed.get());
    }
}
