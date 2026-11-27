use wgpu::Backends;
use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([VULKAN_MEMORY_BUDGET, DX12_MEMORY_BUDGET]);
}

const MIB: u64 = 1024 * 1024;

fn print_heaps(label: &str, report: &wgpu::MemoryBudgetReport) {
    for (index, heap) in report.heaps.iter().enumerate() {
        eprintln!(
            "memory_budget {label} heap {index}: device_local={} size={} MiB budget={:?} MiB usage={:?} MiB",
            heap.device_local,
            heap.size / MIB,
            heap.budget.map(|bytes| bytes / MIB),
            heap.usage.map(|bytes| bytes / MIB),
        );
    }
    if let Some(reason) = &report.budget_unavailable {
        eprintln!("memory_budget {label} budget unavailable: {reason}");
    }
}

#[gpu_test]
static VULKAN_MEMORY_BUDGET: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().skip(FailureCase::backend(!Backends::VULKAN)))
    .run_sync(|ctx| {
        let first = ctx
            .device
            .memory_budget()
            .expect("a Vulkan device states its memory heaps");
        print_heaps("first", &first);

        assert!(
            first
                .heaps
                .iter()
                .any(|heap| heap.device_local && heap.size > 0),
            "{first:?}"
        );

        let extension_enabled = unsafe {
            ctx.device
                .as_hal::<wgpu::hal::vulkan::Api>()
                .expect("a Vulkan device")
                .enabled_device_extensions()
                .iter()
                .any(|name| *name == c"VK_EXT_memory_budget")
        };

        if extension_enabled {
            assert_eq!(first.budget_unavailable, None);
        } else {
            assert!(first.budget_unavailable.is_some(), "{first:?}");
        }
        for heap in &first.heaps {
            if extension_enabled {
                let budget = heap.budget.expect("VK_EXT_memory_budget states a budget");
                let usage = heap.usage.expect("VK_EXT_memory_budget states a usage");
                assert!(usage <= heap.size, "usage {usage} above size {}", heap.size);
                // NVIDIA's driver states a budget above the size of its host heap.
                if heap.device_local {
                    assert!(
                        budget <= heap.size,
                        "budget {budget} above size {}",
                        heap.size
                    );
                }
            } else {
                assert_eq!(heap.budget, None);
                assert_eq!(heap.usage, None);
            }
        }

        let second = ctx.device.memory_budget().unwrap();
        print_heaps("second", &second);
        assert_eq!(first.heaps.len(), second.heaps.len());
        for (a, b) in first.heaps.iter().zip(&second.heaps) {
            assert_eq!(a.size, b.size);
            assert_eq!(a.device_local, b.device_local);
        }
    });

#[gpu_test]
static DX12_MEMORY_BUDGET: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().skip(FailureCase::backend(!Backends::DX12)))
    .run_sync(|ctx| {
        let report = ctx
            .device
            .memory_budget()
            .expect("a DX12 device states its memory heaps");
        print_heaps("dx12", &report);

        assert!(
            report
                .heaps
                .iter()
                .any(|heap| heap.device_local && heap.size > 0),
            "{report:?}"
        );
        assert_eq!(report.budget_unavailable, None, "{report:?}");
        for heap in &report.heaps {
            let budget = heap.budget.expect("DX12 states a budget");
            let usage = heap.usage.expect("DX12 states a usage");
            if heap.device_local {
                assert!(usage <= heap.size, "usage {usage} above size {}", heap.size);
                assert!(
                    budget <= heap.size,
                    "budget {budget} above size {}",
                    heap.size
                );
            }
        }
    });
