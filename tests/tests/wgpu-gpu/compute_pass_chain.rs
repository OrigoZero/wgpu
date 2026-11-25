//! Dispatches that each read what the dispatch before them wrote, spread over
//! many compute passes and several command encoders, end with exactly the
//! values the chain computes.
//!
//! Every dispatch adds one to each element of one buffer and copies the
//! result into the next buffer of a ring, so each dispatch depends on the one
//! before it, within a pass and across the end of one pass and the start of
//! the next. Passes take their per-dispatch trackers from the device's pool
//! and hand them back emptied, and a pass's buffer transitions are made as one
//! memory barrier; a transition left out, or a tracker handed on holding a
//! state from an earlier pass, shows as a wrong count.

use wgpu::util::DeviceExt as _;
use wgpu_test::{gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(COMPUTE_PASS_CHAIN);
}

const ELEMENTS: u32 = 256;
const RING: usize = 4;

const SHADER_SRC: &str = "
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    dst[id.x] = src[id.x] + 1u;
}
";

#[gpu_test]
static COMPUTE_PASS_CHAIN: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().test_features_limits())
    .run_async(compute_pass_chain);

async fn compute_pass_chain(ctx: TestingContext) {
    let size = u64::from(ELEMENTS) * 4;
    let initial: Vec<u32> = (0..ELEMENTS).collect();
    let buffers: Vec<wgpu::Buffer> = (0..RING)
        .map(|i| {
            ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("chain {i}")),
                contents: bytemuck::cast_slice(&initial),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        })
        .collect();
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let layout = pipeline.get_bind_group_layout(0);
    // Group `i` reads buffer `i` and writes buffer `i + 1` of the ring.
    let groups: Vec<wgpu::BindGroup> = (0..RING)
        .map(|i| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buffers[i].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buffers[(i + 1) % RING].as_entire_binding() },
                ],
            })
        })
        .collect();

    // Passes of 1, 2, 3 and 7 dispatches, in three encoders: every pass
    // boundary falls between two dependent dispatches.
    let encoders: [&[u32]; 3] = [&[1, 2, 3], &[7, 1], &[2, 3, 7, 1, 1]];
    let mut step = 0usize;
    for passes in encoders {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for &dispatches in passes {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            for _ in 0..dispatches {
                pass.set_bind_group(0, &groups[step % RING], &[]);
                pass.dispatch_workgroups(ELEMENTS.div_ceil(64), 1, 1);
                step += 1;
            }
        }
        ctx.queue.submit([encoder.finish()]);
    }

    // The last dispatch wrote buffer `step % RING`, holding `initial + step`.
    let last = &buffers[step % RING];
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(last, 0, &readback, 0, size);
    ctx.queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::PollType::wait_indefinitely()).await.unwrap();
    let data = readback.slice(..).get_mapped_range().unwrap();
    let values: &[u32] = bytemuck::cast_slice(&data);
    for (i, &value) in values.iter().enumerate() {
        assert_eq!(value, i as u32 + step as u32, "element {i} after {step} chained dispatches");
    }
}
