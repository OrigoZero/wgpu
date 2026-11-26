//! A compute pass records its barriers inline, against the state the
//! commands before it left each resource in, and a pass reading a surface a
//! render pass discarded reads it cleared.
//!
//! One texture passes between a render pass and compute passes in one
//! encoder: written as a render attachment, then sampled; written as storage,
//! then sampled; written as an attachment and discarded, then sampled. Every
//! read lands in a buffer of its own, so a transition left out (a layout the
//! texture is not in, a write not yet visible) or a discarded surface left
//! uncleared shows as a wrong value. Done twice.

use wgpu_test::{
    gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(COMPUTE_PASS_INLINE_BARRIERS);
}

const STORE_SRC: &str = "
@group(0) @binding(0) var dst: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(1) fn main() {
    textureStore(dst, vec2<i32>(0, 0), vec4<u32>(7u, 0u, 0u, 0u));
}
";

const LOAD_SRC: &str = "
@group(0) @binding(0) var src: texture_2d<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(1) fn main() {
    out[0] = textureLoad(src, vec2<i32>(0, 0), 0).x + 1u;
}
";

#[gpu_test]
static COMPUTE_PASS_INLINE_BARRIERS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().test_features_limits())
    .run_async(compute_pass_inline_barriers);

fn pipeline(ctx: &TestingContext, source: &str) -> wgpu::ComputePipeline {
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    ctx.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
}

fn clear(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    value: f64,
    store: wgpu::StoreOp,
) {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("attachment write"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: value,
                    g: 0.0,
                    b: 0.0,
                    a: 0.0,
                }),
                store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}

fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    group: &wgpu::BindGroup,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(1, 1, 1);
}

async fn compute_pass_inline_barriers(ctx: TestingContext) {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("passed between passes"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    const READS: usize = 4;
    let out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reads"),
        size: 256 * READS as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let store = pipeline(&ctx, STORE_SRC);
    let load = pipeline(&ctx, LOAD_SRC);
    let store_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &store.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });
    // Read `i` lands at offset 256 * i of `out`.
    let load_groups: Vec<wgpu::BindGroup> = (0..READS)
        .map(|i| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &load.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &out,
                            offset: 256 * i as u64,
                            size: wgpu::BufferSize::new(4),
                        }),
                    },
                ],
            })
        })
        .collect();
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    for round in 1..=2 {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        // Written as an attachment, then sampled: 5 + 1.
        clear(&mut encoder, &view, 5.0, wgpu::StoreOp::Store);
        dispatch(&mut encoder, &load, &load_groups[0]);
        // Written as storage, then sampled: 7 + 1.
        dispatch(&mut encoder, &store, &store_group);
        dispatch(&mut encoder, &load, &load_groups[1]);
        // Written as an attachment and discarded, then sampled: the
        // discarded surface reads cleared, 0 + 1, and stays cleared.
        clear(&mut encoder, &view, 9.0, wgpu::StoreOp::Discard);
        dispatch(&mut encoder, &load, &load_groups[2]);
        dispatch(&mut encoder, &load, &load_groups[3]);
        encoder.copy_buffer_to_buffer(&out, 0, &readback, 0, out.size());
        ctx.queue.submit([encoder.finish()]);

        readback.slice(..).map_async(wgpu::MapMode::Read, |_| ());
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        let values: Vec<u32> = {
            let data = readback.slice(..).get_mapped_range().unwrap();
            let words: &[u32] = bytemuck::cast_slice(&data);
            (0..READS).map(|i| words[64 * i]).collect()
        };
        readback.unmap();
        assert_eq!(
            values,
            [6, 8, 1, 1],
            "round {round}: each read saw the write before it"
        );
    }
}
