//! Tests for texture copy

use wgpu::*;
use wgpu_test::{fail, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        WRITE_TEXTURE_SUBSET_2D,
        WRITE_TEXTURE_SUBSET_3D,
        WRITE_TEXTURE_NO_OOB,
        WRITE_TEXTURE_VIA_STAGING_BUFFER,
        WRITE_TEXTURES_EVERY_MIP_LEVEL,
        WRITE_TEXTURES_INVALID_WRITE_WRITES_NONE,
        WRITE_TEXTURES_A_LATER_WRITE_LANDS_LAST,
    ]);
}

#[gpu_test]
static WRITE_TEXTURE_SUBSET_2D: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        // Deliberately non-square so a width/height swap is caught. `width` is
        // a multiple of `COPY_BYTES_PER_ROW_ALIGNMENT` so `bytes_per_row` (one
        // byte per R8Uint texel) stays aligned for `copy_texture_to_buffer`.
        let width = 256;
        let height = 192;

        let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let data = vec![1u8; width as usize * 2];
        // Write the first two rows
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height: 2,
                depth_or_array_layers: 1,
            },
        );

        ctx.queue.submit(None);

        let read_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (width * height) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &read_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        ctx.queue.submit(Some(encoder.finish()));

        let slice = read_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        let data: Vec<u8> = slice.get_mapped_range().unwrap().to_vec();

        for byte in &data[..(width as usize * 2)] {
            assert_eq!(*byte, 1);
        }
        for byte in &data[(width as usize * 2)..] {
            assert_eq!(*byte, 0);
        }
    });

#[gpu_test]
static WRITE_TEXTURE_SUBSET_3D: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        // Deliberately non-square (and a depth distinct from both) so any
        // width/height/depth swap is caught. `width` is a multiple of
        // `COPY_BYTES_PER_ROW_ALIGNMENT` so `bytes_per_row` stays aligned.
        let width = 256;
        let height = 192;
        let depth = 4;
        let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D3,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: depth,
            },
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let data = vec![1u8; (width * height) as usize * 2];
        // Write the first two slices
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 2,
            },
        );

        ctx.queue.submit(None);

        let read_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (width * height * depth) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &read_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: depth,
            },
        );

        ctx.queue.submit(Some(encoder.finish()));

        let slice = read_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        let data: Vec<u8> = slice.get_mapped_range().unwrap().to_vec();

        for byte in &data[..((width * height) as usize * 2)] {
            assert_eq!(*byte, 1);
        }
        for byte in &data[((width * height) as usize * 2)..] {
            assert_eq!(*byte, 0);
        }
    });

#[gpu_test]
static WRITE_TEXTURE_NO_OOB: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        // Non-square so a width/height swap is caught.
        let width = 256;
        let height = 192;

        let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::COPY_DST,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let data = vec![1u8; width as usize * 2 + 100]; // check that we don't attempt to copy OOB internally by adding 100 bytes here
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height: 2,
                depth_or_array_layers: 1,
            },
        );
    });

// Test a writeTexture operation that will use the staging buffer.
// If run with the address sanitizer, this serves as a regression
// test for https://github.com/gfx-rs/wgpu/pull/7893.
#[gpu_test]
static WRITE_TEXTURE_VIA_STAGING_BUFFER: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_async(|ctx| async move {
        let width = 89;
        let height = 17;

        let tex = ctx.device.create_texture(&TextureDescriptor {
            label: None,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            format: TextureFormat::R8Uint,
            usage: TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC
                | TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });

        let write_width: u32 = 31;
        let write_height: u32 = 5;
        let write_bytes_per_row: u32 = 113;
        let write_data = (0..(write_height - 1) * write_bytes_per_row + write_width)
            .map(|b| (b % 256) as u8)
            .collect::<Vec<_>>();

        ctx.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &write_data,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(write_bytes_per_row),
                rows_per_image: Some(19),
            },
            Extent3d {
                width: write_width,
                height: write_height,
                depth_or_array_layers: 1,
            },
        );

        ctx.queue.submit(None);

        let read_bytes_per_row = wgt::COPY_BYTES_PER_ROW_ALIGNMENT;
        let read_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: (height * read_bytes_per_row) as u64,
            usage: BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &read_buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(read_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        ctx.queue.submit(Some(encoder.finish()));

        let slice = read_buffer.slice(..);
        slice.map_async(MapMode::Read, |_| ());
        ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
        let read_data: Vec<u8> = slice.get_mapped_range().unwrap().to_vec();

        for x in 0..write_width {
            for y in 0..write_height {
                assert_eq!(
                    read_data[(y * read_bytes_per_row + x) as usize],
                    write_data[(y * write_bytes_per_row + x) as usize]
                );
            }
        }
    });

/// The texture both `write_textures` tests write: 64x16 RGBA8, its whole
/// chain of 7 levels, two layers.
fn mip_chain_texture(device: &Device) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some("write_textures chain"),
        size: Extent3d {
            width: 64,
            height: 16,
            depth_or_array_layers: 2,
        },
        mip_level_count: 7,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::COPY_DST | TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Every texel of mip `level` of layer `layer` of `texture`, read back.
async fn read_level(ctx: &wgpu_test::TestingContext, texture: &Texture, layer: u32, level: u32) -> Vec<u8> {
    let (width, height) = ((64u32 >> level).max(1), (16u32 >> level).max(1));
    let bytes_per_row = COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: u64::from(bytes_per_row * height),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture,
            mip_level: level,
            origin: Origin3d { x: 0, y: 0, z: layer },
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit(Some(encoder.finish()));
    let slice = buffer.slice(..);
    slice.map_async(MapMode::Read, |_| ());
    ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    let texels = (0..height as usize)
        .flat_map(|row| {
            let start = row * bytes_per_row as usize;
            mapped[start..start + width as usize * 4].to_vec()
        })
        .collect();
    drop(mapped);
    buffer.unmap();
    texels
}

/// One `write_textures` call writes a whole mip chain of one layer, each
/// level from its own bytes (rows of a level narrower than the copy
/// alignment included), and leaves the other layer as it was.
#[gpu_test]
static WRITE_TEXTURES_EVERY_MIP_LEVEL: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let texture = mip_chain_texture(&ctx.device);
        let levels: Vec<Vec<u8>> = (0..7u32)
            .map(|level| {
                let texels = ((64u32 >> level).max(1) * (16u32 >> level).max(1)) as usize;
                (0..texels * 4).map(|i| (i as u8).wrapping_mul(7).wrapping_add(level as u8 * 31)).collect()
            })
            .collect();
        let writes: Vec<TextureWrite<'_>> = (0u32..)
            .zip(&levels)
            .map(|(level, data)| {
                let (width, height) = ((64u32 >> level).max(1), (16u32 >> level).max(1));
                TextureWrite {
                    texture: TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: level,
                        origin: Origin3d { x: 0, y: 0, z: 1 },
                        aspect: TextureAspect::All,
                    },
                    data,
                    data_layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * 4),
                        rows_per_image: Some(height),
                    },
                    size: Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                }
            })
            .collect();
        ctx.queue.write_textures(&writes);
        ctx.queue.submit(None);

        for (level, expected) in (0u32..).zip(&levels) {
            assert_eq!(&read_level(&ctx, &texture, 1, level).await, expected, "level {level} of the written layer");
            let untouched = read_level(&ctx, &texture, 0, level).await;
            assert!(untouched.iter().all(|&byte| byte == 0), "level {level} of the other layer stays zero");
        }
    });

/// A batch holding an invalid write reports it and writes none of the batch,
/// the valid writes before it included.
#[gpu_test]
static WRITE_TEXTURES_INVALID_WRITE_WRITES_NONE: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let texture = mip_chain_texture(&ctx.device);
        let data = vec![200u8; 64 * 16 * 4];
        let write = |level: u32, width: u32, height: u32| TextureWrite {
            texture: TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            data: &data,
            data_layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        };
        // Level 1 is 32x8: a 64-wide write overruns it.
        let writes = [write(0, 64, 16), write(1, 64, 8)];
        fail(
            &ctx.device,
            || ctx.queue.write_textures(&writes),
            Some("overrunning the bounds of the destination texture"),
        );
        ctx.queue.submit(None);

        let level0 = read_level(&ctx, &texture, 0, 0).await;
        assert!(level0.iter().all(|&byte| byte == 0), "the valid write ahead of the invalid one is not made");
    });

/// Writes of one batch into a level an earlier write of the batch wrote land
/// after it, whether the two are next to each other or another texture's
/// write lies between them: the level holds what the last write wrote.
#[gpu_test]
static WRITE_TEXTURES_A_LATER_WRITE_LANDS_LAST: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let texture = mip_chain_texture(&ctx.device);
        let other = mip_chain_texture(&ctx.device);
        let fills: Vec<Vec<u8>> = [10u8, 20, 30, 40, 50].iter().map(|&v| vec![v; 64 * 16 * 4]).collect();
        fn write<'a>(texture: &'a Texture, layer: u32, data: &'a [u8]) -> TextureWrite<'a> {
            TextureWrite {
                texture: TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: layer },
                    aspect: TextureAspect::All,
                },
                data,
                data_layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(64 * 4),
                    rows_per_image: Some(16),
                },
                size: Extent3d {
                    width: 64,
                    height: 16,
                    depth_or_array_layers: 1,
                },
            }
        }
        let writes = [
            // Layer 0 twice in a row.
            write(&texture, 0, &fills[0]),
            write(&texture, 0, &fills[1]),
            // Layer 1, then the other texture, then layer 1 again.
            write(&texture, 1, &fills[2]),
            write(&other, 0, &fills[3]),
            write(&texture, 1, &fills[4]),
        ];
        ctx.queue.write_textures(&writes);
        ctx.queue.submit(None);

        assert!(read_level(&ctx, &texture, 0, 0).await.iter().all(|&byte| byte == 20), "layer 0 holds its second write");
        assert!(read_level(&ctx, &texture, 1, 0).await.iter().all(|&byte| byte == 50), "layer 1 holds its last write");
        assert!(read_level(&ctx, &other, 0, 0).await.iter().all(|&byte| byte == 40), "the other texture holds its write");
    });
