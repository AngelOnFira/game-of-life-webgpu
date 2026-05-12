//! Compute + render pipelines and their bind-group layouts.
//!
//! **This file is almost entirely WebGPU init boilerplate** — pipeline,
//! pipeline-layout, bind-group-layout, and shader-module descriptors. None of
//! it runs per-frame. Build once in [`Pipelines::new`], then reuse every tick.
//!
//! ## Compute pipeline (`gol_step`)
//!
//! - Source: SPIR-V emitted by `spirv-builder` from the `gol-shader` crate.
//!   Naga (inside wgpu's `webgpu` backend) translates it to WGSL at runtime.
//! - Layout: one bind group with (uniform Params, storage `src`, storage `dst`).
//! - Dispatched 8×8 per workgroup, ceil(grid / 8) groups in each axis.
//!
//! ## Render pipeline (`gol_render`)
//!
//! - Source: handwritten WGSL — see `shaders/gol_render.wgsl`.
//! - Layout: (uniform Params, storage read-only grid).
//! - Draws a fullscreen triangle and indexes the grid buffer in the fragment
//!   shader.

use std::num::NonZeroU64;

use crate::buffers::Params;

/// SPIR-V emitted by `spirv-builder` from the `gol-shader` crate. The env var
/// name follows the crate name with hyphens converted to underscores.
const GOL_STEP_SPV: &[u8] = include_bytes!(env!("gol_shader.spv"));

/// Handwritten WGSL render kernel. Kept as a string so anyone reading this
/// crate can find the fragment-shader logic in one place.
const GOL_RENDER_WGSL: &str = include_str!("shaders/gol_render.wgsl");

pub struct Pipelines {
    pub compute: wgpu::ComputePipeline,
    pub render: wgpu::RenderPipeline,
    pub compute_bgl: wgpu::BindGroupLayout,
    pub render_bgl: wgpu::BindGroupLayout,
}

impl Pipelines {
    pub fn new(device: &wgpu::Device, render_format: wgpu::TextureFormat) -> Self {
        let compute_bgl = compute_bind_group_layout(device);
        let render_bgl = render_bind_group_layout(device);
        let compute = build_compute_pipeline(device, &compute_bgl);
        let render = build_render_pipeline(device, &render_bgl, render_format);
        Self {
            compute,
            render,
            compute_bgl,
            render_bgl,
        }
    }
}

// ---------------------------------------------------------------------------
// Compute side

fn compute_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gol.compute.bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<Params>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(4),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(4),
                },
                count: None,
            },
        ],
    })
}

fn build_compute_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gol.compute.shader (rust-gpu SPIR-V)"),
        source: wgpu::util::make_spirv(GOL_STEP_SPV),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gol.compute.pl"),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gol.compute"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("gol_step"),
        compilation_options: Default::default(),
        cache: None,
    })
}

// ---------------------------------------------------------------------------
// Render side

fn render_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gol.render.bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<Params>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(4),
                },
                count: None,
            },
        ],
    })
}

fn build_render_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gol.render.shader"),
        source: wgpu::ShaderSource::Wgsl(GOL_RENDER_WGSL.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gol.render.pl"),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("gol.render"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}
