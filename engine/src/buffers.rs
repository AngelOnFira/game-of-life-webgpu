//! The ping-pong storage buffers and their bind groups.
//!
//! Two buffers swap roles every tick: at generation `g`, `buffers[g & 1]` is
//! read and `buffers[1 - g & 1]` is written. The two bind groups are baked
//! ahead of time so per-tick dispatch is just `set_bind_group` + `dispatch`.
//!
//! On resize we keep the [`Pipelines`] (they're size-independent) and rebuild
//! buffers + bind groups + the uniform.

use wgpu::util::DeviceExt;

use crate::patterns::{Pattern, stamp_centred};
use crate::pipelines::Pipelines;

/// Matches the `Params` struct in both the rust-gpu shader and the WGSL render
/// shader. `repr(C)` keeps layout deterministic across the three languages.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Params {
    pub width: u32,
    pub height: u32,
}

pub struct Grid {
    pub size: u32,
    pub params_buffer: wgpu::Buffer,
    pub buffers: [wgpu::Buffer; 2],
    /// `compute_bg[i]` reads `buffers[i]` and writes `buffers[1 - i]`.
    pub compute_bg: [wgpu::BindGroup; 2],
    /// `render_bg[i]` reads `buffers[i]`.
    pub render_bg: [wgpu::BindGroup; 2],
}

impl Grid {
    /// Build buffers and bind groups for an `n × n` grid.
    ///
    /// If `seed_glider`, drops a single glider near the centre of `buffers[0]`
    /// so the user sees motion immediately. Otherwise both buffers are blank.
    pub fn new(device: &wgpu::Device, pipelines: &Pipelines, n: u32, seed_glider: bool) -> Self {
        let params_buffer = create_params_buffer(device, n);
        let buffers = create_grid_buffers(device, n, seed_glider);
        let compute_bg = bind_compute_groups(device, &pipelines.compute_bgl, &params_buffer, &buffers);
        let render_bg = bind_render_groups(device, &pipelines.render_bgl, &params_buffer, &buffers);
        Self {
            size: n,
            params_buffer,
            buffers,
            compute_bg,
            render_bg,
        }
    }

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipelines: &Pipelines,
        new_size: u32,
    ) {
        let params = Params { width: new_size, height: new_size };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        self.buffers = create_grid_buffers(device, new_size, false);
        self.compute_bg = bind_compute_groups(device, &pipelines.compute_bgl, &self.params_buffer, &self.buffers);
        self.render_bg = bind_render_groups(device, &pipelines.render_bgl, &self.params_buffer, &self.buffers);
        self.size = new_size;
    }

    pub fn current_buffer(&self, generation: u64) -> &wgpu::Buffer {
        &self.buffers[(generation & 1) as usize]
    }
}

// ---------------------------------------------------------------------------
// Resource construction helpers

fn create_params_buffer(device: &wgpu::Device, n: u32) -> wgpu::Buffer {
    let params = Params { width: n, height: n };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gol.params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

fn create_grid_buffers(device: &wgpu::Device, n: u32, seed_glider: bool) -> [wgpu::Buffer; 2] {
    let cells = (n as usize).pow(2);
    let bytes = (cells as u64) * 4;
    let buf_a = if seed_glider {
        let seed = glider_seed(n);
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gol.grid.a"),
            contents: bytemuck::cast_slice(&seed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    } else {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gol.grid.a"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let buf_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gol.grid.b"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    [buf_a, buf_b]
}

fn bind_compute_groups(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    bufs: &[wgpu::Buffer; 2],
) -> [wgpu::BindGroup; 2] {
    let one = |src: usize| {
        let dst = 1 - src;
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gol.compute.bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bufs[src].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bufs[dst].as_entire_binding() },
            ],
        })
    };
    [one(0), one(1)]
}

fn bind_render_groups(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    bufs: &[wgpu::Buffer; 2],
) -> [wgpu::BindGroup; 2] {
    let one = |idx: usize| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gol.render.bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bufs[idx].as_entire_binding() },
            ],
        })
    };
    [one(0), one(1)]
}

fn glider_seed(n: u32) -> Vec<u32> {
    // Reuse the pattern definition so there's one source of truth for
    // "a glider near the centre".
    let mut g = vec![0u32; (n as usize).pow(2)];
    let centre = n as i32 / 2;
    for (x, y, v) in stamp_centred(Pattern::Glider, centre, centre, n) {
        g[(y as usize) * (n as usize) + (x as usize)] = v;
    }
    g
}
