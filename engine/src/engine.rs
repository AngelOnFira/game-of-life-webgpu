//! The public `GolEngine` type. This file deliberately stays short — most of
//! the GPU machinery lives in `pipelines.rs` and `buffers.rs`. Operations
//! (clear/randomize/stamp/step/render) are here.

use std::time::Duration;
use web_time::Instant;

use crate::buffers::Grid;
use crate::cpu::CpuSim;
use crate::pipelines::Pipelines;
use crate::readback::PendingReadback;
use crate::timing::Timing;

/// Compute shader workgroup edge. Must match `shader/src/lib.rs`.
pub const WORKGROUP_SIZE: u32 = 8;

/// Grid edge length at startup. The host app can resize at runtime.
pub const INITIAL_GRID: u32 = 1024;

/// Which simulation backend is driving the grid.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum SimBackend {
    /// Compute shader on the GPU (rust-gpu kernel running via WebGPU).
    #[default]
    Gpu,
    /// Reference implementation in Rust on the CPU (one `Vec<u32>` per step,
    /// uploaded to the GPU buffer for display).
    Cpu,
}

/// Per-step diagnostics, returned from [`GolEngine::step`].
#[derive(Copy, Clone, Debug, Default)]
pub struct StepStats {
    /// Wall-clock time spent inside [`GolEngine::step`] on this thread.
    /// For GPU mode this is essentially driver-submission overhead; for CPU
    /// mode this is the actual simulation cost.
    pub cpu_time: Duration,
    /// Number of ticks dispatched in this call.
    pub ticks: u32,
    /// Which backend ran the ticks.
    pub backend: SimBackend,
    /// Most recent GPU-side compute time, if the WebGPU `timestamp-query`
    /// feature is available. Lags real time by ≥1 frame because readback
    /// is asynchronous. `None` if the feature is unavailable in this browser
    /// or no GPU measurement has completed yet.
    pub gpu_time: Option<Duration>,
}

pub struct GolEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,

    pipelines: Pipelines,
    grid: Grid,

    cpu: CpuSim,
    backend: SimBackend,
    timing: Option<Timing>,

    /// Set when the user requested GPU→CPU but the readback hasn't completed
    /// yet. Engine keeps running in GPU mode until the data lands; then the
    /// backend flips to CPU with the freshly-copied state.
    pending_to_cpu: Option<PendingReadback>,

    /// Generation counter — `buffers[generation & 1]` is the *current* grid.
    /// Only the GPU path touches this; the CPU path always writes to slot 0
    /// after each step.
    generation: u64,

    /// Most recent `step` measurement, for the host app's diagnostics panel.
    last_stats: StepStats,
}

impl GolEngine {
    /// Build all GPU resources for a `grid_size × grid_size` simulation.
    ///
    /// `render_format` is the surface texture format eframe is rendering into.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        render_format: wgpu::TextureFormat,
        grid_size: u32,
    ) -> Self {
        let pipelines = Pipelines::new(&device, render_format);
        let grid = Grid::new(&device, &pipelines, grid_size, /* seed_glider */ true);
        let cpu = CpuSim::new(grid_size);
        let timing = Timing::new(&device, &queue);
        if timing.is_some() {
            log::info!("GPU timestamp queries available — measuring compute time on device");
        } else {
            log::info!("GPU timestamp queries unavailable in this browser");
        }
        Self {
            device,
            queue,
            pipelines,
            grid,
            cpu,
            backend: SimBackend::Gpu,
            generation: 0,
            last_stats: StepStats::default(),
            timing,
            pending_to_cpu: None,
        }
    }

    /// `true` if GPU-side compute timing is available.
    pub fn has_gpu_timing(&self) -> bool {
        self.timing.is_some()
    }

    /// Result of the most recent `step` call (zeroed if none yet).
    pub fn recent_stats(&self) -> StepStats {
        self.last_stats
    }

    pub fn grid_size(&self) -> u32 {
        self.grid.size
    }

    pub fn backend(&self) -> SimBackend {
        self.backend
    }

    /// Switch the active simulation backend, preserving the current grid state.
    ///
    /// - **CPU → GPU**: synchronous. The CPU `Vec` is uploaded into the GPU
    ///   buffer that GPU mode will read on the next tick.
    /// - **GPU → CPU**: asynchronous. WebGPU only exposes async `map_async`,
    ///   so we queue a `copy_buffer_to_buffer` from the live GPU buffer into
    ///   a MAP_READ buffer; the engine keeps running in GPU mode for the
    ///   ≥2 frames it takes the readback to complete, then flips to CPU
    ///   mode populated with the captured data.
    pub fn set_backend(&mut self, backend: SimBackend) {
        // Re-clicking GPU while a GPU→CPU readback is in flight cancels it.
        if backend == SimBackend::Gpu {
            self.pending_to_cpu = None;
        }
        if backend == self.backend {
            return;
        }
        match (self.backend, backend) {
            (SimBackend::Cpu, SimBackend::Gpu) => {
                // Upload CPU state to slot 0 and reset generation so GPU reads
                // from slot 0 on its first tick.
                self.queue.write_buffer(
                    &self.grid.buffers[0],
                    0,
                    bytemuck::cast_slice(self.cpu.data()),
                );
                self.generation = 0;
                self.backend = SimBackend::Gpu;
            }
            (SimBackend::Gpu, SimBackend::Cpu) => {
                // Queue an async readback of the *current* GPU buffer. We stay
                // in GPU mode until `step` notices the readback completed.
                self.pending_to_cpu = Some(PendingReadback::new(&self.device, self.grid.size));
            }
            _ => {}
        }
    }

    pub fn resize(&mut self, new_size: u32) {
        if new_size == self.grid.size {
            self.clear();
            return;
        }
        // Resize invalidates any in-flight readback (size mismatch).
        self.pending_to_cpu = None;
        self.grid
            .resize(&self.device, &self.queue, &self.pipelines, new_size);
        self.cpu.resize(new_size);
        self.generation = 0;
    }

    pub fn clear(&mut self) {
        self.cpu.clear();
        let blank = vec![0u32; (self.grid.size as usize).pow(2)];
        self.queue.write_buffer(
            self.grid.current_buffer(self.generation),
            0,
            bytemuck::cast_slice(&blank),
        );
        self.generation = 0;
    }

    pub fn randomize(&mut self, seed: u64) {
        let data = self.cpu.randomize(seed);
        self.queue.write_buffer(
            self.grid.current_buffer(self.generation),
            0,
            bytemuck::cast_slice(data),
        );
    }

    /// Toggle individual cells alive in the current buffer.
    pub fn stamp(&mut self, cells: &[(u32, u32, u32)]) {
        self.cpu.stamp(cells);
        if matches!(self.backend, SimBackend::Cpu) {
            // CPU is authoritative; upload the full CPU buffer.
            self.queue.write_buffer(
                self.grid.current_buffer(self.generation),
                0,
                bytemuck::cast_slice(self.cpu.data()),
            );
            return;
        }
        // GPU is authoritative; poke individual u32s into the current buffer.
        let stride = self.grid.size as u64 * 4;
        let buf = self.grid.current_buffer(self.generation);
        for &(x, y, v) in cells {
            if x >= self.grid.size || y >= self.grid.size {
                continue;
            }
            let offset = y as u64 * stride + x as u64 * 4;
            self.queue.write_buffer(buf, offset, bytemuck::bytes_of(&v));
        }
    }

    /// Advance the simulation by `ticks` steps. Records GPU work into
    /// `encoder` when in GPU mode; performs CPU work and uploads the result
    /// otherwise.
    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder, ticks: u32) -> StepStats {
        let start = Instant::now();
        // Poll the *previous* frame's pending readback BEFORE recording any new
        // commands. If we did it after `step_gpu`, this frame's
        // `copy_buffer_to_buffer` into the readback buffer would still be in the
        // encoder, and calling `map_async` on the same buffer makes egui_wgpu's
        // upcoming submit error with "buffer used in submit while mapped".
        //
        // Poll-before-record also enforces the invariant that any given frame
        // either kicks off a `map_async` OR records a new copy — never both.
        if let Some(t) = &self.timing {
            t.poll_readback();
        }
        // GPU→CPU readback state machine: advance to mapping once the
        // recorded copy has been submitted, and finalise the switch once
        // bytes are back.
        if let Some(p) = &self.pending_to_cpu {
            p.try_start_map();
        }
        if let Some(data) = self.pending_to_cpu.as_ref().and_then(|p| p.take_if_done()) {
            self.cpu.adopt(data);
            self.pending_to_cpu = None;
            self.backend = SimBackend::Cpu;
            self.generation = 0;
        }

        if ticks > 0 {
            match self.backend {
                SimBackend::Gpu => self.step_gpu(encoder, ticks),
                SimBackend::Cpu => self.step_cpu(ticks),
            }
        }

        // If a GPU→CPU readback is still waiting to be encoded, do it now —
        // after step_gpu, so we read whichever buffer holds the freshest state.
        if let Some(p) = &self.pending_to_cpu
            && p.needs_record()
            && p.grid_size == self.grid.size
            && matches!(self.backend, SimBackend::Gpu)
        {
            let src = self.grid.current_buffer(self.generation);
            let bytes = (self.grid.size as u64).pow(2) * 4;
            encoder.copy_buffer_to_buffer(src, 0, &p.buffer, 0, bytes);
            p.mark_recorded();
        }
        let stats = StepStats {
            cpu_time: start.elapsed(),
            ticks,
            backend: self.backend,
            gpu_time: self.timing.as_ref().and_then(Timing::latest),
        };
        if ticks > 0 {
            self.last_stats = stats;
        } else {
            // Even on idle frames, surface a fresh `gpu_time` if a readback
            // just completed — but keep the old `cpu_time` so the panel stays put.
            self.last_stats.gpu_time = stats.gpu_time;
        }
        stats
    }

    fn step_gpu(&mut self, encoder: &mut wgpu::CommandEncoder, ticks: u32) {
        // Try to record GPU-side timestamps around the *whole* batch of ticks
        // (cheap: at most two timestamps per frame regardless of `ticks`).
        let timestamp_writes = self
            .timing
            .as_ref()
            .and_then(Timing::try_start_pass);
        let timing_active = timestamp_writes.is_some();

        let groups = self.grid.size.div_ceil(WORKGROUP_SIZE);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gol.compute.pass"),
                timestamp_writes,
            });
            pass.set_pipeline(&self.pipelines.compute);
            for _ in 0..ticks {
                let src = (self.generation & 1) as usize;
                pass.set_bind_group(0, &self.grid.compute_bg[src], &[]);
                pass.dispatch_workgroups(groups, groups, 1);
                self.generation = self.generation.wrapping_add(1);
            }
        }
        if timing_active
            && let Some(t) = &self.timing
        {
            t.finish_pass(encoder);
        }
    }

    fn step_cpu(&mut self, ticks: u32) {
        for _ in 0..ticks {
            self.cpu.step();
        }
        // For CPU mode, always display from slot 0.
        self.queue.write_buffer(
            &self.grid.buffers[0],
            0,
            bytemuck::cast_slice(self.cpu.data()),
        );
        self.generation = 0;
    }

    /// Draw the current grid into the supplied render pass as a fullscreen quad.
    pub fn render(&self, render_pass: &mut wgpu::RenderPass<'static>) {
        render_pass.set_pipeline(&self.pipelines.render);
        let bg_idx = match self.backend {
            // GPU: whichever buffer is current after the latest step.
            SimBackend::Gpu => (self.generation & 1) as usize,
            // CPU: always slot 0 (that's where we upload).
            SimBackend::Cpu => 0,
        };
        render_pass.set_bind_group(0, &self.grid.render_bg[bg_idx], &[]);
        render_pass.draw(0..3, 0..1);
    }
}
