//! Per-frame paint callback. Translates UI intentions (clear/randomize/step/…)
//! into engine calls, then asks the engine to render itself into egui's pass.

use egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};
use gol_engine::{GolEngine, SimBackend};

/// One frame's worth of work. Cheap to construct; carries no GPU handles.
///
/// Ops are applied in `prepare` in this order so user actions take effect
/// before the simulation steps for that frame:
/// 1. `set_backend` (clears as a side-effect when changed)
/// 2. `resize`
/// 3. `clear`
/// 4. `randomize` (seeded)
/// 5. `paint` (stamps individual cells)
/// 6. `ticks` simulation steps
#[derive(Default)]
pub struct GolCallback {
    pub set_backend: Option<SimBackend>,
    pub resize: Option<u32>,
    pub clear: bool,
    pub randomize: Option<u64>,
    pub paint: Vec<(u32, u32, u32)>,
    pub ticks: u32,
}

impl CallbackTrait for GolCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _screen: &ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(engine) = resources.get_mut::<GolEngine>() else {
            return Vec::new();
        };
        if let Some(b) = self.set_backend {
            engine.set_backend(b);
        }
        if let Some(n) = self.resize {
            engine.resize(n);
        }
        if self.clear {
            engine.clear();
        }
        if let Some(seed) = self.randomize {
            engine.randomize(seed);
        }
        if !self.paint.is_empty() {
            engine.stamp(&self.paint);
        }
        engine.step(encoder, self.ticks);
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::epaint::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        if let Some(engine) = resources.get::<GolEngine>() {
            engine.render(render_pass);
        }
    }
}
