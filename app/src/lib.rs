#![warn(clippy::all)]

mod app;
mod callback;
mod fps_graph;

pub use app::GameOfLifeApp;

#[cfg(target_arch = "wasm32")]
use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Customise the wgpu device eframe will create:
///
/// 1. Opt into the `timestamp-query` WebGPU feature when the adapter exposes
///    it, so [`gol_engine::GolEngine`] can measure GPU compute time.
/// 2. Request the adapter's maximum `max_storage_buffer_binding_size` and
///    `max_buffer_size` so very large grids (8k²+ → 256 MiB per ping-pong
///    buffer) fit. The default limits are tuned for compatibility, not
///    capacity, and would reject the buffer with a panic at create-time.
#[cfg(target_arch = "wasm32")]
fn gpu_timing_aware_wgpu_options() -> egui_wgpu::WgpuConfiguration {
    let mut cfg = egui_wgpu::WgpuConfiguration::default();
    if let egui_wgpu::WgpuSetup::CreateNew(ref mut setup) = cfg.wgpu_setup {
        let inner = setup.device_descriptor.clone();
        setup.device_descriptor = Arc::new(move |adapter| {
            let mut desc = inner(adapter);
            let adapter_limits = adapter.limits();
            if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
                desc.required_features |= wgpu::Features::TIMESTAMP_QUERY;
            }
            desc.required_limits.max_storage_buffer_binding_size =
                adapter_limits.max_storage_buffer_binding_size;
            desc.required_limits.max_buffer_size = adapter_limits.max_buffer_size;
            desc
        });
    }
    cfg
}

/// Web entry point — called from `index.html` once the wasm module loads.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);

    let web_options = eframe::WebOptions {
        wgpu_options: gpu_timing_aware_wgpu_options(),
        ..eframe::WebOptions::default()
    };

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("no window")
            .document()
            .expect("no document");
        let canvas = document
            .get_element_by_id("gol-canvas")
            .expect("missing <canvas id=\"gol-canvas\">")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("#gol-canvas is not a canvas");

        let result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(GameOfLifeApp::new(cc)))),
            )
            .await;

        if let Err(err) = result {
            let msg = format!("eframe failed to start: {err:?}");
            log::error!("{msg}");
            if let Some(elem) = document.get_element_by_id("loading-message") {
                elem.set_inner_html(&format!(
                    "<p style=\"color:#c33\">{msg}</p>\
                     <p>Open the browser console for details.</p>"
                ));
            }
        }
    });

    Ok(())
}
