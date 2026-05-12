//! GPU + CPU Game-of-Life engine.
//!
//! This crate contains *only* the simulation and rendering core — no egui, no
//! eframe, no app glue. If you came here from the README to see how the GPU
//! side works, read [`engine::GolEngine::new`] first to see what gets built,
//! then look at [`engine::GolEngine::step`] to see the per-frame flow.
//!
//! ## File map
//!
//! - [`pipelines`] — one-time setup: bind-group layouts, compute & render pipelines
//! - [`buffers`]   — ping-pong storage buffers + bind groups (rebuilt on resize)
//! - [`cpu`]       — reference CPU implementation of the same B3/S23 kernel
//! - [`patterns`]  — classic GoL patterns (glider, gun, pulsar, …)
//!
//! ## Per-frame call graph
//!
//! ```text
//!  GolEngine::step(encoder, ticks)
//!  ├── SimBackend::Gpu → ComputePass on `encoder`, alternating bind groups
//!  └── SimBackend::Cpu → cpu::step in a Vec<u32>, then queue.write_buffer
//!
//!  GolEngine::render(render_pass)
//!  └── set_pipeline + set_bind_group + draw(0..3) — fullscreen triangle
//! ```

mod buffers;
mod cpu;
mod engine;
mod patterns;
mod pipelines;
mod readback;
mod rle;
mod timing;

pub use engine::{GolEngine, INITIAL_GRID, SimBackend, StepStats, WORKGROUP_SIZE};
pub use patterns::{Pattern, stamp_centred};
pub use rle::{RleError, RlePattern};
