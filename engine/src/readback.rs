//! Async readback of the current GPU grid buffer into a `Vec<u32>`.
//!
//! Used when the host app switches from GPU to CPU mode mid-simulation:
//! the GPU's buffer is authoritative, so we copy it back to populate the CPU
//! shadow grid before flipping the backend. Web's WebGPU has no synchronous
//! readback, so this dances through three states:
//!
//! ```text
//!   NeedsRecord ─ [step records copy_buffer_to_buffer] ─► AwaitingSubmit
//!   AwaitingSubmit ─ [next frame: map_async] ─────────► Mapping
//!   Mapping ─ [browser fires map callback] ──────────► Done(Vec<u32>)
//! ```
//!
//! [`GolEngine::step`] advances this state machine and, on `Done`, copies the
//! payload into the CPU sim and switches the backend.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};

#[repr(u8)]
enum State {
    NeedsRecord = 0,
    AwaitingSubmit = 1,
    Mapping = 2,
    Done = 3,
}

pub struct PendingReadback {
    /// `MAP_READ | COPY_DST` buffer the GPU's grid is copied into.
    pub buffer: wgpu::Buffer,
    /// Edge length of the grid being read back. Bails out if the engine
    /// resizes mid-readback (we discard the in-flight buffer in that case).
    pub grid_size: u32,
    pub state: Arc<AtomicU8>,
    pub data: Arc<Mutex<Vec<u32>>>,
}

impl PendingReadback {
    /// Allocate a fresh readback. Caller is responsible for recording the
    /// `copy_buffer_to_buffer` into the current frame's encoder.
    pub fn new(device: &wgpu::Device, grid_size: u32) -> Self {
        let bytes = (grid_size as u64).pow(2) * 4;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gol.readback.to_cpu"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            grid_size,
            state: Arc::new(AtomicU8::new(State::NeedsRecord as u8)),
            data: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// `true` when the readback hasn't been encoded yet.
    pub fn needs_record(&self) -> bool {
        self.state.load(Ordering::Acquire) == State::NeedsRecord as u8
    }

    /// Transition NeedsRecord → AwaitingSubmit. Caller has just written the
    /// copy into the encoder; egui_wgpu will submit it shortly.
    pub fn mark_recorded(&self) {
        self.state
            .store(State::AwaitingSubmit as u8, Ordering::Release);
    }

    /// Kick off the map if egui has submitted the recorded copy. Idempotent.
    pub fn try_start_map(&self) {
        if self
            .state
            .compare_exchange(
                State::AwaitingSubmit as u8,
                State::Mapping as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return;
        }
        let buf = self.buffer.clone();
        let state = self.state.clone();
        let data = self.data.clone();
        self.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    let view = buf.slice(..).get_mapped_range();
                    let out: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
                    drop(view);
                    buf.unmap();
                    *data.lock().unwrap() = out;
                    state.store(State::Done as u8, Ordering::Release);
                } else {
                    // Treat error as "discard"; the engine will drop the pending
                    // readback and fall back to clearing on switch.
                    state.store(State::NeedsRecord as u8, Ordering::Release);
                }
            });
    }

    /// If the map has completed, returns the populated `Vec<u32>`.
    pub fn take_if_done(&self) -> Option<Vec<u32>> {
        if self.state.load(Ordering::Acquire) != State::Done as u8 {
            return None;
        }
        let mut data = self.data.lock().unwrap();
        Some(std::mem::take(&mut *data))
    }
}
