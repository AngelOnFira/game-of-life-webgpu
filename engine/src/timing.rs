//! Optional GPU-side timing of the compute pass via wgpu's timestamp queries.
//!
//! On browsers that expose the `timestamp-query` WebGPU feature (Chrome ≥ 121,
//! some Firefox builds with the flag on), this module reads the actual GPU
//! time delta from a `QuerySet`. Otherwise it's a no-op and the UI shows
//! "unavailable".
//!
//! ## Async dance
//!
//! Readback is asynchronous. We use single-buffered measurement: at most one
//! readback is in flight. The state machine is:
//!
//! ```text
//!   idle ──[start_pass()]──► recording
//!     ▲                          │
//!     │                       [finish_pass()]
//!     │                          │
//!     │                          ▼
//!     │                       submitted  (egui_wgpu does the queue submit)
//!     │                          │
//!     │                          ▼
//!     │                    [poll_readback() on next frame]
//!     │                          │
//!     │                          ▼
//!     └─────[map_async cb]──── mapped → record result, unmap → idle
//! ```
//!
//! Calling `start_pass` while a readback is in flight returns `None`, so the
//! caller skips timestamping that frame.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

/// Bytes per `u64` timestamp × 2 timestamps (start + end).
const READBACK_BYTES: u64 = 16;

#[repr(u8)]
enum State {
    Idle = 0,
    Submitted = 1,
    Mapping = 2,
}

pub struct Timing {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    timestamp_period: f32,
    state: Arc<AtomicU8>,
    latest_ns: Arc<Mutex<Option<u64>>>,
}

impl Timing {
    /// Returns `Some(Timing)` if the device has the timestamp-query feature
    /// enabled. Otherwise `None` and timing is not tracked.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gol.timing.queries"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gol.timing.resolve"),
            size: READBACK_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gol.timing.readback"),
            size: READBACK_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Self {
            query_set,
            resolve_buf,
            readback_buf,
            timestamp_period: queue.get_timestamp_period(),
            state: Arc::new(AtomicU8::new(State::Idle as u8)),
            latest_ns: Arc::new(Mutex::new(None)),
        })
    }

    /// If timing is idle, returns the descriptor pair to record around the
    /// compute pass. Otherwise (still waiting on a previous readback), returns
    /// `None` and the caller should skip timing this frame.
    pub fn try_start_pass(&self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        if self.state.load(Ordering::Acquire) != State::Idle as u8 {
            return None;
        }
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        })
    }

    /// Record the resolve + copy commands into the egui encoder. Called only
    /// when `try_start_pass` returned `Some` and the pass was actually timed.
    pub fn finish_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..2, &self.resolve_buf, 0);
        encoder.copy_buffer_to_buffer(&self.resolve_buf, 0, &self.readback_buf, 0, READBACK_BYTES);
        self.state.store(State::Submitted as u8, Ordering::Release);
    }

    /// Called once per frame. If a readback is queued, asks wgpu to map the
    /// buffer; the result lands in `latest_ns` whenever the GPU completes.
    pub fn poll_readback(&self) {
        if self
            .state
            .compare_exchange(
                State::Submitted as u8,
                State::Mapping as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return;
        }
        let buf = self.readback_buf.clone();
        let state = self.state.clone();
        let latest = self.latest_ns.clone();
        let period = self.timestamp_period;
        self.readback_buf
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    let view = buf.slice(..).get_mapped_range();
                    let [t0, t1] = bytemuck::cast_slice::<u8, u64>(&view[..16])
                        .try_into()
                        .expect("16 bytes -> [u64; 2]");
                    drop(view);
                    buf.unmap();
                    let ticks = t1.saturating_sub(t0);
                    let ns = (ticks as f64) * (period as f64);
                    *latest.lock().unwrap() = Some(ns as u64);
                }
                state.store(State::Idle as u8, Ordering::Release);
            });
    }

    /// Most recent successfully measured GPU compute time.
    pub fn latest(&self) -> Option<Duration> {
        let ns = (*self.latest_ns.lock().unwrap())?;
        Some(Duration::from_nanos(ns))
    }
}
