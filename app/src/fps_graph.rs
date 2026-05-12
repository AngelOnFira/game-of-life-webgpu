//! Sliding-window *compute* time chart rendered with `egui_plot`.
//!
//! Plots two series over the last `CAPACITY` frames:
//!
//! - **CPU dispatch** — wall-clock time spent inside `GolEngine::step` on the
//!   main thread. For GPU mode this is mostly driver-submission overhead; for
//!   CPU mode it's the actual simulation cost.
//! - **GPU compute** — actual on-device compute time read from the
//!   `timestamp-query` feature. Only plotted in frames where the engine
//!   surfaced a measurement (the browser may not support the feature, or the
//!   async readback may not have completed yet for the latest frame).
//!
//! We deliberately do *not* plot total frame time — that's clamped to the
//! display's vsync (~16.7 ms) and obscures sub-millisecond compute deltas,
//! which is the whole point of the demo.
//!
//! Outliers above p99 across both series are dropped from display + summary
//! stats so a tab-out spike doesn't squash the chart.

use std::collections::VecDeque;

use egui_plot::{Line, Plot, PlotPoints};

use gol_engine::StepStats;

/// ~30 seconds at 60 fps.
const CAPACITY: usize = 30 * 60;

/// Pixel height of the chart in the side panel.
const HEIGHT: f32 = 200.0;

/// Y axis is extended below zero by this fraction of the data range so the
/// legend in the bottom-left has its own canvas area instead of overlaying
/// the lines.
const LEGEND_HEADROOM: f64 = 0.5;

#[derive(Copy, Clone, Default)]
struct Sample {
    cpu_ms: f32,
    /// Only `Some` when the GPU reported a timestamp this frame.
    gpu_ms: Option<f32>,
}

pub struct FpsGraph {
    samples: VecDeque<Sample>,
}

impl Default for FpsGraph {
    fn default() -> Self {
        Self {
            samples: VecDeque::with_capacity(CAPACITY),
        }
    }
}

impl FpsGraph {
    pub fn push_step(&mut self, stats: StepStats) {
        if self.samples.len() == CAPACITY {
            self.samples.pop_front();
        }
        self.samples.push_back(Sample {
            cpu_ms: stats.cpu_time.as_secs_f32() * 1000.0,
            gpu_ms: stats.gpu_time.map(|d| d.as_secs_f32() * 1000.0),
        });
    }

    /// p99 across all values in both series — used both to clip the Y axis
    /// and to filter outliers from summary stats.
    fn p99_ms(&self) -> f32 {
        let mut all: Vec<f32> = self
            .samples
            .iter()
            .flat_map(|s| std::iter::once(s.cpu_ms).chain(s.gpu_ms))
            .collect();
        if all.is_empty() {
            return 0.0;
        }
        all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((all.len() as f32) * 0.99) as usize;
        all[idx.min(all.len() - 1)]
    }

    /// Average CPU dispatch (filtering > p99).
    pub fn avg_cpu_ms(&self) -> f32 {
        let cap = self.p99_ms();
        let kept: Vec<f32> = self
            .samples
            .iter()
            .map(|s| s.cpu_ms)
            .filter(|&x| x <= cap)
            .collect();
        if kept.is_empty() {
            0.0
        } else {
            kept.iter().sum::<f32>() / kept.len() as f32
        }
    }

    /// p99 CPU dispatch.
    pub fn p99_cpu_ms(&self) -> f32 {
        self.p99_ms()
    }

    pub fn show(&self, ui: &mut egui::Ui) {
        let cap = self.p99_ms();

        let cpu_points: PlotPoints = self
            .samples
            .iter()
            .enumerate()
            .filter(|&(_, s)| s.cpu_ms <= cap)
            .map(|(i, s)| [i as f64, s.cpu_ms as f64])
            .collect();

        // GPU series exists only in frames where a timestamp landed; we
        // skip None frames so the line doesn't snap to zero.
        let gpu_points: PlotPoints = self
            .samples
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                let gpu = s.gpu_ms?;
                (gpu <= cap).then_some([i as f64, gpu as f64])
            })
            .collect();
        let has_gpu = !gpu_points.points().is_empty();

        let y_max = (cap.max(0.5) * 1.1) as f64;
        // Add negative-Y headroom so the legend in the bottom-left has its
        // own area below the data and doesn't obscure any sample lines.
        let y_min = -y_max * LEGEND_HEADROOM;

        Plot::new("compute_strip")
            .height(HEIGHT)
            .show_axes([false, true])
            .show_grid([false, true])
            .y_axis_label("ms (compute)")
            .allow_zoom(false)
            .allow_drag(false)
            .allow_scroll(false)
            .allow_boxed_zoom(false)
            .include_x(0.0)
            .include_x(CAPACITY as f64)
            .include_y(y_min)
            .include_y(y_max)
            .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftBottom))
            .show(ui, |plot| {
                plot.line(
                    Line::new("CPU dispatch", cpu_points)
                        .color(egui::Color32::from_rgb(120, 180, 230)),
                );
                if has_gpu {
                    plot.line(
                        Line::new("GPU compute", gpu_points)
                            .color(egui::Color32::from_rgb(220, 140, 220)),
                    );
                }
            });
    }
}
