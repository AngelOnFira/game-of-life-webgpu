use eframe::CreationContext;
use gol_engine::{
    GolEngine, INITIAL_GRID, Pattern, RlePattern, SimBackend, StepStats, stamp_centred,
};

use crate::callback::GolCallback;
use crate::fps_graph::FpsGraph;

const GRID_SIZES: &[u32] = &[1024, 2048, 4096, 8096];

/// Which side-panel tab is visible.
#[derive(Copy, Clone, PartialEq, Eq, Default)]
enum Tab {
    #[default]
    Sim,
    Import,
    About,
}

pub struct GameOfLifeApp {
    // --- simulation control state ---
    running: bool,
    ticks_per_frame: u32,
    grid_size: u32,
    /// What the user picked in the UI.
    backend: SimBackend,
    /// What the engine is actually running. Differs briefly during the
    /// async GPU→CPU handover.
    engine_backend: SimBackend,
    selected_pattern: Pattern,

    // --- one-shot actions, drained into the callback each frame ---
    pending_clear: bool,
    pending_randomize: bool,
    pending_step: bool,
    pending_resize: Option<u32>,
    pending_paint: Vec<(u32, u32, u32)>,
    pending_backend: Option<SimBackend>,

    // --- interaction state ---
    painting: bool,
    /// Free-text RLE buffer the user pastes patterns into.
    rle_text: String,
    /// Sticky message from the last RLE-load attempt (success or error).
    rle_status: Option<(bool, String)>,

    // --- diagnostics ---
    last_frame: Option<web_time::Instant>,
    frame_dt_ms: f32,
    fps_graph: FpsGraph,
    last_step_stats: StepStats,
    /// Set in `new` if eframe gave us a working wgpu render state.
    has_wgpu: bool,

    // --- UI state ---
    tab: Tab,
}

impl GameOfLifeApp {
    pub fn new(cc: &CreationContext<'_>) -> Self {
        let mut has_wgpu = false;
        if let Some(render_state) = cc.wgpu_render_state.as_ref() {
            let device = render_state.device.clone();
            let queue = render_state.queue.clone();
            let format = render_state.target_format;
            let engine = GolEngine::new(device, queue, format, INITIAL_GRID);
            render_state
                .renderer
                .write()
                .callback_resources
                .insert(engine);
            has_wgpu = true;
        } else {
            log::error!("eframe did not provide a wgpu RenderState — WebGPU likely unavailable");
        }

        Self {
            running: true,
            ticks_per_frame: 1,
            grid_size: INITIAL_GRID,
            backend: SimBackend::Gpu,
            engine_backend: SimBackend::Gpu,
            selected_pattern: Pattern::GosperGliderGun,
            pending_clear: false,
            pending_randomize: false,
            pending_step: false,
            pending_resize: None,
            pending_paint: Vec::new(),
            pending_backend: None,
            painting: false,
            rle_text: String::new(),
            rle_status: None,
            last_frame: None,
            frame_dt_ms: 0.0,
            fps_graph: FpsGraph::default(),
            last_step_stats: StepStats::default(),
            has_wgpu,
            tab: Tab::default(),
        }
    }

    /// Update frame-time tracking and pull the latest engine stats out of
    /// `callback_resources`. Called once per frame.
    fn sample_diagnostics(&mut self, frame: &mut eframe::Frame) {
        let now = web_time::Instant::now();
        if let Some(prev) = self.last_frame {
            let dt = (now - prev).as_secs_f32() * 1000.0;
            self.frame_dt_ms = 0.9 * self.frame_dt_ms + 0.1 * dt;
        }
        self.last_frame = Some(now);

        if let Some(rs) = frame.wgpu_render_state()
            && let Some(engine) = rs.renderer.read().callback_resources.get::<GolEngine>()
        {
            let stats = engine.recent_stats();
            self.last_step_stats = stats;
            self.engine_backend = engine.backend();
            // Feed the chart with *compute* time (CPU dispatch + GPU when
            // available). Frame time is bounded by vsync at ~16.7 ms and
            // would hide the sub-millisecond compute we want to compare.
            self.fps_graph.push_step(stats);
        }
    }

    /// Parse the current RLE textarea, clear the grid, and stamp the pattern
    /// at the centre. On failure store the error in `rle_status` for the
    /// user to see.
    fn try_load_rle(&mut self) {
        match RlePattern::parse(&self.rle_text) {
            Ok(pat) => {
                if pat.width > self.grid_size || pat.height > self.grid_size {
                    self.rle_status = Some((
                        false,
                        format!(
                            "pattern is {}×{} but grid is {2}×{2}; pick a larger grid in the Sim tab",
                            pat.width, pat.height, self.grid_size,
                        ),
                    ));
                    return;
                }
                let c = self.grid_size as i32 / 2;
                let cells = pat.stamp_centred(c, c, self.grid_size);
                let stamped = cells.len();
                self.pending_clear = true;
                self.pending_paint.extend(cells);
                self.rle_status = Some((
                    true,
                    format!(
                        "cleared grid; stamped {} cells ({}×{}) at centre",
                        stamped, pat.width, pat.height,
                    ),
                ));
            }
            Err(err) => {
                self.rle_status = Some((false, format!("parse error: {err}")));
            }
        }
    }

    fn show_side_panel(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("controls")
            .resizable(false)
            .default_size(300.0)
            .show_inside(ui, |ui| {
                ui.heading("Game of Life");
                ui.label("Rust GPU → WebGPU demo");

                if !self.has_wgpu {
                    ui.separator();
                    ui.colored_label(egui::Color32::LIGHT_RED, "WebGPU not available.");
                    return;
                }

                // Tab strip
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.tab, Tab::Sim, "Sim");
                    ui.selectable_value(&mut self.tab, Tab::Import, "Import");
                    ui.selectable_value(&mut self.tab, Tab::About, "About");
                });
                ui.separator();

                match self.tab {
                    Tab::Sim => self.tab_sim(ui),
                    Tab::Import => self.tab_import(ui),
                    Tab::About => self.tab_about(ui),
                }
            });
    }

    // -----------------------------------------------------------------
    // Tab bodies

    fn tab_sim(&mut self, ui: &mut egui::Ui) {
        // --- backend toggle ---
        ui.label("simulation backend");
        ui.horizontal(|ui| {
            let mut b = self.backend;
            ui.selectable_value(&mut b, SimBackend::Gpu, "GPU (rust-gpu)");
            ui.selectable_value(&mut b, SimBackend::Cpu, "CPU (Rust)");
            if b != self.backend {
                self.backend = b;
                self.pending_backend = Some(b);
            }
        });
        if self.engine_backend != self.backend {
            ui.small(
                egui::RichText::new("waiting for GPU readback…")
                    .color(egui::Color32::from_rgb(220, 180, 80)),
            );
        }

        ui.separator();

        // --- playback ---
        ui.horizontal(|ui| {
            let label = if self.running { "⏸ Pause" } else { "▶ Play" };
            if ui.button(label).clicked() {
                self.running = !self.running;
            }
            if ui
                .add_enabled(!self.running, egui::Button::new("⏭ Step"))
                .clicked()
            {
                self.pending_step = true;
            }
        });

        ui.separator();

        // --- simulation rate ---
        ui.strong("simulation rate");
        ui.add(
            egui::Slider::new(&mut self.ticks_per_frame, 0..=1000)
                .logarithmic(true)
                .text("ticks/frame"),
        );
        let fps_for_rate = if self.frame_dt_ms > 0.0 {
            1000.0 / self.frame_dt_ms
        } else {
            0.0
        };
        let ticks_per_sec = fps_for_rate * self.ticks_per_frame as f32;
        ui.label(format!(
            "≈ {:.0} ticks/sec  (render ≈ {:.0} fps × {} tick(s)/frame)",
            ticks_per_sec, fps_for_rate, self.ticks_per_frame,
        ));
        ui.small(
            "Each tick advances every cell by one Game-of-Life generation. \
             Rendering is vsync-capped to ~60 fps, but the simulation isn't — \
             crank ticks/frame to see compute throughput past 60 ticks/sec.",
        );

        ui.separator();

        // --- buffer actions ---
        ui.horizontal(|ui| {
            if ui.button("🗑 Clear").clicked() {
                self.pending_clear = true;
            }
            if ui.button("🎲 Random").clicked() {
                self.pending_randomize = true;
            }
        });

        ui.separator();

        // --- grid size ---
        ui.label("grid size");
        egui::ComboBox::from_id_salt("grid_size")
            .selected_text(format!("{0}×{0}", self.grid_size))
            .show_ui(ui, |ui| {
                for &n in GRID_SIZES {
                    if ui
                        .selectable_label(n == self.grid_size, format!("{n}×{n}"))
                        .clicked()
                        && n != self.grid_size
                    {
                        self.pending_resize = Some(n);
                        self.grid_size = n;
                    }
                }
            });
        if matches!(self.backend, SimBackend::Cpu) && self.grid_size >= 1024 {
            ui.colored_label(
                egui::Color32::from_rgb(220, 180, 80),
                format!("⚠ CPU on {0}×{0} is slow", self.grid_size),
            );
        }

        ui.separator();

        // --- preset patterns ---
        ui.label("preset pattern");
        let mut sp = self.selected_pattern;
        egui::ComboBox::from_id_salt("pattern")
            .selected_text(sp.name())
            .show_ui(ui, |ui| {
                for &p in Pattern::ALL {
                    ui.selectable_value(&mut sp, p, p.name());
                }
            });
        self.selected_pattern = sp;
        if ui.button("⤓ Stamp at centre").clicked() {
            let c = self.grid_size as i32 / 2;
            self.pending_paint.extend(stamp_centred(
                self.selected_pattern,
                c,
                c,
                self.grid_size,
            ));
        }

        ui.separator();
        ui.small("• Left-click + drag in the grid to paint cells.");

        ui.separator();

        // --- timing + 30s compute chart ---
        ui.strong("timing");

        let stats = self.last_step_stats;
        let active_label = match stats.backend {
            SimBackend::Gpu => "active: GPU (rust-gpu compute shader)",
            SimBackend::Cpu => "active: CPU (Rust kernel on the main thread)",
        };
        ui.colored_label(egui::Color32::from_rgb(120, 200, 240), active_label);

        let ticks_run = stats.ticks.max(1);
        let cpu_total_ms = stats.cpu_time.as_secs_f32() * 1000.0;
        let cpu_per_tick_us = (cpu_total_ms * 1000.0) / ticks_run as f32;

        match stats.backend {
            SimBackend::Cpu => {
                ui.label(format!(
                    "CPU compute: {:.2} ms / step  ({:.1} µs/tick × {})",
                    cpu_total_ms, cpu_per_tick_us, stats.ticks,
                ));
            }
            SimBackend::Gpu => {
                match stats.gpu_time {
                    Some(gt) => {
                        let gpu_ms = gt.as_secs_f32() * 1000.0;
                        ui.label(format!(
                            "GPU compute: {:.3} ms",
                            gpu_ms,
                        ));
                    }
                    None => {
                        ui.label(
                            egui::RichText::new(
                                "GPU compute: timestamp-query unavailable in this browser",
                            )
                            .small()
                            .color(egui::Color32::from_gray(160)),
                        );
                    }
                }
            }
        }

        let fps = if self.frame_dt_ms > 0.0 {
            1000.0 / self.frame_dt_ms
        } else {
            0.0
        };

        ui.add_space(6.0);
        ui.label(format!(
            "30-second compute: avg {:.2} ms · p99 {:.2} ms",
            self.fps_graph.avg_cpu_ms(),
            self.fps_graph.p99_cpu_ms(),
        ));
        self.fps_graph.show(ui);
    }

    fn tab_import(&mut self, ui: &mut egui::Ui) {
        ui.strong("Import RLE pattern");
        ui.horizontal(|ui| {
            ui.label("Browse patterns:");
            ui.hyperlink_to(
                "conwaylife.com/wiki/Category:Patterns",
                "https://conwaylife.com/wiki/Category:Patterns",
            );
        });

        ui.add_space(4.0);

        egui::ScrollArea::vertical()
            .id_salt("rle_scroll")
            .auto_shrink([false, false])
            .max_height(280.0)
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut self.rle_text)
                        .desired_width(f32::INFINITY)
                        .desired_rows(14)
                        .hint_text("x = 3, y = 3, rule = B3/S23\nbob$2bo$3o!"),
                );
            });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button("⤓ Clear grid & stamp at centre").clicked() {
                self.try_load_rle();
            }
            if ui.button("clear text").clicked() {
                self.rle_text.clear();
                self.rle_status = None;
            }
        });
        if let Some((ok, msg)) = &self.rle_status {
            let col = if *ok {
                egui::Color32::from_rgb(80, 200, 120)
            } else {
                egui::Color32::from_rgb(220, 100, 90)
            };
            ui.colored_label(col, msg);
        }
    }

    fn tab_about(&mut self, ui: &mut egui::Ui) {
        ui.strong("Game of Life on the GPU, in Rust");
        ui.add_space(4.0);
        ui.label("Compute kernel: rust-gpu → SPIR-V → naga → WGSL");
        ui.label("Render: WGSL fullscreen triangle");
        ui.label("UI: egui via eframe + egui_wgpu callback");
        ui.add_space(8.0);
        ui.label(
            "Toggle CPU/GPU in the Sim tab to compare the same B3/S23 \
             kernel running sequentially in Rust vs in parallel on the GPU. \
             Per-tick timings and a 30-second compute chart live below the \
             controls on the same tab.",
        );
    }

    fn show_grid(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(8, 9, 11)))
            .show_inside(ui, |ui| {
                if !self.has_wgpu {
                    ui.centered_and_justified(|ui| {
                        ui.heading("WebGPU required");
                    });
                    return;
                }

                let available = ui.available_size();
                let edge = available.x.min(available.y);
                let (rect, response) =
                    ui.allocate_exact_size(egui::vec2(edge, edge), egui::Sense::click_and_drag());

                if response.drag_started() {
                    self.painting = true;
                }
                if response.drag_stopped() {
                    self.painting = false;
                }
                if (self.painting || response.clicked())
                    && let Some(pos) = response.hover_pos()
                    && rect.contains(pos)
                {
                    let u = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 0.999_999);
                    let v = ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 0.999_999);
                    let gx = (u * self.grid_size as f32) as u32;
                    let gy = (v * self.grid_size as f32) as u32;
                    let radius: i32 = if self.grid_size >= 1024 { 2 } else { 1 };
                    for dy in -radius..=radius {
                        for dx in -radius..=radius {
                            let x = gx as i32 + dx;
                            let y = gy as i32 + dy;
                            if x >= 0
                                && y >= 0
                                && (x as u32) < self.grid_size
                                && (y as u32) < self.grid_size
                            {
                                self.pending_paint.push((x as u32, y as u32, 1));
                            }
                        }
                    }
                }

                let cb = GolCallback {
                    set_backend: self.pending_backend.take(),
                    resize: self.pending_resize.take(),
                    clear: std::mem::take(&mut self.pending_clear),
                    // Pull a fresh seed from the OS each time the user hits
                    // "Random" — on wasm this reaches `crypto.getRandomValues`
                    // via the rand crate's `wasm_js` getrandom backend.
                    randomize: if std::mem::take(&mut self.pending_randomize) {
                        Some(rand::random::<u64>())
                    } else {
                        None
                    },
                    paint: std::mem::take(&mut self.pending_paint),
                    ticks: if self.running {
                        self.ticks_per_frame
                    } else if std::mem::take(&mut self.pending_step) {
                        1
                    } else {
                        0
                    },
                };
                ui.painter()
                    .add(egui_wgpu::Callback::new_paint_callback(rect, cb));
            });
    }
}

impl eframe::App for GameOfLifeApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        self.sample_diagnostics(frame);
        self.show_side_panel(ui);
        self.show_grid(ui);
        ui.ctx().request_repaint();
    }

    #[cfg(target_arch = "wasm32")]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(&mut *self)
    }
}
