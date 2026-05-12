//! Reference CPU implementation of the B3/S23 step on a toroidal grid.
//!
//! Maintained alongside the GPU compute pipeline so the user can A/B-test
//! "rust-on-GPU" vs "rust-on-CPU" performance and convince themselves the
//! two pipelines produce identical state.
//!
//! The CPU keeps an owned `Vec<u32>` (one cell per `u32` for simple memory
//! layout — same encoding as the GPU buffer). `step` runs one generation
//! into a double-buffer, then swaps.

pub struct CpuSim {
    size: u32,
    front: Vec<u32>,
    back: Vec<u32>,
}

impl CpuSim {
    pub fn new(size: u32) -> Self {
        let cells = (size as usize).pow(2);
        Self {
            size,
            front: vec![0; cells],
            back: vec![0; cells],
        }
    }

    pub fn resize(&mut self, new_size: u32) {
        let cells = (new_size as usize).pow(2);
        self.front = vec![0; cells];
        self.back = vec![0; cells];
        self.size = new_size;
    }

    pub fn clear(&mut self) {
        self.front.fill(0);
        self.back.fill(0);
    }

    /// Replace `front` with externally-supplied data (e.g. a GPU readback).
    /// Silently no-ops if the length doesn't match the configured grid.
    pub fn adopt(&mut self, data: Vec<u32>) {
        let expected = (self.size as usize).pow(2);
        if data.len() != expected {
            log::warn!(
                "CpuSim::adopt: ignoring buffer of len {} (expected {})",
                data.len(),
                expected
            );
            return;
        }
        self.front = data;
        self.back.fill(0);
    }

    /// Returns a reference to the current grid for upload.
    pub fn data(&self) -> &[u32] {
        &self.front
    }

    pub fn randomize(&mut self, seed: u64) -> &[u32] {
        // Deterministic ChaCha-based PRNG seeded from the user-supplied seed —
        // identical sequence on native and wasm, no OS entropy required.
        use rand::{RngExt as _, SeedableRng as _};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        for cell in self.front.iter_mut() {
            // ~25% alive matches a classic "primordial soup" GoL start.
            *cell = u32::from(rng.random_bool(0.25));
        }
        &self.front
    }

    pub fn stamp(&mut self, cells: &[(u32, u32, u32)]) {
        let n = self.size as usize;
        for &(x, y, v) in cells {
            if (x as usize) < n && (y as usize) < n {
                self.front[(y as usize) * n + (x as usize)] = v;
            }
        }
    }

    /// One B3/S23 step. `front` → `back`, then swap.
    pub fn step(&mut self) {
        let w = self.size as usize;
        let h = w;
        for y in 0..h {
            let ym = if y == 0 { h - 1 } else { y - 1 };
            let yp = if y + 1 == h { 0 } else { y + 1 };
            let yw = y * w;
            let ymw = ym * w;
            let ypw = yp * w;
            for x in 0..w {
                let xm = if x == 0 { w - 1 } else { x - 1 };
                let xp = if x + 1 == w { 0 } else { x + 1 };
                let n = self.front[ymw + xm]
                    + self.front[ymw + x]
                    + self.front[ymw + xp]
                    + self.front[yw + xm]
                    + self.front[yw + xp]
                    + self.front[ypw + xm]
                    + self.front[ypw + x]
                    + self.front[ypw + xp];
                let me = self.front[yw + x];
                self.back[yw + x] = ((me == 1 && (n == 2 || n == 3)) || (me == 0 && n == 3)) as u32;
            }
        }
        std::mem::swap(&mut self.front, &mut self.back);
    }
}
