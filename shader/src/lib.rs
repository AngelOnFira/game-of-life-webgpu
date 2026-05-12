//! Conway's Game of Life compute kernel, authored in Rust via `rust-gpu`.
//!
//! Cross-compiled to SPIR-V (`spirv-unknown-vulkan1.1`) by `app/build.rs` using
//! `spirv-builder`. The resulting binary is consumed by the app via
//! `include_bytes!(env!("gol_shader.spv"))`, then translated to WGSL inside
//! the browser by naga at runtime (transparently through wgpu's `spirv`
//! feature).
//!
//! Bindings:
//! - `set=0 binding=0` uniform `Params { width, height }`
//! - `set=0 binding=1` storage_buffer read-only — previous generation
//! - `set=0 binding=2` storage_buffer read-write — next generation
//!
//! These must match the WGSL render shader's bind-group layout and the
//! compute pipeline layout declared in `app/src/gpu.rs`.

#![cfg_attr(target_arch = "spirv", no_std)]
// HACK from upstream rust-gpu examples: ensures we see warnings emitted during
// nested spirv build.
#![cfg_attr(target_arch = "spirv", deny(warnings))]

use spirv_std::glam::UVec3;
use spirv_std::spirv;

/// Matches the WGSL `struct Params { width: u32, height: u32 }`.
/// `repr(C)` keeps the layout deterministic across both languages.
#[repr(C)]
pub struct Params {
    pub width: u32,
    pub height: u32,
}

/// Conway's B3/S23 step on a toroidal grid.
///
/// Workgroup size 8×8 = 64 invocations — well within every WebGPU adapter's
/// `maxComputeInvocationsPerWorkgroup` (≥256).
#[spirv(compute(threads(8, 8)))]
pub fn gol_step(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &Params,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] src: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dst: &mut [u32],
) {
    let w = params.width;
    let h = params.height;
    let x = gid.x;
    let y = gid.y;
    if x >= w || y >= h {
        return;
    }

    // Wrap into [0, dim) without modulo by adding dim then masking with modulo
    // — modulo on u32 is well-defined in SPIR-V; the form `(x + w - 1) % w`
    // avoids signed/underflow concerns.
    let xm = (x + w - 1) % w;
    let xp = (x + 1) % w;
    let ym = (y + h - 1) % h;
    let yp = (y + 1) % h;

    let idx = |xi: u32, yi: u32| -> usize { (yi * w + xi) as usize };

    let n = src[idx(xm, ym)]
        + src[idx(x, ym)]
        + src[idx(xp, ym)]
        + src[idx(xm, y)]
        + src[idx(xp, y)]
        + src[idx(xm, yp)]
        + src[idx(x, yp)]
        + src[idx(xp, yp)];

    let me = src[idx(x, y)];
    // B3/S23: born if dead with exactly 3 neighbours, survives if alive with 2 or 3.
    let alive_next = (me == 1 && (n == 2 || n == 3)) || (me == 0 && n == 3);
    dst[idx(x, y)] = alive_next as u32;
}
