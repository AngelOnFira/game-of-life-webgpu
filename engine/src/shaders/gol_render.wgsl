// Conway's Game of Life — render
//
// Fullscreen-triangle vertex shader (no vertex buffer; vertex index in [0,3)).
// Fragment samples the grid storage buffer and emits monochrome pixels.

struct Params {
    width: u32,
    height: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grid: array<u32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Oversized triangle covering NDC [-1,1]² with UV [0,1]² mapped to it.
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: VsOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv  = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // UV is roughly [0,1]² inside the viewport; clamp for safety.
    let u = clamp(in.uv.x, 0.0, 1.0);
    let v = clamp(in.uv.y, 0.0, 1.0);
    let gx = min(u32(u * f32(params.width )), params.width  - 1u);
    let gy = min(u32(v * f32(params.height)), params.height - 1u);
    let alive = grid[gy * params.width + gx];

    // Subtle dimming when dead so the grid background is dark grey, not pure black.
    let on  = vec3<f32>(0.95, 0.95, 0.90);
    let off = vec3<f32>(0.05, 0.06, 0.08);
    let col = mix(off, on, f32(alive));
    return vec4<f32>(col, 1.0);
}
