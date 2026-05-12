// Conway's Game of Life — compute step (reference WGSL implementation).
//
// **Not currently used** by the app — the live kernel is the rust-gpu one in
// `shader/src/lib.rs`. Kept here as a side-by-side reference for readers
// curious how the same algorithm looks in WGSL, and as a fallback you can
// drop into `gpu.rs` (swap `make_spirv` for `ShaderSource::Wgsl`) if you
// need to bisect a rust-gpu translation issue.
//
// B3/S23 on a toroidal grid. One u32 per cell (0 = dead, 1 = alive).
// 8×8 workgroups = 64 invocations per group (safe across all WebGPU adapters).

struct Params {
    width: u32,
    height: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(8, 8)
fn gol_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    if (gid.x >= w || gid.y >= h) { return; }

    let x = gid.x;
    let y = gid.y;
    let xm = (x + w - 1u) % w;
    let xp = (x + 1u) % w;
    let ym = (y + h - 1u) % h;
    let yp = (y + 1u) % h;

    let n =
        src[ym * w + xm] +
        src[ym * w + x ] +
        src[ym * w + xp] +
        src[y  * w + xm] +
        src[y  * w + xp] +
        src[yp * w + xm] +
        src[yp * w + x ] +
        src[yp * w + xp];

    let me = src[y * w + x];
    var next: u32 = 0u;
    if (me == 1u && (n == 2u || n == 3u)) { next = 1u; }
    if (me == 0u && n == 3u)              { next = 1u; }
    dst[y * w + x] = next;
}
