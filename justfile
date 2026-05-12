# Conway's Game of Life — Rust GPU on WebGPU demo
#
# Common commands. Run `just` (no args) to see this list.
#
# Everything builds in release mode by default. Dev compile is dramatically
# slower for naga + wgpu (egui jitters), and the shader and engine code is
# the interesting part anyway — we want it running at proper speed locally
# too, not just in CI.

# The repo name as it appears on GitHub Pages — used for --public-url.
pages_path := "/wasm-rustgpu-game-of-life/"

# Show available recipes.
default:
    @just --list

# Serve the demo locally on http://127.0.0.1:8265 (auto-rebuilds on file changes).
serve:
    trunk serve --release --open

# Same as `serve` but binds to all interfaces so other devices on the LAN can connect.
serve-public:
    trunk serve --release --address 0.0.0.0

# Build a release bundle into ./dist (no Pages prefix; for local inspection).
build:
    trunk build --release

# Build a production bundle ready for GitHub Pages (release + public-url prefix).
build-release:
    trunk build --release --public-url {{pages_path}}

# Run SPIR-V → WGSL round-trip tests (the translation the browser does at runtime).
test:
    cargo test --release --manifest-path app/Cargo.toml --tests

# Lint with clippy. Targets wasm32 because that's the platform that ships.
lint:
    cargo clippy --release --target wasm32-unknown-unknown --manifest-path app/Cargo.toml -- -D warnings

# Format both crates.
fmt:
    cargo fmt --manifest-path app/Cargo.toml --all
    cargo fmt --manifest-path shader/Cargo.toml --all

# Full pre-push check: format, lint, test, release build.
ci: fmt lint test build-release

# Wipe target dirs (host + shader nested build + trunk output).
clean:
    rm -rf target shader/target dist

# Force a rebuild of the rust-gpu shader when incremental detection misses a change.
shader-rebuild:
    touch shader/src/lib.rs
    cargo build --release --manifest-path app/Cargo.toml --target wasm32-unknown-unknown
