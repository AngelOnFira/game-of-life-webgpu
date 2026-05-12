//! Cross-compile the `gol-shader` crate to SPIR-V using `spirv-builder`.
//!
//! Sets `gol_shader.spv` as an env var (hyphens → underscores) pointing to
//! the resulting `.spv` file, which `pipelines.rs` reads via
//! `include_bytes!(env!("gol_shader.spv"))`.

use spirv_builder::SpirvBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("SKIP_SHADER_BUILD").is_ok() {
        let placeholder = std::env::current_dir()?.join("../shader/placeholder.spv");
        if !placeholder.exists() {
            let header: [u32; 5] = [0x07230203, 0x00010000, 0, 0, 0];
            let bytes: Vec<u8> = header.iter().flat_map(|w| w.to_le_bytes()).collect();
            std::fs::write(&placeholder, bytes)?;
        }
        println!("cargo::rustc-env=gol_shader.spv={}", placeholder.display());
        return Ok(());
    }

    let mut builder = SpirvBuilder::new("../shader", "spirv-unknown-vulkan1.1");
    builder.build_script.defaults = true;
    builder.build_script.env_shader_spv_path = Some(true);
    builder.build()?;
    Ok(())
}
