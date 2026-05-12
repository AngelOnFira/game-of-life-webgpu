//! Verify the rust-gpu-compiled SPIR-V translates cleanly to WGSL via naga —
//! exactly what happens in the browser at runtime through wgpu's `webgpu`
//! backend. If this test fails, the demo will fail to start.

const SPV: &[u8] = include_bytes!(env!("gol_shader.spv"));

#[test]
fn spv_parses() {
    let module = naga::front::spv::parse_u8_slice(SPV, &naga::front::spv::Options::default())
        .expect("rust-gpu output should parse as SPIR-V");
    assert_eq!(module.entry_points.len(), 1, "expected one entry point (gol_step)");
    assert_eq!(module.entry_points[0].stage, naga::ShaderStage::Compute);
}

#[test]
fn spv_validates() {
    let module = naga::front::spv::parse_u8_slice(SPV, &naga::front::spv::Options::default())
        .expect("parse");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .expect("naga should validate the SPIR-V module");
}

#[test]
fn spv_translates_to_wgsl() {
    let module = naga::front::spv::parse_u8_slice(SPV, &naga::front::spv::Options::default())
        .expect("parse");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator.validate(&module).expect("validate");

    let mut wgsl = String::new();
    let mut writer =
        naga::back::wgsl::Writer::new(&mut wgsl, naga::back::wgsl::WriterFlags::empty());
    writer.write(&module, &info).expect("WGSL emission");

    // Sanity: the bind-group layout used by `pipelines.rs` must survive translation.
    assert!(
        wgsl.contains("@group(0) @binding(0)"),
        "missing params binding\n{wgsl}"
    );
    assert!(
        wgsl.contains("@group(0) @binding(1)"),
        "missing src binding\n{wgsl}"
    );
    assert!(
        wgsl.contains("@group(0) @binding(2)"),
        "missing dst binding\n{wgsl}"
    );
    assert!(wgsl.contains("storage"), "missing storage qualifier\n{wgsl}");
}
