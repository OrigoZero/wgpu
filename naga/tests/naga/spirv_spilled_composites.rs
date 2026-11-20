//! A composite value subscripted with a runtime index is spilled to a
//! function-local temporary. The value is stored there once, where it is
//! computed, and every access reads that store: a loop over the composite
//! stores nothing per iteration.

const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_MATRIX: u32 = 24;
const OP_TYPE_POINTER: u32 = 32;
const OP_FUNCTION: u32 = 54;
const OP_FUNCTION_END: u32 = 56;
const OP_VARIABLE: u32 = 59;
const OP_STORE: u32 = 62;
const OP_LOOP_MERGE: u32 = 246;
const STORAGE_CLASS_FUNCTION: u32 = 7;

/// Where each store into a function-local composite temporary stands in its
/// function: `(function index, inside a loop)`.
struct Stores {
    stores: Vec<(usize, bool)>,
}

fn spirv(source: &str, entry_point: &str, stage: naga::ShaderStage) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    );
    let info = validator.validate(&module).unwrap();
    let pipeline_options = naga::back::spv::PipelineOptions {
        entry_point: entry_point.to_string(),
        shader_stage: stage,
    };
    let options = naga::back::spv::Options::default();
    let words =
        naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline_options)).unwrap();

    // The module the writer produced reads back as valid SPIR-V.
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let parsed = naga::front::spv::parse_u8_slice(
        &bytes,
        &naga::front::spv::Options {
            adjust_coordinate_space: false,
            strict_capabilities: false,
            block_ctx_dump_prefix: None,
        },
    )
    .unwrap();
    validator.validate(&parsed).unwrap();
    words
}

/// The stores into function-local variables of array or matrix type, with
/// whether each one stands after an `OpLoopMerge` of its function (the loop
/// header precedes its body in a structured SPIR-V function, so a store after
/// the function's first loop header is a store in or after that loop).
fn composite_stores(words: &[u32]) -> Stores {
    let mut composite_types = std::collections::HashSet::new();
    let mut composite_pointers = std::collections::HashSet::new();
    let mut composite_variables = std::collections::HashSet::new();
    let mut stores = Vec::new();
    let mut function = 0usize;
    let mut after_loop = false;
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let op = words[at] & 0xffff;
        let operands = &words[at + 1..at + count];
        match op {
            OP_TYPE_ARRAY | OP_TYPE_MATRIX => {
                composite_types.insert(operands[0]);
            }
            OP_TYPE_POINTER
                if operands[1] == STORAGE_CLASS_FUNCTION
                    && composite_types.contains(&operands[2]) =>
            {
                composite_pointers.insert(operands[0]);
            }
            OP_VARIABLE
                if operands[2] == STORAGE_CLASS_FUNCTION
                    && composite_pointers.contains(&operands[0]) =>
            {
                composite_variables.insert(operands[1]);
            }
            OP_FUNCTION => {
                function += 1;
                after_loop = false;
            }
            OP_FUNCTION_END => after_loop = false,
            OP_LOOP_MERGE => after_loop = true,
            OP_STORE if composite_variables.contains(&operands[0]) => {
                stores.push((function, after_loop))
            }
            _ => {}
        }
        at += count;
    }
    Stores { stores }
}

#[test]
fn a_called_composite_indexed_in_a_loop_is_stored_once_before_the_loop() {
    let source = r#"
struct Params { a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec4<f32>, count: u32 }
@group(0) @binding(0) var<uniform> params: Params;

fn point_list() -> array<vec4<f32>, 4> {
    return array<vec4<f32>, 4>(params.a, params.b, params.c, params.d);
}

@fragment
fn main() -> @location(0) vec4<f32> {
    let points = point_list();
    var sum = vec4<f32>(0.0);
    for (var i = 0u; i < params.count; i = i + 1u) {
        let p = points[i];
        if p.w > 0.5 {
            sum = sum + points[i] * 2.0;
        } else {
            sum = sum + points[(i + 1u) % 4u];
        }
    }
    return sum;
}
"#;
    let stores = composite_stores(&spirv(source, "main", naga::ShaderStage::Fragment)).stores;
    assert_eq!(
        stores.len(),
        1,
        "one store of `points` for its three runtime-indexed accesses: {stores:?}"
    );
    assert!(
        !stores[0].1,
        "the store stands where `points` is computed, before the loop: {stores:?}"
    );
}

#[test]
fn a_composed_array_and_a_matrix_indexed_in_a_loop_are_each_stored_once_before_it() {
    let source = r#"
struct Params { a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec4<f32>, count: u32 }
@group(0) @binding(0) var<uniform> params: Params;

@fragment
fn main() -> @location(0) vec4<f32> {
    let points = array<vec4<f32>, 4>(params.a, params.b, params.c, params.d);
    let m = mat4x4<f32>(params.d, params.c, params.b, params.a);
    var sum = vec4<f32>(0.0);
    for (var i = 0u; i < params.count; i = i + 1u) {
        sum = sum + points[i % 4u] + m[i % 4u];
        sum = sum * points[(i + 2u) % 4u].x;
    }
    return sum;
}
"#;
    let stores = composite_stores(&spirv(source, "main", naga::ShaderStage::Fragment)).stores;
    assert_eq!(
        stores.len(),
        2,
        "one store each of `points` and `m`: {stores:?}"
    );
    assert!(
        stores.iter().all(|s| !s.1),
        "both stores stand before the loop: {stores:?}"
    );
}

#[test]
fn a_composite_computed_in_each_iteration_is_stored_once_in_it() {
    let source = r#"
struct Params { a: vec4<f32>, b: vec4<f32>, count: u32 }
@group(0) @binding(0) var<uniform> params: Params;

@fragment
fn main() -> @location(0) vec4<f32> {
    var sum = vec4<f32>(0.0);
    for (var i = 0u; i < params.count; i = i + 1u) {
        let pair = array<vec4<f32>, 2>(params.a * f32(i), params.b);
        sum = sum + pair[i % 2u] + pair[(i + 1u) % 2u];
    }
    return sum;
}
"#;
    let stores = composite_stores(&spirv(source, "main", naga::ShaderStage::Fragment)).stores;
    assert_eq!(
        stores.len(),
        1,
        "one store of `pair`, where each iteration computes it: {stores:?}"
    );
    assert!(
        stores[0].1,
        "the store stands in the loop that computes `pair`: {stores:?}"
    );
}

#[test]
fn a_composite_argument_indexed_in_a_loop_is_stored_once_at_the_function_entry() {
    let source = r#"
struct Params { a: vec4<f32>, b: vec4<f32>, count: u32 }
@group(0) @binding(0) var<uniform> params: Params;

fn total(points: array<vec4<f32>, 2>, count: u32) -> vec4<f32> {
    var sum = vec4<f32>(0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        sum = sum + points[i % 2u] * points[(i + 1u) % 2u];
    }
    return sum;
}

@fragment
fn main() -> @location(0) vec4<f32> {
    return total(array<vec4<f32>, 2>(params.a, params.b), params.count);
}
"#;
    let stores = composite_stores(&spirv(source, "main", naga::ShaderStage::Fragment)).stores;
    assert_eq!(
        stores.len(),
        1,
        "one store of the argument `points`: {stores:?}"
    );
    assert!(
        !stores[0].1,
        "the store stands at the function's entry, before its loop: {stores:?}"
    );
}
