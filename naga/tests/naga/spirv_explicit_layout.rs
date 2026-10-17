/*!
Test that the SPIR-V backend puts explicit layout decorations only on the types
of variables in storage classes that take them.

Vulkan refuses `Offset`, `ArrayStride`, `MatrixStride`, `ColMajor` and
`RowMajor` on the type of a variable in `Workgroup` storage, and in `Function`
and `Private` storage from SPIR-V 1.5 on (`VUID-StandaloneSpirv-None-10684`).
Before SPIR-V 1.5 a `Function` or `Private` variable of a type host-shareable
storage holds keeps that type's layout, and is loaded and stored as it is.
*/

#![cfg(all(feature = "wgsl-in", spv_out))]

use hashbrown::{HashMap, HashSet};

use rspirv::dr::{Instruction, Module, Operand};
use spirv::{Decoration, Op, StorageClass, Word};

/// A shader whose struct and array types are held both by host-shareable
/// variables and by `Function`, `Private` and `Workgroup` variables, and copied
/// between them whole, by member, by element, through pointer arguments, by
/// value arguments, by initializers and by `workgroupUniformLoad`.
const SHARED_TYPES: &str = r#"
struct Inner { m: mat3x3<f32>, v: array<vec4<i32>, 4> }
struct S { a: vec4<f32>, inner: Inner, list: array<Inner, 2>, b: u32 }
struct Buf { items: array<S, 4>, n: u32, tail: array<u32> }
@group(0) @binding(0) var<storage, read_write> buf: Buf;
@group(0) @binding(1) var<uniform> uni: S;
var<private> p: S;
var<private> pinit: array<vec4<i32>, 4> = array(vec4(1), vec4(2), vec4(3), vec4(4));
var<workgroup> w: array<S, 2>;
var<workgroup> wu: S;
const C = Inner(mat3x3<f32>(), array(vec4(5), vec4(6), vec4(7), vec4(8)));

fn by_ptr(x: ptr<function, S>) -> u32 { (*x).b += 1u; return u32((*x).inner.v[1].y); }
fn by_val(arr: array<vec4<i32>, 4>, i: u32) -> i32 { return arr[i].x; }
fn by_private(x: ptr<private, S>) { (*x).b = 7u; }

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32) {
    var local: S = buf.items[li % 4u];
    var c: Inner = C;
    var z: S;
    p = uni;
    p.list[1] = buf.items[0].inner;
    w[li % 2u] = buf.items[li % 4u];
    wu = local;
    let shared_copy = workgroupUniformLoad(&wu);
    local.inner = w[0].list[li % 2u];
    let r = by_ptr(&local);
    by_private(&p);
    let arr = local.inner.v;
    let picked = by_val(arr, li % 4u) + arr[li % 4u].w + pinit[li % 4u].z + c.v[li % 4u].x;
    buf.items[li % 4u] = local;
    buf.items[0].inner = p.list[li % 2u];
    buf.items[1] = z;
    buf.n = r + u32(picked) + shared_copy.b + p.b;
    buf.tail[li] = u32(w[1].inner.m[0].x);
}
"#;

/// A shader whose struct and array types only `Function`, `Private` and
/// `Workgroup` variables hold.
const LOCAL_TYPES: &str = r#"
struct L { a: vec4<f32>, m: mat2x2<f32>, v: array<f32, 8> }
var<private> p: L;
var<workgroup> w: array<L, 4>;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(4)
fn main(@builtin(local_invocation_index) li: u32) {
    var l: L;
    l.v[li] = f32(li);
    w[li] = l;
    p = w[(li + 1u) % 4u];
    out[li] = p.v[li] + p.m[0].x;
}
"#;

fn write_spirv(source: &str, lang_version: (u8, u8)) -> Module {
    use naga::back::spv;
    use naga::valid;

    let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
        panic!(
            "expected WGSL to parse successfully:\n{}",
            e.emit_to_string(source)
        );
    });
    let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
        .validate(&module)
        .expect("validation failed");
    let options = spv::Options {
        lang_version,
        ..spv::Options::default()
    };
    let words = spv::write_vec(&module, &info, &options, None).unwrap();
    rspirv::dr::load_words(words).expect("Produced invalid SPIR-V")
}

fn id_operand(operand: &Operand) -> Word {
    match *operand {
        Operand::IdRef(id) => id,
        ref other => panic!("expected an id operand, found {other:?}"),
    }
}

const LAYOUT_DECORATIONS: [Decoration; 5] = [
    Decoration::Offset,
    Decoration::ArrayStride,
    Decoration::MatrixStride,
    Decoration::ColMajor,
    Decoration::RowMajor,
];

/// The ids of the types that carry a layout decoration, on themselves or on a
/// member.
fn laid_out_types(module: &Module) -> HashSet<Word> {
    let mut laid_out = HashSet::new();
    for annotation in &module.annotations {
        let (target, decoration) = match annotation.class.opcode {
            Op::Decorate => (&annotation.operands[0], &annotation.operands[1]),
            Op::MemberDecorate => (&annotation.operands[0], &annotation.operands[2]),
            _ => continue,
        };
        if let Operand::Decoration(decoration) = *decoration {
            if LAYOUT_DECORATIONS.contains(&decoration) {
                laid_out.insert(id_operand(target));
            }
        }
    }
    laid_out
}

/// Every type `id` holds, itself included.
fn reachable_types(types: &HashMap<Word, &Instruction>, id: Word, into: &mut HashSet<Word>) {
    if !into.insert(id) {
        return;
    }
    let Some(instruction) = types.get(&id) else {
        return;
    };
    match instruction.class.opcode {
        Op::TypeStruct => {
            for member in &instruction.operands {
                reachable_types(types, id_operand(member), into);
            }
        }
        Op::TypeArray | Op::TypeRuntimeArray => {
            reachable_types(types, id_operand(&instruction.operands[0]), into);
        }
        _ => {}
    }
}

/// Every type a pointer in `classes` points to, with every type it holds.
fn types_held_in(module: &Module, classes: &[StorageClass]) -> HashSet<Word> {
    let types: HashMap<Word, &Instruction> = module
        .types_global_values
        .iter()
        .filter_map(|instruction| instruction.result_id.map(|id| (id, instruction)))
        .collect();
    let mut held = HashSet::new();
    for instruction in &module.types_global_values {
        if instruction.class.opcode != Op::TypePointer {
            continue;
        }
        let Operand::StorageClass(class) = instruction.operands[0] else {
            panic!("OpTypePointer without a storage class");
        };
        if classes.contains(&class) {
            reachable_types(&types, id_operand(&instruction.operands[1]), &mut held);
        }
    }
    held
}

/// The storage classes whose variables refuse explicit layout at SPIR-V
/// `lang_version`.
fn refusing_classes(lang_version: (u8, u8)) -> &'static [StorageClass] {
    if lang_version >= (1, 5) {
        &[
            StorageClass::Function,
            StorageClass::Private,
            StorageClass::Workgroup,
        ]
    } else {
        &[StorageClass::Workgroup]
    }
}

const LANG_VERSIONS: [(u8, u8); 5] = [(1, 0), (1, 3), (1, 4), (1, 5), (1, 6)];

fn assert_no_layout_in_refusing_classes(source: &str) {
    for lang_version in LANG_VERSIONS {
        let module = write_spirv(source, lang_version);
        let laid_out = laid_out_types(&module);
        let held = types_held_in(&module, refusing_classes(lang_version));
        let offending: Vec<Word> = held.intersection(&laid_out).copied().collect();
        assert!(
            offending.is_empty(),
            "SPIR-V {lang_version:?}: types {offending:?} carry explicit layout and are held \
             by variables of {:?}",
            refusing_classes(lang_version)
        );
    }
}

/// Whether any function of `module` holds an instruction of `opcode`.
fn uses(module: &Module, opcode: Op) -> bool {
    module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .any(|instruction| instruction.class.opcode == opcode)
}

#[test]
fn types_shared_with_host_shareable_storage_are_held_without_layout() {
    assert_no_layout_in_refusing_classes(SHARED_TYPES);
}

#[test]
fn types_only_local_storage_holds_carry_no_layout() {
    assert_no_layout_in_refusing_classes(LOCAL_TYPES);
    let module = write_spirv(LOCAL_TYPES, (1, 6));
    let layout_decorations = module
        .annotations
        .iter()
        .filter(|annotation| {
            annotation.operands.iter().any(|operand| {
                matches!(*operand, Operand::Decoration(decoration)
                    if LAYOUT_DECORATIONS.contains(&decoration))
            })
        })
        .count();
    // The storage buffer's runtime array and its wrapper struct's member.
    assert_eq!(layout_decorations, 2);
}

#[test]
fn host_shareable_storage_keeps_its_layout() {
    for lang_version in [(1, 0), (1, 6)] {
        let module = write_spirv(SHARED_TYPES, lang_version);
        let laid_out = laid_out_types(&module);
        let held = types_held_in(
            &module,
            &[StorageClass::StorageBuffer, StorageClass::Uniform],
        );
        let types: HashMap<Word, &Instruction> = module
            .types_global_values
            .iter()
            .filter_map(|instruction| instruction.result_id.map(|id| (id, instruction)))
            .collect();
        for id in held {
            let opcode = types[&id].class.opcode;
            if matches!(
                opcode,
                Op::TypeStruct | Op::TypeArray | Op::TypeRuntimeArray
            ) {
                assert!(
                    laid_out.contains(&id),
                    "SPIR-V {lang_version:?}: {opcode:?} {id} in host-shareable storage has no layout"
                );
            }
        }
    }
}

#[test]
fn a_value_moves_between_layouts_with_copy_logical_from_spirv_1_4() {
    for (lang_version, copies) in [((1, 3), false), ((1, 4), true), ((1, 6), true)] {
        let module = write_spirv(SHARED_TYPES, lang_version);
        let copy_logical = uses(&module, Op::CopyLogical);
        assert_eq!(
            copy_logical, copies,
            "SPIR-V {lang_version:?}: OpCopyLogical used: {copy_logical}"
        );
    }
}

/// A shader whose struct and array types a storage buffer holds and only
/// `Function` and `Private` variables hold besides, copied between them whole.
const FUNCTION_AND_PRIVATE_SHARED_TYPES: &str = r#"
struct S { a: vec4<f32>, m: mat2x3<f32>, v: array<vec4<i32>, 4> }
@group(0) @binding(0) var<storage, read_write> buf: array<S, 2>;
var<private> p: S;

@compute @workgroup_size(1)
fn main(@builtin(local_invocation_index) li: u32) {
    var local = buf[0];
    p = buf[1];
    buf[li % 2u] = local;
    buf[0].v = p.v;
}
"#;

#[test]
fn function_and_private_variables_hold_the_laid_out_type_before_spirv_1_5() {
    for lang_version in LANG_VERSIONS {
        let module = write_spirv(FUNCTION_AND_PRIVATE_SHARED_TYPES, lang_version);
        let laid_out = laid_out_types(&module);
        let held = types_held_in(&module, &[StorageClass::Function, StorageClass::Private]);
        let holds_laid_out = !held.is_disjoint(&laid_out);
        let converts = uses(&module, Op::CopyLogical) || uses(&module, Op::CompositeConstruct);
        assert_eq!(
            (holds_laid_out, converts),
            (lang_version < (1, 5), lang_version >= (1, 5)),
            "SPIR-V {lang_version:?}: Function or Private variables hold a laid out type: \
             {holds_laid_out}, loads and stores convert between layouts: {converts}"
        );
    }
}
