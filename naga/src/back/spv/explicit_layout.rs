/*!
Explicit layout decorations, and the types that hold values without them.

Vulkan takes the explicit layout decorations (`Offset`, `ArrayStride`,
`MatrixStride`, `ColMajor`) on the types of variables in the storage classes
that share memory with the host: `Uniform`, `StorageBuffer`, `PushConstant`.
It refuses them on the type of a variable in `Workgroup` storage at every
SPIR-V version, and in `Function` and `Private` storage from SPIR-V 1.5 on
(`VUID-StandaloneSpirv-None-10684`).

So the writer decorates exactly the types reachable from a global in a storage
class that takes explicit layout ([`Writer::collect_explicit_layout_types`]),
and declares every other type without layout. A type that is both, such as a
struct a storage buffer holds and a workgroup variable holds too, has two
declarations:

- The explicitly laid out declaration, which is the type of every SPIR-V value
  of that Naga type and of every variable in a storage class that accepts
  explicit layout at the module's SPIR-V version.

- Its *implicit-layout variant* ([`Writer::implicit_layout_type_id`]): the same
  members or elements, declared without decorations, holding implicit-layout
  variants of its own members. A variable in a storage class that refuses
  explicit layout at the module's SPIR-V version ([`refuses_explicit_layout`])
  holds this variant.

Before SPIR-V 1.5 a `Function` or `Private` variable holds the type of its
values, laid out or not, which those versions accept, so it is loaded and stored
with no conversion.

A load from such a variable reads the variant and converts the value to the
explicitly laid out type; a store converts the other way
([`Writer::write_layout_conversion`]). From SPIR-V 1.4 on the conversion is
one `OpCopyLogical`; before it, the value is taken apart and put back together
member by member.
*/

use alloc::{string::String, vec::Vec};

use spirv::Word;

use super::{Instruction, Writer};

/// Whether a variable in `class` must hold a type without explicit layout in a
/// module of SPIR-V `version` (the version word of the module's header).
pub(super) const fn refuses_explicit_layout(class: spirv::StorageClass, version: Word) -> bool {
    match class {
        spirv::StorageClass::Workgroup => true,
        spirv::StorageClass::Function | spirv::StorageClass::Private => version >= 0x10500,
        _ => false,
    }
}

/// The members of a declared struct or the element of a declared fixed-size
/// array, from which the writer declares the type's implicit-layout variant.
#[derive(Clone)]
pub(super) enum CompositeShape {
    Struct {
        /// The SPIR-V type of each member, in order.
        members: Vec<Word>,
        /// The Naga IR name of each member, for `OpMemberName`.
        member_names: Vec<Option<String>>,
        /// The Naga IR name of the struct, for `OpName`.
        name: Option<String>,
    },
    Array {
        element: Word,
        /// The id of the `u32` constant holding `length`.
        length_id: Word,
        length: u32,
    },
}

/// A declared composite type: its shape, and whether its declaration carries
/// explicit layout decorations.
#[derive(Clone)]
pub(super) struct DeclaredComposite {
    pub shape: CompositeShape,
    pub explicit_layout: bool,
}

/// A constant the writer declared whose type may have an implicit-layout
/// variant.
#[derive(Clone)]
pub(super) enum DeclaredConstant {
    Composite {
        type_id: Word,
        constituents: Vec<Word>,
    },
    Null {
        type_id: Word,
    },
}

impl Writer {
    /// Record in [`Writer::explicit_layout_types`] every type reachable from a
    /// global in a storage class that takes explicit layout.
    ///
    /// `TaskPayloadWorkgroupEXT` takes explicit layout too, and its types keep
    /// their decorations.
    pub(super) fn collect_explicit_layout_types(&mut self, module: &crate::Module) {
        let mut pending = Vec::new();
        for (_, var) in module.global_variables.iter() {
            match var.space {
                crate::AddressSpace::Uniform
                | crate::AddressSpace::Storage { .. }
                | crate::AddressSpace::Immediate
                | crate::AddressSpace::TaskPayload => pending.push(var.ty),
                _ => {}
            }
        }
        while let Some(ty) = pending.pop() {
            if !self.explicit_layout_types.insert(ty) {
                continue;
            }
            match module.types[ty].inner {
                crate::TypeInner::Struct { ref members, .. } => {
                    pending.extend(members.iter().map(|member| member.ty));
                }
                crate::TypeInner::Array { base, .. }
                | crate::TypeInner::BindingArray { base, .. } => pending.push(base),
                _ => {}
            }
        }
    }

    /// The type a variable in `class` holds for values of `type_id`.
    pub(super) fn variable_type_id(&mut self, type_id: Word, class: spirv::StorageClass) -> Word {
        if refuses_explicit_layout(class, self.physical_layout.version) {
            self.implicit_layout_type_id(type_id)
        } else {
            type_id
        }
    }

    /// The implicit-layout variant of `type_id`: `type_id` itself when neither
    /// it nor anything it holds carries explicit layout, and otherwise the same
    /// composite declared without layout decorations, declared on first use.
    pub(super) fn implicit_layout_type_id(&mut self, type_id: Word) -> Word {
        if let Some(&variant) = self.implicit_layout_variants.get(&type_id) {
            return variant;
        }
        let Some(declared) = self.declared_composites.get(&type_id).cloned() else {
            return type_id;
        };
        let variant = match declared.shape {
            CompositeShape::Struct {
                members,
                member_names,
                name,
            } => {
                let variant_members: Vec<Word> = members
                    .iter()
                    .map(|&member| self.implicit_layout_type_id(member))
                    .collect();
                if !declared.explicit_layout && variant_members == members {
                    type_id
                } else {
                    let id = self.id_gen.next();
                    Instruction::type_struct(id, &variant_members)
                        .to_words(&mut self.logical_layout.declarations);
                    if self.flags.contains(super::WriterFlags::DEBUG) {
                        if let Some(ref name) = name {
                            self.debugs.push(Instruction::name(id, name));
                        }
                        for (index, member_name) in member_names.iter().enumerate() {
                            if let Some(ref member_name) = *member_name {
                                self.debugs.push(Instruction::member_name(
                                    id,
                                    index as u32,
                                    member_name,
                                ));
                            }
                        }
                    }
                    self.declared_composites.insert(
                        id,
                        DeclaredComposite {
                            shape: CompositeShape::Struct {
                                members: variant_members,
                                member_names,
                                name,
                            },
                            explicit_layout: false,
                        },
                    );
                    id
                }
            }
            CompositeShape::Array {
                element,
                length_id,
                length,
            } => {
                let variant_element = self.implicit_layout_type_id(element);
                if !declared.explicit_layout && variant_element == element {
                    type_id
                } else {
                    let id = self.id_gen.next();
                    Instruction::type_array(id, variant_element, length_id)
                        .to_words(&mut self.logical_layout.declarations);
                    self.declared_composites.insert(
                        id,
                        DeclaredComposite {
                            shape: CompositeShape::Array {
                                element: variant_element,
                                length_id,
                                length,
                            },
                            explicit_layout: false,
                        },
                    );
                    id
                }
            }
        };
        self.implicit_layout_variants.insert(type_id, variant);
        variant
    }

    /// The constant of `constant_id`'s value whose type is the implicit-layout
    /// variant of `constant_id`'s type, declared on first use.
    ///
    /// Returns `None` when `constant_id` is not a composite or null constant
    /// the writer declared.
    pub(super) fn implicit_layout_constant(&mut self, constant_id: Word) -> Option<Word> {
        if let Some(&variant) = self.implicit_layout_constants.get(&constant_id) {
            return Some(variant);
        }
        let variant = match self.declared_constants.get(&constant_id).cloned() {
            Some(DeclaredConstant::Null { type_id }) => {
                let variant_type_id = self.implicit_layout_type_id(type_id);
                if variant_type_id == type_id {
                    constant_id
                } else {
                    self.get_constant_null(variant_type_id)
                }
            }
            Some(DeclaredConstant::Composite {
                type_id,
                constituents,
            }) => {
                let variant_type_id = self.implicit_layout_type_id(type_id);
                if variant_type_id == type_id {
                    constant_id
                } else {
                    let mut variant_constituents = Vec::with_capacity(constituents.len());
                    for constituent in constituents {
                        // A scalar, vector or matrix constituent has no
                        // variant and stands as it is.
                        variant_constituents.push(
                            self.implicit_layout_constant(constituent)
                                .unwrap_or(constituent),
                        );
                    }
                    let id = self.id_gen.next();
                    Instruction::constant_composite(variant_type_id, id, &variant_constituents)
                        .to_words(&mut self.logical_layout.declarations);
                    self.declared_constants.insert(
                        id,
                        DeclaredConstant::Composite {
                            type_id: variant_type_id,
                            constituents: variant_constituents,
                        },
                    );
                    id
                }
            }
            None => return None,
        };
        self.implicit_layout_constants.insert(constant_id, variant);
        Some(variant)
    }

    /// Convert `value_id`, a value of `from_type_id`, to `to_type_id`, where
    /// one of the two is the other's implicit-layout variant. Returns the
    /// converted value, or `value_id` when the two types are the same.
    pub(super) fn write_layout_conversion(
        &mut self,
        body: &mut Vec<Instruction>,
        value_id: Word,
        from_type_id: Word,
        to_type_id: Word,
    ) -> Word {
        if from_type_id == to_type_id {
            return value_id;
        }
        let id = self.id_gen.next();
        if self.physical_layout.version >= 0x10400 {
            body.push(Instruction::copy_logical(to_type_id, id, value_id));
            return id;
        }
        let from = self.declared_composites[&from_type_id].shape.clone();
        let to = self.declared_composites[&to_type_id].shape.clone();
        let parts: Vec<(Word, Word)> = match (from, to) {
            (
                CompositeShape::Struct { members: from, .. },
                CompositeShape::Struct { members: to, .. },
            ) => from.into_iter().zip(to).collect(),
            (
                CompositeShape::Array {
                    element: from,
                    length,
                    ..
                },
                CompositeShape::Array { element: to, .. },
            ) => (0..length).map(|_| (from, to)).collect(),
            _ => unreachable!("a type and its implicit-layout variant have the same shape"),
        };
        let mut constituents = Vec::with_capacity(parts.len());
        for (index, (from_part, to_part)) in parts.into_iter().enumerate() {
            let part_id = self.id_gen.next();
            body.push(Instruction::composite_extract(
                from_part,
                part_id,
                value_id,
                &[index as Word],
            ));
            constituents.push(self.write_layout_conversion(body, part_id, from_part, to_part));
        }
        body.push(Instruction::composite_construct(
            to_type_id,
            id,
            &constituents,
        ));
        id
    }

    /// Load the value of `value_type_id` that `pointer_id`, a pointer in
    /// `class`, points to, and return its id.
    pub(super) fn write_value_load(
        &mut self,
        body: &mut Vec<Instruction>,
        value_type_id: Word,
        pointer_id: Word,
        class: spirv::StorageClass,
    ) -> Word {
        let held_type_id = self.variable_type_id(value_type_id, class);
        let id = self.id_gen.next();
        body.push(Instruction::load(held_type_id, id, pointer_id, None));
        self.write_layout_conversion(body, id, held_type_id, value_type_id)
    }

    /// Store `value_id`, a value of `value_type_id`, through `pointer_id`, a
    /// pointer in `class`.
    pub(super) fn write_value_store(
        &mut self,
        body: &mut Vec<Instruction>,
        value_type_id: Word,
        value_id: Word,
        pointer_id: Word,
        class: spirv::StorageClass,
    ) {
        let held_type_id = self.variable_type_id(value_type_id, class);
        let held_id = self.write_layout_conversion(body, value_id, value_type_id, held_type_id);
        body.push(Instruction::store(pointer_id, held_id, None));
    }
}
