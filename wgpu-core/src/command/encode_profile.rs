//! The CPU time [`CommandEncoder::finish`](super::CommandEncoder) spends
//! encoding the commands an encoder recorded, by the kind of work.
//!
//! Passes are recorded as commands and encoded into the backend when the
//! encoder is finished, so `finish` carries the validation, resource
//! tracking, barrier generation and backend calls of every pass the encoder
//! holds. Each finish adds what it spent to the calling thread's account;
//! [`take_encode_profile`] reads the account and empties it.
//!
//! The account is kept only while [`set_encode_profiling`] has turned it on
//! for the thread, so a thread that never asks pays for no clock reads.

#[cfg(feature = "std")]
use core::cell::{Cell, RefCell};

/// One kind of encoding work the account separates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum EncodeStage {
    /// The whole of one `finish`'s encoding.
    Finish,
    /// A render pass, from its start to its prelude closed.
    RenderPass,
    /// A render pass's start: attachments, load ops, timestamp writes.
    RenderPassStart,
    /// The bind groups a render pass sets.
    RenderSetBindGroup,
    /// The pipelines a render pass sets.
    RenderSetPipeline,
    /// The draws of a render pass, indirect ones included.
    RenderDraw,
    /// Every other command of a render pass.
    RenderOther,
    /// A render pass's end: its prelude's barriers and discard fixups.
    RenderPassEnd,
    /// A compute pass, from its start to its prelude closed.
    ComputePass,
    /// A compute pass's start: the backend pass, trackers, timestamp writes.
    ComputePassStart,
    /// The bind groups a compute pass sets.
    ComputeSetBindGroup,
    /// The pipelines a compute pass sets.
    ComputeSetPipeline,
    /// The dispatches of a compute pass, with the barriers each one settles.
    ComputeDispatch,
    /// Within a compute dispatch: merging the bound groups into the
    /// dispatch's usage scope and moving it into the pass's trackers.
    ComputeDispatchScope,
    /// Within a compute dispatch: the bind groups' init actions and the
    /// backend's bind group calls.
    ComputeDispatchBind,
    /// Within a compute dispatch: the barriers drained from the trackers and
    /// handed to the backend.
    ComputeDispatchBarriers,
    /// Within a compute dispatch: the backend's dispatch call.
    ComputeDispatchRecord,
    /// Every other command of a compute pass.
    ComputeOther,
    /// A compute pass's end: its prelude's barriers and discard fixups.
    ComputePassEnd,
    /// Copies between buffers and textures.
    Copy,
    /// Buffer and texture clears.
    Clear,
    /// Acceleration structure builds.
    AccelerationStructureBuild,
    /// Query set resolves and timestamp writes outside passes.
    Query,
    /// Every other command outside a pass.
    Other,
    /// Closing the encoder after its last command.
    Close,
}

impl EncodeStage {
    /// Every stage, in the order the account lists them.
    pub const ALL: [EncodeStage; STAGES] = [
        EncodeStage::Finish,
        EncodeStage::RenderPass,
        EncodeStage::RenderPassStart,
        EncodeStage::RenderSetBindGroup,
        EncodeStage::RenderSetPipeline,
        EncodeStage::RenderDraw,
        EncodeStage::RenderOther,
        EncodeStage::RenderPassEnd,
        EncodeStage::ComputePass,
        EncodeStage::ComputePassStart,
        EncodeStage::ComputeSetBindGroup,
        EncodeStage::ComputeSetPipeline,
        EncodeStage::ComputeDispatch,
        EncodeStage::ComputeDispatchScope,
        EncodeStage::ComputeDispatchBind,
        EncodeStage::ComputeDispatchBarriers,
        EncodeStage::ComputeDispatchRecord,
        EncodeStage::ComputeOther,
        EncodeStage::ComputePassEnd,
        EncodeStage::Copy,
        EncodeStage::Clear,
        EncodeStage::AccelerationStructureBuild,
        EncodeStage::Query,
        EncodeStage::Other,
        EncodeStage::Close,
    ];

    /// The stage's name, in `snake_case`.
    pub const fn name(self) -> &'static str {
        match self {
            EncodeStage::Finish => "finish",
            EncodeStage::RenderPass => "render_pass",
            EncodeStage::RenderPassStart => "render_pass_start",
            EncodeStage::RenderSetBindGroup => "render_set_bind_group",
            EncodeStage::RenderSetPipeline => "render_set_pipeline",
            EncodeStage::RenderDraw => "render_draw",
            EncodeStage::RenderOther => "render_other",
            EncodeStage::RenderPassEnd => "render_pass_end",
            EncodeStage::ComputePass => "compute_pass",
            EncodeStage::ComputePassStart => "compute_pass_start",
            EncodeStage::ComputeSetBindGroup => "compute_set_bind_group",
            EncodeStage::ComputeSetPipeline => "compute_set_pipeline",
            EncodeStage::ComputeDispatch => "compute_dispatch",
            EncodeStage::ComputeDispatchScope => "compute_dispatch_scope",
            EncodeStage::ComputeDispatchBind => "compute_dispatch_bind",
            EncodeStage::ComputeDispatchBarriers => "compute_dispatch_barriers",
            EncodeStage::ComputeDispatchRecord => "compute_dispatch_record",
            EncodeStage::ComputeOther => "compute_other",
            EncodeStage::ComputePassEnd => "compute_pass_end",
            EncodeStage::Copy => "copy",
            EncodeStage::Clear => "clear",
            EncodeStage::AccelerationStructureBuild => "acceleration_structure_build",
            EncodeStage::Query => "query",
            EncodeStage::Other => "other",
            EncodeStage::Close => "close",
        }
    }
}

const STAGES: usize = 25;

/// What the thread's finishes spent, stage by stage, since the account was
/// last taken.
#[derive(Clone, Copy, Debug, Default)]
pub struct EncodeProfile {
    counts: [u32; STAGES],
    nanos: [u64; STAGES],
    /// The backend command buffers the finishes handed to their command
    /// buffers.
    pub raw_command_buffers: u32,
}

impl EncodeProfile {
    /// How many times the stage ran.
    pub fn count(&self, stage: EncodeStage) -> u32 {
        self.counts[stage as usize]
    }

    /// The CPU time the stage took, in nanoseconds.
    pub fn nanos(&self, stage: EncodeStage) -> u64 {
        self.nanos[stage as usize]
    }
}

#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
mod clock {
    pub(crate) type Mark = Option<std::time::Instant>;

    pub(crate) fn now(on: bool) -> Mark {
        on.then(std::time::Instant::now)
    }

    pub(crate) fn since(mark: Mark) -> Option<u64> {
        mark.map(|m| m.elapsed().as_nanos() as u64)
    }
}

#[cfg(not(all(feature = "std", not(target_arch = "wasm32"))))]
mod clock {
    pub(crate) type Mark = Option<()>;

    pub(crate) fn now(_on: bool) -> Mark {
        None
    }

    pub(crate) fn since(_mark: Mark) -> Option<u64> {
        None
    }
}

#[cfg(feature = "std")]
std::thread_local! {
    static ON: Cell<bool> = const { Cell::new(false) };
    static ACCOUNT: RefCell<EncodeProfile> = const { RefCell::new(EncodeProfile {
        counts: [0; STAGES],
        nanos: [0; STAGES],
        raw_command_buffers: 0,
    }) };
}

/// Turns the calling thread's account on or off.
pub fn set_encode_profiling(on: bool) {
    #[cfg(feature = "std")]
    ON.with(|c| c.set(on));
    #[cfg(not(feature = "std"))]
    let _ = on;
}

/// Reads the calling thread's account and empties it.
pub fn take_encode_profile() -> EncodeProfile {
    #[cfg(feature = "std")]
    {
        ACCOUNT.with(|a| core::mem::take(&mut *a.borrow_mut()))
    }
    #[cfg(not(feature = "std"))]
    {
        EncodeProfile::default()
    }
}

/// A stage begun, timed while the thread's account is on.
#[derive(Clone, Copy)]
pub(crate) struct Mark(clock::Mark);

pub(crate) fn start() -> Mark {
    #[cfg(feature = "std")]
    let on = ON.with(Cell::get);
    #[cfg(not(feature = "std"))]
    let on = false;
    Mark(clock::now(on))
}

/// Adds the time since `mark` to `stage`.
pub(crate) fn record(stage: EncodeStage, mark: Mark) {
    if let Some(nanos) = clock::since(mark.0) {
        #[cfg(feature = "std")]
        ACCOUNT.with(|a| {
            let mut p = a.borrow_mut();
            p.counts[stage as usize] += 1;
            p.nanos[stage as usize] += nanos;
        });
        #[cfg(not(feature = "std"))]
        let _ = (stage, nanos);
    }
}

/// Adds `count` backend command buffers to the account.
pub(crate) fn record_raw_command_buffers(count: usize) {
    #[cfg(feature = "std")]
    if ON.with(Cell::get) {
        ACCOUNT.with(|a| a.borrow_mut().raw_command_buffers += count as u32);
    }
    #[cfg(not(feature = "std"))]
    let _ = count;
}
