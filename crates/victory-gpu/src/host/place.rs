//! The title's buffer copies, done on the card at their place in the list
//! (`VICTORY_COPIES_IN_PLACE`, on by default; `0`: all at submission, on the
//! CPU, as before).
//!
//! A buffer copy was done in title memory when its list was submitted, before
//! any draw of the submission ran, and the host copies refreshed after it.
//! That is another order than the list's: draws ahead of the copy read what
//! it put; of two copies to the same bytes with a draw behind each, both
//! draws read the second; a source that a dispatch of the same submission
//! writes first was read as it was before it; and a destination the
//! submissions in the air still read changed under them (where its host copy
//! has one side only, `crate::shadow::card_all`).
//!
//! Now the list copies, in the destination's host copy: out of the source's
//! host copy where shaders write that one, else out of upload memory of the
//! list that takes the source's bytes when the list is submitted (a source
//! shaders never wrote is the title's CPU's, or a copy's put there earlier in
//! the submission). What is written there is a shader's write from then on:
//! the bytes are declared, and what the card made of them is put into title
//! memory before the title is told of the submission.
//!
//! Title memory takes the copy at submission as well, as it did: fills of
//! upload memory, textures and host copies that lie over the same bytes read
//! it there. The reference of the destination's host copy takes the same
//! bytes ([`Shadow::placed_copy`]), so what the landing finds is what the card
//! made of them afterwards (a counter copied to zero and counted up; a source
//! that a shader wrote in between), and nothing is sent up over it. Where
//! flights in the air may write the source or the destination this is left to
//! the landing: what they wrote is not in title memory yet, and what they
//! will put there must not stand over this copy's bytes. If a step of the
//! submission reads the destination on the CPU, it waits for them instead
//! ([`Placed::waits`]).
//!
//! Copies to memory that has no host copy yet are done as before: nothing on
//! the card has read or written it. So are those of more than 8 MiB out of
//! memory shaders do not write (streaming).

use super::*;
use crate::shadow::Shadow;

pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COPIES_IN_PLACE").is_none_or(|v| v != "0"))
}

/// The most bytes copied out of memory shaders do not write.
const MOST: usize = 8 << 20;

/// Since the last frame-rate line: copies at their place, their bytes, those
/// out of a host copy, and texture copies out of one.
pub(super) static STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// What the CPU does at submission for a copy at its place in a list.
pub(super) struct Placed {
    pub(super) dst: usize,
    pub(super) src: usize,
    pub(super) len: usize,
    /// The destination's host copy.
    pub(super) shadow: std::sync::Arc<Shadow>,
    /// Upload memory the list copies from, to be given the source's bytes.
    from: Option<usize>,
}

impl Placed {
    /// Whether flights that write `written` are waited for.
    pub(super) fn waits(&self, written: &[(usize, usize)]) -> Waits {
        let source = touches(written, self.src, self.src + self.len);
        if source && self.from.is_some() {
            return Waits::Yes;
        }
        match source || touches(written, self.dst, self.dst + self.len) {
            true => Waits::ForReaders,
            false => Waits::No,
        }
    }

    /// At submission, in the order of the steps: the list's upload memory
    /// takes the source's bytes and, if `free` (no flight in the air may write
    /// source or destination), title memory at the destination does. Gives
    /// the bytes if it took them.
    pub(super) fn prepare(&self, free: bool) -> Option<Vec<u8>> {
        if !free && self.from.is_none() {
            return None;
        }
        let mut bytes = vec![0u8; self.len];
        // (What is not there reads as zeros, as any copy's source.)
        copy_mapped(bytes.as_mut_ptr(), self.src as *const u8, self.len, false);
        if let Some(to) = self.from {
            // SAFETY: the list's own upload memory of that many bytes.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), to as *mut u8, self.len) };
        }
        if !free {
            return None;
        }
        copy_mapped(self.dst as *mut u8, bytes.as_ptr(), self.len, true);
        crate::watch::wrote(self.dst, self.len);
        // Title memory that is not all there was left alone.
        let mut there = vec![0u8; self.len];
        (read_into(self.dst as *const u8, &mut there) && there == bytes).then_some(bytes)
    }
}

/// Whether a copy waits for flights before its submission is prepared: not;
/// yes, as it takes a source on the CPU that they may write; or only if
/// another step of the submission reads its destination on the CPU.
pub(super) enum Waits {
    No,
    ForReaders,
    Yes,
}

/// Incremental union of the destinations that a later CPU reader may
/// have to wait for. The small path stays contiguous; large, fragmented
/// sets use a tree instead of sorting and rebuilding the entire union
/// for every placed copy. Intervals are nonempty, disjoint and nonadjacent.
enum ReaderRanges {
    Small(Vec<(usize, usize)>),
    Large(std::collections::BTreeMap<usize, usize>),
}

impl ReaderRanges {
    const SMALL_LIMIT: usize = 32;

    fn new() -> Self {
        Self::Small(Vec::new())
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Small(ranges) => ranges.is_empty(),
            Self::Large(ranges) => ranges.is_empty(),
        }
    }

    fn touches(&self, from: usize, to: usize) -> bool {
        if from >= to {
            return false;
        }
        match self {
            Self::Small(ranges) => touches(ranges, from, to),
            Self::Large(ranges) => ranges.range(..to).next_back()
                .is_some_and(|(_, &end)| end > from),
        }
    }

    fn insert(&mut self, mut from: usize, mut to: usize) {
        if from >= to {
            return;
        }
        match self {
            Self::Small(ranges) => {
                let first = ranges.partition_point(|r| r.1 < from);
                let mut last = first;
                while let Some(&(a, b)) = ranges.get(last) {
                    if a > to {
                        break;
                    }
                    from = from.min(a);
                    to = to.max(b);
                    last += 1;
                }
                if first == last {
                    ranges.insert(first, (from, to));
                } else {
                    ranges[first] = (from, to);
                    ranges.drain(first + 1..last);
                }
                if ranges.len() > Self::SMALL_LIMIT {
                    *self = Self::Large(ranges.iter().copied().collect());
                }
            }
            Self::Large(ranges) => {
                if let Some((&a, &b)) = ranges.range(..=from).next_back() {
                    if b >= from {
                        if b >= to {
                            return;
                        }
                        from = a;
                        ranges.remove(&a);
                    }
                }
                while let Some((&a, &b)) = ranges.range(from..).next() {
                    if a > to {
                        break;
                    }
                    to = to.max(b);
                    ranges.remove(&a);
                }
                ranges.insert(from, to);
            }
        }
    }
}

/// Whether a step of `runs` reads on the CPU what one of their copies at its
/// place, ahead of it, waits for its readers ([`Waits::ForReaders`]) with. (A
/// reader ahead of the copy wants what was there before it, and the steps are
/// done in this order: `VICTORY_PLACED_READERS=0` counts those as well, as
/// before.)
pub(super) fn read_by(runs: &[&HostList], written: &[(usize, usize)]) -> bool {
    static ALL: OnceLock<bool> = OnceLock::new();
    let all = *ALL.get_or_init(|| std::env::var_os("VICTORY_PLACED_READERS").is_some_and(|v| v == "0"));
    let mut ranges = ReaderRanges::new();
    let waits = |p: &Placed| matches!(p.waits(written), Waits::ForReaders);
    if all {
        for list in runs {
            let steps = &list.deferred().copies;
            let placed = steps.iter().filter_map(|s| if let Step::Placed(p) = s { Some(p) } else { None });
            for p in placed.filter(|p| waits(p)) {
                ranges.insert(p.dst, p.dst + p.len);
            }
        }
    }
    runs.iter().any(|list| {
        list.deferred().copies.iter().any(|step| {
            // What reads, by kind ([`READERS`]).
            let (read, kind) = match step {
                Step::Placed(p) => {
                    let read = p.from.is_some() && ranges.touches(p.src, p.src + p.len);
                    if !all && waits(p) {
                        ranges.insert(p.dst, p.dst + p.len);
                    }
                    (read, 0)
                }
                _ if ranges.is_empty() => (false, 0),
                &Step::Copy(_, src, len) => (ranges.touches(src, src + len), 1),
                &Step::Fill(_, src, len, _) => (ranges.touches(src, src + len), 2),
                Step::Texture(staging) => {
                    let (at, len) = staging.memory();
                    (ranges.touches(at, at + len), 3)
                }
                Step::Host(_, (from, to)) => (ranges.touches(*from, *to), 4),
                _ => (false, 0),
            };
            #[cfg(feature = "diagnostics")]
            if read {
                READERS[kind].fetch_add(1, Ordering::Relaxed);
            }
            let _ = kind;
            read
        })
    })
}

/// Diagnostics: the steps found reading what a copy at its place writes
/// ([`read_by`]) since the last frame line: another such copy, a buffer copy,
/// a fill, a texture, a step of ours.
#[cfg(feature = "diagnostics")]
pub(super) static READERS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Everything recorded so far is done, and seen, before what follows.
///
/// # Safety
/// An open list.
unsafe fn wait_for_all(host: &Host, list: &ID3D12GraphicsCommandList) {
    static ENHANCED: OnceLock<bool> = OnceLock::new();
    let enhanced = *ENHANCED.get_or_init(|| {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
        // SAFETY: a feature structure of its size.
        let known = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS12,
                (&raw mut o).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>() as u32,
            )
        };
        known.is_ok() && o.EnhancedBarriersSupported.as_bool()
    });
    // SAFETY: the caller's.
    unsafe {
        if enhanced && let Ok(list) = list.cast::<ID3D12GraphicsCommandList7>() {
            let all = [D3D12_GLOBAL_BARRIER {
                SyncBefore: D3D12_BARRIER_SYNC_ALL,
                SyncAfter: D3D12_BARRIER_SYNC_ALL,
                AccessBefore: D3D12_BARRIER_ACCESS_COMMON,
                AccessAfter: D3D12_BARRIER_ACCESS_COMMON,
            }];
            list.Barrier(&[D3D12_BARRIER_GROUP {
                Type: D3D12_BARRIER_TYPE_GLOBAL,
                NumBarriers: 1,
                Anonymous: D3D12_BARRIER_GROUP_0 {
                    pGlobalBarriers: all.as_ptr(),
                },
            }]);
            return;
        }
        let alias = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                    pResourceBefore: std::mem::ManuallyDrop::new(None),
                    pResourceAfter: std::mem::ManuallyDrop::new(None),
                }),
            },
        };
        let uav = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        };
        list.ResourceBarrier(&[alias, uav]);
    }
}

/// The host copy of `[src, src + len)` if shaders write it: what it holds at
/// a place in a list is not what title memory holds at submission.
fn shaders_own(src: usize, len: usize) -> Option<std::sync::Arc<Shadow>> {
    crate::shadow::peek(src as u64, len as u64).filter(|s| s.written() && !s.in_place())
}

impl HostList {
    /// Copies `len` bytes of title memory at `src` to `dst`, at this place of
    /// the list (see the file). `false` if it is to be done at submission.
    pub(crate) fn place_copy(&self, dst: usize, src: usize, len: usize) -> bool {
        if !on() {
            return false;
        }
        let Some(host) = host() else {
            return false;
        };
        let Some(to) = crate::shadow::peek(dst as u64, len as u64).filter(|s| !s.in_place()) else {
            return false;
        };
        let own = shaders_own(src, len);
        let scratch = match &own {
            Some(_) => None,
            None if len <= MOST => match self.scratch_at(len as u64) {
                Some(scratch) => Some(scratch),
                None => return false,
            },
            None => return false,
        };
        let at = to.offset(dst as u64);
        // What the title's CPU wrote around and under it so far goes first.
        to.refresh(self, dst as u64, len as u64);
        // SAFETY (both): an open list; live buffers (the list holds the
        // copies and its scratch memory), ranges inside them.
        unsafe { wait_for_all(host, &self.list) };
        match (&own, &scratch) {
            (Some(from), _) => {
                from.refresh(self, src as u64, len as u64);
                let offset = from.offset(src as u64);
                if from.buffer.as_raw() == to.buffer.as_raw() {
                    let Some(between) = self.between(host, len as u64) else {
                        return false;
                    };
                    unsafe {
                        self.list.CopyBufferRegion(&between, 0, &from.buffer, offset, len as u64);
                        self.list.ResourceBarrier(&[transition(&between, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                        wait_for_all(host, &self.list);
                        self.list.CopyBufferRegion(&to.buffer, at, &between, 0, len as u64);
                    }
                } else {
                    unsafe { self.list.CopyBufferRegion(&to.buffer, at, &from.buffer, offset, len as u64) };
                }
                STATS[2].fetch_add(1, Ordering::Relaxed);
            }
            (None, Some((from, offset, _))) => unsafe {
                self.list.CopyBufferRegion(&to.buffer, at, from, *offset, len as u64);
            },
            (None, None) => return false,
        }
        unsafe { wait_for_all(host, &self.list) };
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[1].fetch_add(len as u64, Ordering::Relaxed);
        to.gpu_writes(self, dst as u64, len as u64);
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Placed(Placed {
            dst,
            src,
            len,
            shadow: to,
            from: scratch.map(|s| s.2 as usize),
        }));
        deferred.title_written();
        self.changed_something();
        true
    }

    /// For a copy of `[src, src + len)` into a texture: a buffer (readable by
    /// copies) that holds what shaders wrote there, at this place of the list.
    /// `None` where shaders do not write it.
    pub(crate) fn place_source(&self, src: usize, len: usize) -> Option<ID3D12Resource> {
        if !on() {
            return None;
        }
        let host = host()?;
        let from = shaders_own(src, len)?;
        let between = self.between(host, len as u64)?;
        from.refresh(self, src as u64, len as u64);
        STATS[3].fetch_add(1, Ordering::Relaxed);
        // SAFETY: an open list; live buffers, ranges inside them.
        unsafe {
            wait_for_all(host, &self.list);
            self.list.CopyBufferRegion(&between, 0, &from.buffer, from.offset(src as u64), len as u64);
            self.list.ResourceBarrier(&[transition(&between, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            wait_for_all(host, &self.list);
        }
        Some(between)
    }

    /// A buffer on the card for this recording alone.
    fn between(&self, host: &Host, len: u64) -> Option<ID3D12Resource> {
        let buffer = host.uav_buffer(len.next_multiple_of(1 << 16))?;
        name(&buffer, || "bytes between copies".into());
        self.keep(&buffer);
        Some(buffer)
    }
}

#[cfg(test)]
mod tests {

    use super::{ReaderRanges, join_ranges, touches};

    fn check_union(index: &ReaderRanges, reference: &[(usize, usize)]) {
        let actual: Vec<_> = match index {
            ReaderRanges::Small(ranges) => ranges.clone(),
            ReaderRanges::Large(ranges) => ranges.iter().map(|(&a, &b)| (a, b)).collect(),
        };
        assert_eq!(actual, reference);
        assert_eq!(index.is_empty(), reference.is_empty());
    }

    #[test]
    fn placed_reader_index_matches_union() {
        let mut seed = 0x18a5_390du32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as usize
        };
        for fragmented in [false, true] {
            for _ in 0..32 {
                let mut reference = Vec::new();
                let mut index = ReaderRanges::new();
                for i in 0..512 {
                    let from = if fragmented { (next() % 4096) * 16 } else { next() % 8192 };
                    let to = from + next() % if fragmented { 12 } else { 256 };
                    // A placed step reads before adding its own destination.
                    let q = next() % 65536;
                    let end = q + next() % 128;
                    assert_eq!(index.touches(q, end), touches(&reference, q, end));
                    index.insert(from, to);
                    join_ranges(&mut reference, [(from, to)]);
                    if i % 13 == 0 { check_union(&index, &reference); }
                    assert_eq!(index.touches(from, to), touches(&reference, from, to));
                }
                check_union(&index, &reference);
            }
        }
        let mut index = ReaderRanges::new();
        let mut reference = Vec::new();
        for i in 0..64 {
            index.insert(i * 16, i * 16 + 8);
            join_ranges(&mut reference, [(i * 16, i * 16 + 8)]);
        }
        assert!(matches!(&index, ReaderRanges::Large(_)));
        for (a, b) in [(8, 16), (0, 1024), (5, 7), (7, 7), (8, 1),
                       (usize::MAX - 8, usize::MAX)] {
            index.insert(a, b);
            join_ranges(&mut reference, [(a, b)]);
            check_union(&index, &reference);
            for (q, e) in [(0, 0), (0, 1), (1024, 1025), (usize::MAX - 1, usize::MAX)] {
                assert_eq!(index.touches(q, e), touches(&reference, q, e));
            }
        }
    }

    /// CPU-only benchmark; no graphics device or game workload required.
    #[test]
    #[ignore]
    fn benchmark_placed_reader_index() {
        use std::hint::black_box;
        use std::time::Instant;
        for count in [16usize, 128, 1024, 2800] {
            for fragmented in [false, true] {
                let input: Vec<_> = (0..count).rev().map(|i| {
                    let from = i * 32;
                    (from, from + if fragmented { 8 } else { 48 })
                }).collect();
                let old = || {
                    let mut ranges = Vec::new();
                    let mut hits = 0;
                    for &(a, b) in black_box(&input) {
                        hits += usize::from(touches(&ranges, a + 8, a + 32));
                        join_ranges(&mut ranges, [(a, b)]);
                    }
                    (hits, ranges)
                };
                let new = || {
                    let mut ranges = ReaderRanges::new();
                    let mut hits = 0;
                    for &(a, b) in black_box(&input) {
                        hits += usize::from(ranges.touches(a + 8, a + 32));
                        ranges.insert(a, b);
                    }
                    (hits, ranges)
                };
                let (hits, reference) = old();
                let (actual, index) = new();
                assert_eq!(hits, actual);
                check_union(&index, &reference);
                black_box(old());
                black_box(new());
                let measure = |indexed: bool| {
                    let start = Instant::now();
                    for _ in 0..8 {
                        if indexed { drop(black_box(new())); }
                        else { drop(black_box(old())); }
                    }
                    start.elapsed().as_secs_f64() * 1e6 / 8.0
                };
                let (mut a, mut b) = (Vec::new(), Vec::new());
                for i in 0..7 {
                    let (x, y) = if i % 2 == 0 {
                        (measure(false), measure(true))
                    } else {
                        let y = measure(true);
                        (measure(false), y)
                    };
                    a.push(x); b.push(y);
                }
                a.sort_by(f64::total_cmp);
                b.sort_by(f64::total_cmp);
                println!("kind=placed_readers_{} blocks={count} reference_us={:.3} worklist_us={:.3} speedup={:.3}",
                    if fragmented {"fragmented"} else {"overlapping"}, a[3], b[3], a[3] / b[3]);
            }
        }
    }
    /// What copies with two sides cost to make (ignored: a measurement for
    /// `VICTORY_CARD_ALL`): buffers of 64 KiB on the card and in the PC's
    /// memory.
    #[test]
    #[ignore]
    fn small_buffers_cost() {
        let Some(host) = super::host() else {
            return;
        };
        const N: usize = 1000;
        let used = || host.video_memory().map_or(0, |m| m[1] >> 20);
        let before = used();
        let t = std::time::Instant::now();
        let card: Vec<_> = (0..N).map(|_| host.uav_buffer(1 << 16)).collect();
        let card_us = t.elapsed().as_micros() as f64 / N as f64;
        let after = used();
        let t = std::time::Instant::now();
        let pc: Vec<_> = (0..N).map(|_| host.shared_buffer(1 << 16)).collect();
        let pc_us = t.elapsed().as_micros() as f64 / N as f64;
        println!(
            "{N} buffers of 64 KiB: on the card {card_us:.0} us each, {} MB of its memory; in the PC's memory {pc_us:.0} us each",
            after.saturating_sub(before)
        );
        drop((card, pc));
    }
}
