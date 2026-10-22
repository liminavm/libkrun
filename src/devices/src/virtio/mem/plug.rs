//! Which blocks of the region the guest has plugged, and the answer to each request.

use super::defs::uapi::*;

/// A request from the guest, decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Request {
    Plug { addr: u64, nb_blocks: u16 },
    Unplug { addr: u64, nb_blocks: u16 },
    UnplugAll,
    State { addr: u64, nb_blocks: u16 },
    Unknown(u16),
}

impl Request {
    pub fn decode(raw: &[u8; REQ_SIZE]) -> Self {
        let kind = u16::from_le_bytes([raw[0], raw[1]]);
        let addr = u64::from_le_bytes(raw[8..16].try_into().unwrap());
        let nb_blocks = u16::from_le_bytes([raw[16], raw[17]]);
        match kind {
            VIRTIO_MEM_REQ_PLUG => Request::Plug { addr, nb_blocks },
            VIRTIO_MEM_REQ_UNPLUG => Request::Unplug { addr, nb_blocks },
            VIRTIO_MEM_REQ_UNPLUG_ALL => Request::UnplugAll,
            VIRTIO_MEM_REQ_STATE => Request::State { addr, nb_blocks },
            other => Request::Unknown(other),
        }
    }
}

/// The answer to a request, and the block ranges, as `(first block, count)`, it unplugged.
#[derive(Debug, PartialEq, Eq)]
pub struct Outcome {
    pub resp: u16,
    pub state: u16,
    pub unplugged: Vec<(usize, usize)>,
}

impl Outcome {
    fn resp(resp: u16) -> Self {
        Self {
            resp,
            state: 0,
            unplugged: Vec::new(),
        }
    }

    pub fn encode(&self) -> [u8; RESP_SIZE] {
        let mut raw = [0u8; RESP_SIZE];
        raw[0..2].copy_from_slice(&self.resp.to_le_bytes());
        raw[8..10].copy_from_slice(&self.state.to_le_bytes());
        raw
    }
}

pub struct Plugged {
    addr: u64,
    block_size: u64,
    blocks: Vec<bool>,
    plugged: usize,
}

impl Plugged {
    pub fn new(addr: u64, block_size: u64, nr_blocks: usize) -> Self {
        Self {
            addr,
            block_size,
            blocks: vec![false; nr_blocks],
            plugged: 0,
        }
    }

    pub fn plugged_size(&self) -> u64 {
        self.plugged as u64 * self.block_size
    }

    /// Answer `req`, given the size the host currently asks for. The whole region is usable.
    pub fn handle(&mut self, req: Request, requested_size: u64) -> Outcome {
        match req {
            Request::Plug { addr, nb_blocks } => {
                let Some(range) = self.range(addr, nb_blocks) else {
                    return Outcome::resp(VIRTIO_MEM_RESP_ERROR);
                };
                if self.blocks[range.clone()].iter().any(|&b| b) {
                    return Outcome::resp(VIRTIO_MEM_RESP_ERROR);
                }
                if self.plugged_size() + nb_blocks as u64 * self.block_size > requested_size {
                    return Outcome::resp(VIRTIO_MEM_RESP_NACK);
                }
                self.blocks[range].fill(true);
                self.plugged += nb_blocks as usize;
                Outcome::resp(VIRTIO_MEM_RESP_ACK)
            }
            Request::Unplug { addr, nb_blocks } => {
                let Some(range) = self.range(addr, nb_blocks) else {
                    return Outcome::resp(VIRTIO_MEM_RESP_ERROR);
                };
                if !self.blocks[range.clone()].iter().all(|&b| b) {
                    return Outcome::resp(VIRTIO_MEM_RESP_ERROR);
                }
                self.blocks[range.clone()].fill(false);
                self.plugged -= nb_blocks as usize;
                Outcome {
                    unplugged: vec![(range.start, range.len())],
                    ..Outcome::resp(VIRTIO_MEM_RESP_ACK)
                }
            }
            Request::UnplugAll => {
                let mut unplugged = Vec::new();
                let mut i = 0;
                while i < self.blocks.len() {
                    if !self.blocks[i] {
                        i += 1;
                        continue;
                    }
                    let start = i;
                    while i < self.blocks.len() && self.blocks[i] {
                        self.blocks[i] = false;
                        i += 1;
                    }
                    unplugged.push((start, i - start));
                }
                self.plugged = 0;
                Outcome {
                    unplugged,
                    ..Outcome::resp(VIRTIO_MEM_RESP_ACK)
                }
            }
            Request::State { addr, nb_blocks } => {
                let Some(range) = self.range(addr, nb_blocks) else {
                    return Outcome::resp(VIRTIO_MEM_RESP_ERROR);
                };
                let blocks = &self.blocks[range];
                let state = if blocks.iter().all(|&b| b) {
                    VIRTIO_MEM_STATE_PLUGGED
                } else if blocks.iter().all(|&b| !b) {
                    VIRTIO_MEM_STATE_UNPLUGGED
                } else {
                    VIRTIO_MEM_STATE_MIXED
                };
                Outcome {
                    state,
                    ..Outcome::resp(VIRTIO_MEM_RESP_ACK)
                }
            }
            Request::Unknown(_) => Outcome::resp(VIRTIO_MEM_RESP_ERROR),
        }
    }

    /// The blocks `[addr, addr + nb_blocks * block_size)` covers, if it is a non-empty, aligned
    /// range inside the region.
    fn range(&self, addr: u64, nb_blocks: u16) -> Option<std::ops::Range<usize>> {
        let offset = addr.checked_sub(self.addr)?;
        if nb_blocks == 0 || !offset.is_multiple_of(self.block_size) {
            return None;
        }
        let first = usize::try_from(offset / self.block_size).ok()?;
        let end = first.checked_add(nb_blocks as usize)?;
        (end <= self.blocks.len()).then_some(first..end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = 0x10_0000_0000;
    const BLOCK: u64 = 2 << 20;

    fn plug(addr: u64, nb_blocks: u16) -> Request {
        Request::Plug { addr, nb_blocks }
    }

    fn unplug(addr: u64, nb_blocks: u16) -> Request {
        Request::Unplug { addr, nb_blocks }
    }

    fn state(addr: u64, nb_blocks: u16) -> Request {
        Request::State { addr, nb_blocks }
    }

    #[test]
    fn requests_decode_from_the_wire_layout() {
        let mut raw = [0u8; REQ_SIZE];
        raw[0..2].copy_from_slice(&VIRTIO_MEM_REQ_UNPLUG.to_le_bytes());
        raw[8..16].copy_from_slice(&(BASE + BLOCK).to_le_bytes());
        raw[16..18].copy_from_slice(&7u16.to_le_bytes());
        assert_eq!(Request::decode(&raw), unplug(BASE + BLOCK, 7));
        let out = Outcome {
            state: VIRTIO_MEM_STATE_MIXED,
            ..Outcome::resp(VIRTIO_MEM_RESP_NACK)
        };
        assert_eq!(out.encode(), [1, 0, 0, 0, 0, 0, 0, 0, 2, 0]);
    }

    #[test]
    fn a_plug_beyond_the_requested_size_is_refused() {
        let mut p = Plugged::new(BASE, BLOCK, 8);
        assert_eq!(p.handle(plug(BASE, 2), 2 * BLOCK).resp, VIRTIO_MEM_RESP_ACK);
        assert_eq!(
            p.handle(plug(BASE + 2 * BLOCK, 1), 2 * BLOCK).resp,
            VIRTIO_MEM_RESP_NACK
        );
        assert_eq!(p.plugged_size(), 2 * BLOCK);
    }

    #[test]
    fn a_range_outside_the_region_or_misaligned_or_empty_is_an_error() {
        let mut p = Plugged::new(BASE, BLOCK, 8);
        let big = 8 * BLOCK;
        for req in [
            plug(BASE - BLOCK, 1),
            plug(BASE + 7 * BLOCK, 2),
            plug(BASE + BLOCK / 2, 1),
            plug(BASE, 0),
            state(BASE + 8 * BLOCK, 1),
            Request::Unknown(9),
        ] {
            assert_eq!(p.handle(req, big).resp, VIRTIO_MEM_RESP_ERROR, "{req:?}");
        }
        assert_eq!(p.plugged_size(), 0);
    }

    #[test]
    fn plugging_a_plugged_block_or_unplugging_an_unplugged_one_is_an_error() {
        let mut p = Plugged::new(BASE, BLOCK, 8);
        let big = 8 * BLOCK;
        p.handle(plug(BASE + BLOCK, 2), big);
        assert_eq!(p.handle(plug(BASE, 2), big).resp, VIRTIO_MEM_RESP_ERROR);
        assert_eq!(
            p.handle(unplug(BASE + 2 * BLOCK, 2), big).resp,
            VIRTIO_MEM_RESP_ERROR
        );
        assert_eq!(p.plugged_size(), 2 * BLOCK);
    }

    #[test]
    fn unplug_names_the_blocks_to_release_and_state_follows() {
        let mut p = Plugged::new(BASE, BLOCK, 8);
        let big = 8 * BLOCK;
        p.handle(plug(BASE, 4), big);
        assert_eq!(
            p.handle(state(BASE, 4), big).state,
            VIRTIO_MEM_STATE_PLUGGED
        );
        let out = p.handle(unplug(BASE + BLOCK, 2), big);
        assert_eq!(out.resp, VIRTIO_MEM_RESP_ACK);
        assert_eq!(out.unplugged, vec![(1, 2)]);
        assert_eq!(p.handle(state(BASE, 4), big).state, VIRTIO_MEM_STATE_MIXED);
        assert_eq!(
            p.handle(state(BASE + BLOCK, 2), big).state,
            VIRTIO_MEM_STATE_UNPLUGGED
        );
        assert_eq!(p.plugged_size(), 2 * BLOCK);
    }

    #[test]
    fn unplug_all_releases_every_plugged_run() {
        let mut p = Plugged::new(BASE, BLOCK, 8);
        let big = 8 * BLOCK;
        p.handle(plug(BASE, 2), big);
        p.handle(plug(BASE + 5 * BLOCK, 3), big);
        let out = p.handle(Request::UnplugAll, big);
        assert_eq!(out.unplugged, vec![(0, 2), (5, 3)]);
        assert_eq!(p.plugged_size(), 0);
        assert_eq!(
            p.handle(state(BASE, 8), big).state,
            VIRTIO_MEM_STATE_UNPLUGGED
        );
    }
}
