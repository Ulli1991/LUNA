"""Host-side (pure numpy) communication-plan builders for :mod:`amr.shard`.

Topology is static between regrids, so the sparse O(surface) exchanges
(same-level ghost fill, coarse-fine underlay, batched seam, interface reflux)
precompute their all_to_all routing here at (re)grid time. No jax: plans are
plain numpy arrays threaded into the jitted step as arguments.
"""

from __future__ import annotations


def build_exchange_plan(src_slots_np, n_src, n_dev):
    """Sparse exchange plan for a block gather ``dst_row <- src_slots[dst_row, :]``.
    Per ordered device pair (s -> r): the sorted unique REMOTE source blocks r's
    rows need from s; local sources are read directly (never on the wire), which
    makes the traffic O(surface) instead of the all_gather/ring's O(total).
    ``src_slots_np`` (n_rows, R) int, ``n_src`` = source-array size (may differ
    from n_rows, e.g. coarse-fine); both must divide by n_dev.

    Returns (send_idx, recv_pos, K):
      send_idx (n_dev, n_dev, K) int32 -- sender-LOCAL source indices; row [s, r]
          is what s packs for r (0-padded; padding never referenced).
      recv_pos (n_rows, R) int32 -- flat position (s*K + j) of each source in the
          receiver's all_to_all buffer, or -1 if LOCAL to the row's owner.
      K -- per-pair slot budget, pow2 bucket (bounds recompiles across regrids).
    """
    import numpy as _np
    n_rows, R = src_slots_np.shape
    bpd_row = n_rows // n_dev
    bpd_src = n_src // n_dev
    row_dev = _np.arange(n_rows) // bpd_row
    owner = src_slots_np // bpd_src
    uniq = {}                                        # (s, r) -> sorted unique globals
    kmax = 1
    for r in range(n_dev):
        st = src_slots_np[row_dev == r]              # (bpd_row, R)
        ow = owner[row_dev == r]
        for s in range(n_dev):
            if s == r:
                continue
            u = _np.unique(st[ow == s])
            uniq[(s, r)] = u
            kmax = max(kmax, len(u))
    K = 1 << (kmax - 1).bit_length()                 # pow2 bucket
    send_idx = _np.zeros((n_dev, n_dev, K), _np.int32)
    recv_pos = _np.full((n_rows, R), -1, _np.int32)
    for r in range(n_dev):
        rows = slice(r * bpd_row, (r + 1) * bpd_row)
        st = src_slots_np[rows]
        ow = owner[rows]
        for s in range(n_dev):
            if s == r:
                continue
            u = uniq[(s, r)]
            if len(u) == 0:
                continue
            send_idx[s, r, : len(u)] = u - s * bpd_src   # sender-local indices
            m = ow == s
            j = _np.searchsorted(u, st[m])
            block = recv_pos[rows]
            block[m] = s * K + j
            recv_pos[rows] = block
    return send_idx, recv_pos, K


def build_same_level_plan(stencil_np, n_dev):
    """Same-level ghost-fill plan: rows and sources are the SAME block array."""
    return build_exchange_plan(stencil_np, stencil_np.shape[0], n_dev)


def build_reflux_plan(fine_coords_np, fine_neighbors_np, fine_n_active,
                      fine_parent_slots_np, coarse_neighbors_np, n_coarse,
                      cfg, n_dev):
    """Routing plan for ``apply_reflux_iface`` pass B: for every ACTIVE fine C/F
    face, reproduce apply_reflux_blocks' target arithmetic (octant, parent's
    neighbour, cell layer, transverse quadrant) and route the restricted flux
    patch to the target's owner device.

    Returns (send_idx, meta, K):
      send_idx (n_dev, n_dev, K) int32 -- indices into the sender's flattened
          (bpd_f * 2ndim) payload axis (order [+d0, -d0, +d1, ...]); padding
          points at 0 (never read: its meta row is -1).
      meta (n_dev, n_dev*K, 4 + ndim) int32 -- per RECEIVER, aligned with its
          all_to_all buffer rows: [tgt_local_slot | d | sgn | cell_d | o_0..];
          tgt = -1 marks padding.
    """
    import numpy as _np
    ndim = cfg.ndim
    bs = cfg.block_shape
    hs = [s // 2 for s in bs]
    Nf = fine_coords_np.shape[0]
    bpd_f = Nf // n_dev
    bpd_c = n_coarse // n_dev
    f_sent = Nf - 1
    c_sent = n_coarse - 1
    na = int(fine_n_active)
    covered_np = _np.zeros(n_coarse, bool)
    act_par = fine_parent_slots_np[:na]
    covered_np[act_par[act_par >= 0]] = True

    # per (sender, receiver): lists of (payload_flat_idx, meta_row)
    buckets = {(s, r): [] for s in range(n_dev) for r in range(n_dev)}
    q = _np.maximum(fine_coords_np, 0) % 2
    for i in range(na):
        parent = int(fine_parent_slots_np[i])
        if parent < 0:
            continue
        sdev = i // bpd_f
        for d in range(ndim):
            for s2, col, pay_off in ((+1, 2 * d, 0), (-1, 2 * d + 1, 1)):
                if int(fine_neighbors_np[i, col]) != f_sent:
                    continue                              # not a C/F face
                qd = int(q[i, d])
                if s2 > 0:
                    tgt = parent if qd == 0 else int(
                        coarse_neighbors_np[parent, 2 * d])
                    cell = hs[d] if qd == 0 else 0
                else:
                    tgt = parent if qd == 1 else int(
                        coarse_neighbors_np[parent, 2 * d + 1])
                    cell = hs[d] - 1 if qd == 1 else bs[d] - 1
                if tgt < 0 or tgt == c_sent:
                    continue
                if covered_np[tgt]:
                    continue                  # restriction owns covered cells
                rdev = tgt // bpd_c
                offs = [int(q[i, t]) * hs[t] for t in range(ndim)]
                meta_row = [tgt - rdev * bpd_c, d, s2, cell] + offs
                pay_idx = (i - sdev * bpd_f) * (2 * ndim) + 2 * d + pay_off
                buckets[(sdev, rdev)].append((pay_idx, meta_row))

    K = max(1, max(len(v) for v in buckets.values()))
    K = 1 << (K - 1).bit_length()                        # pow2 bucket
    send_idx = _np.zeros((n_dev, n_dev, K), _np.int32)
    meta = _np.full((n_dev, n_dev * K, 4 + ndim), -1, _np.int32)
    for (sdev, rdev), items in buckets.items():
        for j, (pidx, mrow) in enumerate(items):
            send_idx[sdev, rdev, j] = pidx
            meta[rdev, sdev * K + j] = mrow
    return send_idx, meta, K
