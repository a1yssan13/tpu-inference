# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the lightning-indexer compressor orchestrator (Milestone 3).

``compressor_forward_indexer`` chains the projection GEMM,
``save_partial_states`` (Milestone 1) and
``compress_norm_rope_store_indexer`` (Milestone 2 head=128 path) over one
shared ``uint8`` buffer.

Like ``compressor_test``, these tests compare against a *naive NumPy
end-to-end reference* (``_naive_reference``): fp32 projection, a plain-Python
state scatter, then the trusted indexer boundary ground truth
(``compress_norm_rope_store_indexer_ref``, reused from
``compress_norm_rope_indexer_test``). We compare the dequantized indexer KV at
the slots the boundary store actually wrote.

    .venv/bin/python -m pytest \
        tests/kernels/experimental/deepseek_v4/compressor_indexer_test.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.experimental.deepseek_v4.compress_norm_rope import (
    shared_indexer_cache_shape, unpack_indexer_kv_cache)
from tpu_inference.kernels.experimental.deepseek_v4.compressor import (
    compressor_forward_indexer)
from tests.kernels.experimental.deepseek_v4.compress_norm_rope_indexer_test \
    import compress_norm_rope_store_indexer_ref

requires_tpu = pytest.mark.skipif(
    jax.devices()[0].platform != "tpu",
    reason="requires a TPU backend",
)

# KV row-slots per shared-cache page (the MLA storage block size).
PAGE_SIZE = 64


def _make_inputs(
    compress_ratio, overlap, head_dim=128, rope_head_dim=64, quant_block=128,
    hidden_size=256, num_reqs=2, seq_len=None, num_pad=0, seed=0,
):
    """Build a consistent batch: compressor slots follow the block table."""
    rng = np.random.default_rng(seed)
    coff = 1 + int(overlap)
    state_width = coff * head_dim
    state_dim = 2 * state_width
    state_block_size = 4 if compress_ratio == 4 else 8

    if seq_len is None:
        seq_len = 2 * compress_ratio
    num_tokens = num_reqs * seq_len

    max_blocks = (seq_len + state_block_size - 1) // state_block_size
    num_state_pages = num_reqs * max_blocks + 1  # +1 spare (page 0)

    # Compressed-KV output pages follow the state pages in the same buffer.
    num_kv_pages = (num_tokens // PAGE_SIZE) + 4
    num_pages = num_state_pages + num_kv_pages

    # Per-request contiguous physical state pages (page 0 left as spare).
    block_table = np.zeros((num_reqs, max_blocks), np.int32)
    nxt = 1
    for r in range(num_reqs):
        for b in range(max_blocks):
            block_table[r, b] = nxt
            nxt += 1

    positions = np.concatenate(
        [np.arange(seq_len, dtype=np.int32) for _ in range(num_reqs)])
    token_to_req_indices = np.repeat(
        np.arange(num_reqs, dtype=np.int32), seq_len)

    # Compressor (state) slot = physical slot of this token's logical position,
    # so the stage-2 window gather reads back exactly what stage 1 wrote.
    slot_mapping = np.empty(num_tokens, np.int32)
    for t in range(num_tokens):
        r = int(token_to_req_indices[t])
        p = int(positions[t])
        slot_mapping[t] = block_table[r, p // state_block_size] \
            * state_block_size + p % state_block_size

    # KV slots live in the KV page range so boundary writes never clobber the
    # state bytes the gather still needs.
    kv_base = num_state_pages * PAGE_SIZE
    kv_capacity = num_kv_pages * PAGE_SIZE
    kv_slot_mapping = (
        kv_base + rng.permutation(kv_capacity)[:num_tokens]).astype(np.int32)

    if num_pad > 0:
        pad_idx = rng.permutation(num_tokens)[:num_pad]
        slot_mapping[pad_idx] = -1
        kv_slot_mapping[pad_idx] = -1

    hidden_states = rng.standard_normal(
        (num_tokens, hidden_size), dtype=np.float32)
    wkv_wgate = (rng.standard_normal(
        (2 * state_width, hidden_size), dtype=np.float32) * 0.05)
    ape = rng.standard_normal((compress_ratio, state_width), dtype=np.float32)
    norm_weight = rng.standard_normal(head_dim, dtype=np.float32)

    max_pos = seq_len + compress_ratio
    cos_sin_cache = rng.standard_normal(
        (max_pos, rope_head_dim), dtype=np.float32)

    cache_shape = shared_indexer_cache_shape(
        num_pages, PAGE_SIZE, head_dim, quant_block)
    cache = np.zeros(cache_shape, dtype=np.uint8)

    return dict(
        hidden_states=hidden_states, wkv_wgate=wkv_wgate, ape=ape,
        norm_weight=norm_weight, cos_sin_cache=cos_sin_cache,
        positions=positions, slot_mapping=slot_mapping,
        block_table=block_table, token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping, cache=cache,
        state_block_size=state_block_size, head_dim=head_dim,
        rope_head_dim=rope_head_dim, compress_ratio=compress_ratio,
        overlap=overlap, rms_eps=1e-6, quant_block=quant_block)


def _to_jax(kw):
    return {
        k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
        for k, v in kw.items()
    }


def _save_state_ref(kv, score, kw):
    """Plain-NumPy ``save_partial_states``: build the f32 state view."""
    coff = 1 + int(kw["overlap"])
    state_dim = 2 * coff * kw["head_dim"]
    sb = kw["state_block_size"]
    num_pages = kw["cache"].shape[0]
    ape = kw["ape"].astype(np.float32)
    positions = kw["positions"]
    slot_mapping = kw["slot_mapping"]

    flat = np.zeros((num_pages * sb, state_dim), np.float32)
    for t in range(positions.shape[0]):
        slot = int(slot_mapping[t])
        if slot < 0:
            continue
        score_state = score[t] + ape[int(positions[t]) % kw["compress_ratio"]]
        flat[slot] = np.concatenate([kv[t], score_state])
    return flat.reshape(num_pages, sb, state_dim)


def _dequant(fp8, scale, kw):
    """Reconstruct fp32 indexer KV ``[num_slots, head_dim]`` from a record."""
    head_dim = kw["head_dim"]
    n_qb = head_dim // kw["quant_block"]
    fp8 = np.asarray(fp8).reshape(-1, head_dim).astype(np.float32)
    scale = np.asarray(scale).reshape(-1, n_qb).astype(np.float32)
    return (fp8.reshape(-1, n_qb, kw["quant_block"]) *
            scale[:, :, None]).reshape(-1, head_dim)


def _naive_reference(kw):
    """Naive end-to-end NumPy ground truth for the indexer compressor.

    Indexer twin of ``compressor_test._naive_reference``: fp32 projection, the
    partial-state scatter, then the trusted indexer boundary ground truth.
    Returns the dequantized indexer KV per flat KV slot.
    """
    coff = 1 + int(kw["overlap"])
    state_width = coff * kw["head_dim"]

    hidden = kw["hidden_states"].astype(np.float32)
    kv_score = hidden @ kw["wkv_wgate"].astype(np.float32).T
    kv = kv_score[:, :state_width]
    score = kv_score[:, state_width:2 * state_width]

    state_cache = _save_state_ref(kv, score, kw)

    fp8, scale = compress_norm_rope_store_indexer_ref(
        state_cache=state_cache, positions=kw["positions"],
        slot_mapping=kw["slot_mapping"], block_table=kw["block_table"],
        token_to_req_indices=kw["token_to_req_indices"],
        kv_slot_mapping=kw["kv_slot_mapping"], rms_weight=kw["norm_weight"],
        cos_sin_cache=kw["cos_sin_cache"],
        state_block_size=kw["state_block_size"], head_dim=kw["head_dim"],
        rope_head_dim=kw["rope_head_dim"], compress_ratio=kw["compress_ratio"],
        overlap=kw["overlap"], rms_eps=kw["rms_eps"],
        quant_block=kw["quant_block"])
    return _dequant(fp8, scale, kw)


def _dequant_written(act_cache, kw):
    """Unpack the kernel's shared buffer and dequantize every KV row-slot."""
    fp8, scale = unpack_indexer_kv_cache(
        act_cache, kw["head_dim"], kw["quant_block"])
    return _dequant(fp8, scale, kw)


def _written_slots(kw):
    """Flat KV slots actually written (valid boundary tokens)."""
    slots = []
    for t in range(kw["positions"].shape[0]):
        if kw["slot_mapping"][t] < 0 or kw["kv_slot_mapping"][t] < 0:
            continue
        if (int(kw["positions"][t]) + 1) % kw["compress_ratio"] != 0:
            continue
        slots.append(int(kw["kv_slot_mapping"][t]))
    return np.array(sorted(set(slots)), dtype=np.int64)


_CASES = [
    (4, True, 8, 0, 1),
    (4, True, 12, 3, 2),
    (128, False, 256, 0, 3),
    (128, False, 200, 4, 4),
]


@pytest.mark.parametrize(
    "compress_ratio,overlap,seq_len,num_pad,seed", _CASES)
def test_compressor_forward_indexer_matches_reference(
        compress_ratio, overlap, seq_len, num_pad, seed):
    kw = _make_inputs(compress_ratio, overlap, seq_len=seq_len,
                      num_pad=num_pad, seed=seed)
    ref_deq = _naive_reference(kw)

    act_cache = np.asarray(compressor_forward_indexer(**_to_jax(kw)))
    act_deq = _dequant_written(act_cache, kw)

    # Compare dequantized indexer KV at the boundary slots actually written;
    # the tolerance absorbs the projection GEMM's fp32 rounding.
    slots = _written_slots(kw)
    assert slots.size > 0
    np.testing.assert_allclose(
        act_deq[slots], ref_deq[slots], rtol=2e-2, atol=2e-2)


def test_compressor_forward_indexer_eval_shape():
    kw = _make_inputs(4, True, seq_len=8)
    jkw = _to_jax(kw)

    def fn(hidden_states, wkv_wgate, ape, norm_weight, cos_sin_cache,
           positions, slot_mapping, block_table, token_to_req_indices,
           kv_slot_mapping, cache):
        return compressor_forward_indexer(
            hidden_states=hidden_states, wkv_wgate=wkv_wgate, ape=ape,
            norm_weight=norm_weight, cos_sin_cache=cos_sin_cache,
            positions=positions, slot_mapping=slot_mapping,
            block_table=block_table,
            token_to_req_indices=token_to_req_indices,
            kv_slot_mapping=kv_slot_mapping, cache=cache,
            state_block_size=kw["state_block_size"], head_dim=kw["head_dim"],
            rope_head_dim=kw["rope_head_dim"],
            compress_ratio=kw["compress_ratio"], overlap=kw["overlap"],
            rms_eps=kw["rms_eps"], quant_block=kw["quant_block"])

    out_cache = jax.eval_shape(
        fn, jkw["hidden_states"], jkw["wkv_wgate"], jkw["ape"],
        jkw["norm_weight"], jkw["cos_sin_cache"], jkw["positions"],
        jkw["slot_mapping"], jkw["block_table"], jkw["token_to_req_indices"],
        jkw["kv_slot_mapping"], jkw["cache"])
    assert out_cache.shape == kw["cache"].shape
    assert out_cache.dtype == jnp.uint8


@requires_tpu
def test_compressor_forward_indexer_runs_on_tpu():
    kw = _make_inputs(128, False, seq_len=256, num_pad=4, seed=7)
    ref_deq = _naive_reference(kw)

    act = compressor_forward_indexer(**_to_jax(kw))
    act.block_until_ready()
    assert act.devices().pop().platform == "tpu"

    act_deq = _dequant_written(np.asarray(act), kw)
    slots = _written_slots(kw)
    assert slots.size > 0
    np.testing.assert_allclose(
        act_deq[slots], ref_deq[slots], rtol=2e-2, atol=2e-2)
