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
"""Tests for the DeepSeek-V4 lightning-indexer compressor boundary stage.

Covers ``compress_norm_rope_store_indexer`` (the ``head_dim == 128`` path):
  * a NumPy / ml_dtypes ground truth mirroring the GPU indexer kernel
    (window gather, masked softmax pool, RMSNorm, interleaved RoPE, then a
    *single-block* UE8M0 fp8 quant over the whole head — rope tail included),
  * randomized numerical-correctness checks for C4 (overlap) and C128,
  * a lowering-only check via ``jax.eval_shape`` (no device needed), and
  * an execution check guarded to actually run on a TPU.

Run on CPU (Mac): the TPU-guarded test skips, the rest validate the math.
Run on the TPU VM: everything executes for real.

    .venv/bin/python -m pytest \
        tests/kernels/experimental/deepseek_v4/compress_norm_rope_indexer_test.py -v
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from tpu_inference.kernels.experimental.deepseek_v4.compress_norm_rope import (
    compress_norm_rope_store_indexer, pack_state_cache,
    shared_indexer_cache_shape, unpack_indexer_kv_cache)
from tpu_inference.layers.common.quantization import quantize_tensor

requires_tpu = pytest.mark.skipif(
    jax.devices()[0].platform != "tpu",
    reason="requires a TPU backend",
)

# KV row-slots per shared-cache page (the MLA storage block size).
PAGE_SIZE = 64

def _interleaved_rope_ref(x, cos_sin, rope_head_dim):
    """NumPy interleaved (non-NeoX) RoPE on the trailing rope_head_dim elems."""
    head_dim = x.shape[-1]
    half_rope = rope_head_dim // 2
    num_pairs = head_dim // 2
    nope_pairs = num_pairs - half_rope

    pairs = x.reshape(*x.shape[:-1], num_pairs, 2)
    even = pairs[..., 0].copy()
    odd = pairs[..., 1].copy()

    cos = cos_sin[..., :half_rope]
    sin = cos_sin[..., half_rope:rope_head_dim]
    cos_full = np.concatenate(
        [np.ones((*cos.shape[:-1], nope_pairs), x.dtype), cos], axis=-1)
    sin_full = np.concatenate(
        [np.zeros((*sin.shape[:-1], nope_pairs), x.dtype), sin], axis=-1)

    new_even = even * cos_full - odd * sin_full
    new_odd = odd * cos_full + even * sin_full
    out = np.stack([new_even, new_odd], axis=-1)
    return out.reshape(x.shape)


def compress_norm_rope_store_indexer_ref(
    state_cache, positions, slot_mapping, block_table, token_to_req_indices,
    kv_slot_mapping, rms_weight, cos_sin_cache, state_block_size, head_dim,
    rope_head_dim, compress_ratio, overlap, rms_eps, quant_block,
):
    """Naive NumPy ground truth, one token at a time.

    ``state_cache`` is the logical f32 view ``[num_pages, sb, state_dim]`` (the
    same values the kernel reads back from the shared buffer). Returns the
    (fp8, scale) components keyed by flat KV slot (``num_pages * PAGE_SIZE``).
    """
    coff = 1 + int(overlap)
    state_width = coff * head_dim
    window = coff * compress_ratio
    num_pages = state_cache.shape[0]
    num_slots = num_pages * PAGE_SIZE
    n_qb = head_dim // quant_block
    fp8_out = np.zeros((num_slots, head_dim), dtype=ml_dtypes.float8_e4m3fn)
    scale_out = np.zeros((num_slots, n_qb), dtype=np.float32)

    num_tokens = positions.shape[0]
    for t in range(num_tokens):
        if slot_mapping[t] < 0 or kv_slot_mapping[t] < 0:
            continue
        if (int(positions[t]) + 1) % compress_ratio != 0:
            continue

        req = int(token_to_req_indices[t])
        start = int(positions[t]) - window + 1

        kv_win = np.zeros((window, head_dim), np.float32)
        score_win = np.full((window, head_dim), -np.inf, np.float32)
        for w in range(window):
            p = start + w
            if p < 0:
                continue
            bn = int(block_table[req, p // state_block_size])
            bo = p % state_block_size
            head_off = head_dim if w >= compress_ratio else 0
            kv_win[w] = state_cache[bn, bo, head_off:head_off + head_dim]
            score_win[w] = state_cache[bn, bo,
                                       state_width + head_off:
                                       state_width + head_off + head_dim]

        m = np.max(score_win, axis=0, keepdims=True)
        e = np.exp(score_win - m)
        weights = e / np.sum(e, axis=0, keepdims=True)
        compressed_kv = np.sum(weights * kv_win, axis=0)  # [head_dim]

        variance = np.mean(compressed_kv**2)
        normed = compressed_kv / np.sqrt(variance + rms_eps) * rms_weight

        compressed_pos = (int(positions[t]) // compress_ratio) * compress_ratio
        cos_sin = cos_sin_cache[compressed_pos]
        rotated = _interleaved_rope_ref(normed, cos_sin, rope_head_dim)

        q, scale = quantize_tensor(
            jnp.float8_e4m3fn,
            jnp.asarray(rotated[None]),
            axis=-1,
            block_size=quant_block,
        )
        q = np.asarray(q[0]).astype(ml_dtypes.float8_e4m3fn)
        scale = np.asarray(scale)

        slot = int(kv_slot_mapping[t])
        fp8_out[slot] = q
        scale_out[slot] = scale
    return fp8_out, scale_out


def _make_inputs(
    compress_ratio, overlap, head_dim=128, rope_head_dim=64, quant_block=128,
    num_reqs=2, seq_len=None, num_pad=0, seed=0,
):
    """Build a single-request-major batch backed by one shared cache buffer.

    State and compressed-KV share the same ``[num_pages, ...]`` buffer but live
    in disjoint page ranges (state pages first, then KV pages) so the boundary
    writes never clobber state bytes the gather still needs.
    """
    rng = np.random.default_rng(seed)
    coff = 1 + int(overlap)
    state_width = coff * head_dim
    state_dim = 2 * state_width
    state_block_size = 4 if compress_ratio == 4 else 8

    if seq_len is None:
        seq_len = 2 * compress_ratio
    num_tokens = num_reqs * seq_len

    max_pos = seq_len + compress_ratio
    max_blocks = (max_pos + state_block_size - 1) // state_block_size + 1
    num_state_pages = num_reqs * max_blocks + 2

    # Compressed-KV output pages follow the state pages in the same buffer.
    num_kv_pages = (num_tokens // PAGE_SIZE) + 4
    num_pages = num_state_pages + num_kv_pages

    # Logical f32 state view; only state pages are reached via block_table.
    state_cache = rng.standard_normal(
        (num_pages, state_block_size, state_dim), dtype=np.float32)

    block_table = np.zeros((num_reqs, max_blocks), np.int32)
    nxt = 1
    for r in range(num_reqs):
        for b in range(max_blocks):
            block_table[r, b] = nxt
            nxt += 1
    assert nxt <= num_state_pages

    positions = np.concatenate(
        [np.arange(seq_len, dtype=np.int32) for _ in range(num_reqs)])
    token_to_req_indices = np.repeat(
        np.arange(num_reqs, dtype=np.int32), seq_len)

    slot_mapping = np.arange(num_tokens, dtype=np.int32)
    kv_base = num_state_pages * PAGE_SIZE
    kv_capacity = num_kv_pages * PAGE_SIZE
    kv_slot_mapping = (
        kv_base + rng.permutation(kv_capacity)[:num_tokens]).astype(np.int32)

    if num_pad > 0:
        pad_idx = rng.permutation(num_tokens)[:num_pad]
        slot_mapping[pad_idx] = -1
        kv_slot_mapping[pad_idx] = -1

    rms_weight = rng.standard_normal(head_dim, dtype=np.float32)
    cos_sin_cache = rng.standard_normal(
        (max_pos, rope_head_dim), dtype=np.float32)

    # Allocate the shared uint8 buffer and pack the state view into it.
    cache_shape = shared_indexer_cache_shape(
        num_pages, PAGE_SIZE, head_dim, quant_block)
    cache = np.zeros(cache_shape, dtype=np.uint8)
    cache = np.asarray(
        pack_state_cache(jnp.asarray(cache), jnp.asarray(state_cache)))

    return dict(
        cache=cache, state_cache=state_cache, positions=positions,
        slot_mapping=slot_mapping, block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping, rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache, state_block_size=state_block_size,
        head_dim=head_dim, rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio, overlap=overlap, rms_eps=1e-6,
        quant_block=quant_block)


def _kernel_kwargs(kw):
    """Kernel args: everything except the f32 reference-only ``state_cache``."""
    return {k: v for k, v in kw.items() if k != "state_cache"}


def _ref_kwargs(kw):
    """Reference args: everything except the packed ``cache`` buffer."""
    return {k: v for k, v in kw.items() if k != "cache"}


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


def _unpack_written(act_kv, kw, slots):
    """Unpack the kernel's shared buffer and gather the written KV slots."""
    head_dim = kw["head_dim"]
    n_qb = head_dim // kw["quant_block"]
    act_fp8, act_scale = unpack_indexer_kv_cache(
        act_kv, head_dim, kw["quant_block"])
    act_fp8 = np.asarray(act_fp8).reshape(-1, head_dim)[slots]
    act_scale = np.asarray(act_scale).reshape(-1, n_qb)[slots]
    return act_fp8, act_scale


def _to_jax(kw):
    return {
        k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
        for k, v in kw.items()
    }


# compress_ratio, overlap, seq_len, num_pad, seed
_CASES = [
    (4, True, 8, 0, 1),     # C4 overlap, exactly one boundary window per req
    (4, True, 12, 2, 2),    # C4 with padding
    (128, False, 256, 0, 3),  # C128, two boundaries per req
    (128, False, 200, 5, 4),  # C128 with padding, partial last window
    (4, True, 3, 0, 5),     # short seq: no boundary tokens at all
]


@pytest.mark.parametrize(
    "compress_ratio,overlap,seq_len,num_pad,seed", _CASES)
def test_compress_norm_rope_store_indexer_matches_reference(
        compress_ratio, overlap, seq_len, num_pad, seed):
    kw = _make_inputs(compress_ratio, overlap, seq_len=seq_len,
                      num_pad=num_pad, seed=seed)
    exp_fp8, exp_scale = compress_norm_rope_store_indexer_ref(
        **_ref_kwargs(kw))

    act_kv = compress_norm_rope_store_indexer(**_to_jax(_kernel_kwargs(kw)))

    # The shared buffer also holds packed state bytes, so compare only the
    # slots the boundary store actually wrote.
    slots = _written_slots(kw)
    act_fp8, act_scale = _unpack_written(act_kv, kw, slots)

    # Compare the fp8 payload bit-exactly and power-of-two scales tightly.
    np.testing.assert_array_equal(
        act_fp8.astype(np.float32), exp_fp8[slots].astype(np.float32))
    np.testing.assert_allclose(
        act_scale, exp_scale[slots], rtol=1e-6, atol=1e-6)


def test_compress_norm_rope_store_indexer_eval_shape():
    """Lowering-only check: traces without executing (no device required)."""
    kw = _make_inputs(4, True, seq_len=8)
    jkw = _to_jax(_kernel_kwargs(kw))

    def fn(cache, positions, slot_mapping, block_table,
           token_to_req_indices, kv_slot_mapping, rms_weight, cos_sin_cache):
        return compress_norm_rope_store_indexer(
            cache=cache, positions=positions, slot_mapping=slot_mapping,
            block_table=block_table,
            token_to_req_indices=token_to_req_indices,
            kv_slot_mapping=kv_slot_mapping, rms_weight=rms_weight,
            cos_sin_cache=cos_sin_cache,
            state_block_size=kw["state_block_size"], head_dim=kw["head_dim"],
            rope_head_dim=kw["rope_head_dim"],
            compress_ratio=kw["compress_ratio"], overlap=kw["overlap"],
            rms_eps=kw["rms_eps"], quant_block=kw["quant_block"])

    out_kv = jax.eval_shape(
        fn, jkw["cache"], jkw["positions"], jkw["slot_mapping"],
        jkw["block_table"], jkw["token_to_req_indices"],
        jkw["kv_slot_mapping"], jkw["rms_weight"], jkw["cos_sin_cache"])
    assert out_kv.shape == kw["cache"].shape
    assert out_kv.dtype == jnp.uint8


@requires_tpu
def test_compress_norm_rope_store_indexer_runs_on_tpu():
    """Executes on TPU and confirms the backend really is TPU."""
    kw = _make_inputs(128, False, seq_len=256, num_pad=4, seed=7)
    exp_fp8, exp_scale = compress_norm_rope_store_indexer_ref(
        **_ref_kwargs(kw))

    act_kv = compress_norm_rope_store_indexer(**_to_jax(_kernel_kwargs(kw)))
    act_kv.block_until_ready()
    assert act_kv.devices().pop().platform == "tpu"

    slots = _written_slots(kw)
    act_fp8, act_scale = _unpack_written(act_kv, kw, slots)

    np.testing.assert_array_equal(
        act_fp8.astype(np.float32), exp_fp8[slots].astype(np.float32))
    np.testing.assert_allclose(
        act_scale, exp_scale[slots], rtol=1e-5, atol=1e-5)
