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
"""Tests for the DeepSeek-V4 KV compressor boundary stage (Milestone 2).

Covers ``compress_norm_rope_store`` and its pieces:
  * a NumPy / ml_dtypes ground truth mirroring the GPU head=512 kernel
    (window gather, masked softmax pool, RMSNorm, interleaved RoPE, fp8 store),
  * randomized numerical-correctness checks for C4 (overlap) and C128,
  * decode / prefill / padding / all-padding batches,
  * a lowering-only check via ``jax.eval_shape`` (no device needed), and
  * an execution check guarded to actually run on a TPU.

Run on CPU (Mac): TPU-guarded tests skip, the rest validate the math.
Run on the TPU VM: everything executes for real.

    .venv/bin/python -m pytest \
        tests/kernels/experimental/deepseek_v4/compress_norm_rope_test.py -v
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from tpu_inference.kernels.experimental.deepseek_v4.compress_norm_rope import (
    compress_norm_rope_store)
from tpu_inference.layers.common.quantization import quantize_tensor

requires_tpu = pytest.mark.skipif(
    jax.devices()[0].platform != "tpu",
    reason="requires a TPU backend",
)


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


def compress_norm_rope_store_ref(
    state_cache, positions, slot_mapping, block_table, token_to_req_indices,
    kv_slot_mapping, nope_cache, rope_cache, scale_cache, rms_weight,
    cos_sin_cache, block_size, head_dim, rope_head_dim, compress_ratio, overlap,
    rms_eps, quant_block,
):
    """Naive NumPy ground truth, one token at a time."""
    nope_out = nope_cache.copy()
    rope_out = rope_cache.copy()
    scale_out = scale_cache.copy()
    coff = 1 + int(overlap)
    state_width = coff * head_dim
    window = coff * compress_ratio
    kv_blk = nope_cache.shape[1]
    nope_dim = head_dim - rope_head_dim

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
            bn = int(block_table[req, p // block_size])
            bo = p % block_size
            head_off = head_dim if w >= compress_ratio else 0
            kv_win[w] = state_cache[bn, bo, head_off:head_off + head_dim]
            score_win[w] = state_cache[bn, bo,
                                       state_width + head_off:
                                       state_width + head_off + head_dim]

        # Masked softmax over the window axis.
        m = np.max(score_win, axis=0, keepdims=True)
        e = np.exp(score_win - m)
        weights = e / np.sum(e, axis=0, keepdims=True)
        compressed_kv = np.sum(weights * kv_win, axis=0)  # [head_dim]

        variance = np.mean(compressed_kv**2)
        normed = compressed_kv / np.sqrt(variance + rms_eps) * rms_weight

        compressed_pos = (int(positions[t]) // compress_ratio) * compress_ratio
        cos_sin = cos_sin_cache[compressed_pos]
        rotated = _interleaved_rope_ref(normed, cos_sin, rope_head_dim)

        nope = rotated[:nope_dim]
        rope = rotated[nope_dim:]

        q, scale = quantize_tensor(
            jnp.float8_e4m3fn,
            jnp.asarray(nope[None]),
            axis=-1,
            block_size=quant_block,
        )
        q = np.asarray(q[0]).astype(ml_dtypes.float8_e4m3fn)
        scale = np.asarray(scale)
        rope_q = rope.astype(ml_dtypes.bfloat16)

        slot = int(kv_slot_mapping[t])
        nope_out[slot // kv_blk, slot % kv_blk] = q
        rope_out[slot // kv_blk, slot % kv_blk] = rope_q
        scale_out[slot // kv_blk, slot % kv_blk] = scale
    return nope_out, rope_out, scale_out


def _make_inputs(
    compress_ratio, overlap, head_dim=512, rope_head_dim=64, quant_block=64,
    num_reqs=2, seq_len=None, num_pad=0, seed=0,
):
    """Build a single-request-major batch of consecutive positions."""
    rng = np.random.default_rng(seed)
    coff = 1 + int(overlap)
    state_width = coff * head_dim
    block_size = 4 if compress_ratio == 4 else 8

    if seq_len is None:
        seq_len = 2 * compress_ratio
    num_tokens = num_reqs * seq_len

    max_pos = seq_len + compress_ratio
    max_blocks = (max_pos + block_size - 1) // block_size + 1
    num_blocks = num_reqs * max_blocks + 2

    state_cache = rng.standard_normal(
        (num_blocks, block_size, 2 * state_width), dtype=np.float32)

    # Per-request contiguous physical blocks.
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

    slot_mapping = np.arange(num_tokens, dtype=np.int32)
    kv_blk = 8
    kv_num_blocks = (num_tokens // kv_blk) + 4
    kv_slot_mapping = rng.permutation(
        kv_num_blocks * kv_blk)[:num_tokens].astype(np.int32)

    if num_pad > 0:
        pad_idx = rng.permutation(num_tokens)[:num_pad]
        slot_mapping[pad_idx] = -1
        kv_slot_mapping[pad_idx] = -1

    rms_weight = rng.standard_normal(head_dim, dtype=np.float32)
    cos_sin_cache = rng.standard_normal(
        (max_pos, rope_head_dim), dtype=np.float32)

    nope_dim = head_dim - rope_head_dim
    n_qb = nope_dim // quant_block
    nope_cache = np.zeros(
        (kv_num_blocks, kv_blk, nope_dim), dtype=ml_dtypes.float8_e4m3fn)
    rope_cache = np.zeros(
        (kv_num_blocks, kv_blk, rope_head_dim), dtype=ml_dtypes.bfloat16)
    scale_cache = np.zeros((kv_num_blocks, kv_blk, n_qb), dtype=np.float32)

    return dict(
        state_cache=state_cache, positions=positions,
        slot_mapping=slot_mapping, block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping, nope_cache=nope_cache,
        rope_cache=rope_cache, scale_cache=scale_cache, rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache, block_size=block_size, head_dim=head_dim,
        rope_head_dim=rope_head_dim, compress_ratio=compress_ratio,
        overlap=overlap, rms_eps=1e-6, quant_block=quant_block)


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
def test_compress_norm_rope_store_matches_reference(
        compress_ratio, overlap, seq_len, num_pad, seed):
    kw = _make_inputs(compress_ratio, overlap, seq_len=seq_len,
                      num_pad=num_pad, seed=seed)
    exp_nope, exp_rope, exp_scale = compress_norm_rope_store_ref(**kw)

    act_nope, act_rope, act_scale = compress_norm_rope_store(**_to_jax(kw))

    # Compare fp8 / bf16 payloads bit-exactly and power-of-two scales tightly.
    np.testing.assert_array_equal(
        np.asarray(act_nope).astype(np.float32), exp_nope.astype(np.float32))
    np.testing.assert_array_equal(
        np.asarray(act_rope).astype(np.float32), exp_rope.astype(np.float32))
    np.testing.assert_allclose(
        np.asarray(act_scale), exp_scale, rtol=1e-6, atol=1e-6)


def test_compress_norm_rope_store_eval_shape():
    """Lowering-only check: traces without executing (no device required)."""
    kw = _make_inputs(4, True, seq_len=8)
    jkw = _to_jax(kw)

    def fn(state_cache, positions, slot_mapping, block_table,
           token_to_req_indices, kv_slot_mapping, nope_cache, rope_cache,
           scale_cache, rms_weight, cos_sin_cache):
        return compress_norm_rope_store(
            state_cache=state_cache, positions=positions,
            slot_mapping=slot_mapping, block_table=block_table,
            token_to_req_indices=token_to_req_indices,
            kv_slot_mapping=kv_slot_mapping, nope_cache=nope_cache,
            rope_cache=rope_cache, scale_cache=scale_cache,
            rms_weight=rms_weight, cos_sin_cache=cos_sin_cache,
            block_size=kw["block_size"], head_dim=kw["head_dim"],
            rope_head_dim=kw["rope_head_dim"],
            compress_ratio=kw["compress_ratio"], overlap=kw["overlap"],
            rms_eps=kw["rms_eps"], quant_block=kw["quant_block"])

    out_nope, out_rope, out_scale = jax.eval_shape(
        fn, jkw["state_cache"], jkw["positions"], jkw["slot_mapping"],
        jkw["block_table"], jkw["token_to_req_indices"],
        jkw["kv_slot_mapping"], jkw["nope_cache"], jkw["rope_cache"],
        jkw["scale_cache"], jkw["rms_weight"], jkw["cos_sin_cache"])
    assert out_nope.shape == kw["nope_cache"].shape
    assert out_nope.dtype == jnp.float8_e4m3fn
    assert out_rope.shape == kw["rope_cache"].shape
    assert out_rope.dtype == jnp.bfloat16
    assert out_scale.shape == kw["scale_cache"].shape


@requires_tpu
def test_compress_norm_rope_store_runs_on_tpu():
    """Executes on TPU and confirms the backend really is TPU."""
    kw = _make_inputs(128, False, seq_len=256, num_pad=4, seed=7)
    exp_nope, exp_rope, exp_scale = compress_norm_rope_store_ref(**kw)

    act_nope, act_rope, act_scale = compress_norm_rope_store(**_to_jax(kw))
    act_nope.block_until_ready()
    assert act_nope.devices().pop().platform == "tpu"

    np.testing.assert_array_equal(
        np.asarray(act_nope).astype(np.float32), exp_nope.astype(np.float32))
    np.testing.assert_array_equal(
        np.asarray(act_rope).astype(np.float32), exp_rope.astype(np.float32))
    np.testing.assert_allclose(
        np.asarray(act_scale), exp_scale, rtol=1e-5, atol=1e-5)
