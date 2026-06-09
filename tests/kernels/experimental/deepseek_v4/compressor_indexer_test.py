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

``compressor_forward_indexer`` composes the projection GEMM,
``save_partial_states`` (Milestone 1) and
``compress_norm_rope_store_indexer`` (Milestone 2 head=128 path). The
reference drives the two trusted JAX kernels directly after the same JAX
projection matmul.

    .venv/bin/python -m pytest \
        tests/kernels/experimental/deepseek_v4/compressor_indexer_test.py -v
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from tpu_inference.kernels.experimental.deepseek_v4.compress_norm_rope import (
    compress_norm_rope_store_indexer)
from tpu_inference.kernels.experimental.deepseek_v4.compress_store import (
    save_partial_states)
from tpu_inference.kernels.experimental.deepseek_v4.compressor import (
    compressor_forward_indexer)

requires_tpu = pytest.mark.skipif(
    jax.devices()[0].platform != "tpu",
    reason="requires a TPU backend",
)


def _make_inputs(
    compress_ratio, overlap, head_dim=128, rope_head_dim=64, quant_block=128,
    hidden_size=256, num_reqs=2, seq_len=None, num_pad=0, seed=0,
):
    """Build a consistent batch: compressor slots follow the block table."""
    rng = np.random.default_rng(seed)
    coff = 1 + int(overlap)
    state_width = coff * head_dim
    block_size = 4 if compress_ratio == 4 else 8

    if seq_len is None:
        seq_len = 2 * compress_ratio
    num_tokens = num_reqs * seq_len

    max_blocks = (seq_len + block_size - 1) // block_size
    num_state_blocks = num_reqs * max_blocks + 1

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

    slot_mapping = np.empty(num_tokens, np.int32)
    for t in range(num_tokens):
        r = int(token_to_req_indices[t])
        p = int(positions[t])
        slot_mapping[t] = block_table[r, p // block_size] * block_size \
            + p % block_size

    kv_blk = 8
    kv_num_blocks = (num_tokens // kv_blk) + 4
    kv_slot_mapping = rng.permutation(
        kv_num_blocks * kv_blk)[:num_tokens].astype(np.int32)

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

    n_qb = head_dim // quant_block
    state_cache = np.zeros(
        (num_state_blocks, block_size, 2 * state_width), dtype=np.float32)
    fp8_cache = np.zeros(
        (kv_num_blocks, kv_blk, head_dim), dtype=ml_dtypes.float8_e4m3fn)
    scale_cache = np.zeros((kv_num_blocks, kv_blk, n_qb), dtype=np.float32)

    return dict(
        hidden_states=hidden_states, wkv_wgate=wkv_wgate, ape=ape,
        norm_weight=norm_weight, cos_sin_cache=cos_sin_cache,
        positions=positions, slot_mapping=slot_mapping,
        block_table=block_table, token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping, state_cache=state_cache,
        fp8_cache=fp8_cache, scale_cache=scale_cache,
        block_size=block_size, head_dim=head_dim, rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio, overlap=overlap, rms_eps=1e-6,
        quant_block=quant_block)


def _to_jax(kw):
    return {
        k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v)
        for k, v in kw.items()
    }


def _reference(kw):
    coff = 1 + int(kw["overlap"])
    state_width = coff * kw["head_dim"]

    hidden = jnp.asarray(kw["hidden_states"]).astype(jnp.float32)
    kv_score = hidden @ jnp.asarray(kw["wkv_wgate"]).T
    kv = kv_score[:, :state_width]
    score = kv_score[:, state_width:2 * state_width]

    state_cache = save_partial_states(
        kv=kv, score=score, ape=jnp.asarray(kw["ape"]),
        positions=jnp.asarray(kw["positions"]),
        state_cache=jnp.asarray(kw["state_cache"]),
        slot_mapping=jnp.asarray(kw["slot_mapping"]),
        compress_ratio=kw["compress_ratio"])

    fp8_cache, scale_cache = compress_norm_rope_store_indexer(
        state_cache=state_cache,
        positions=jnp.asarray(kw["positions"]),
        slot_mapping=jnp.asarray(kw["slot_mapping"]),
        block_table=jnp.asarray(kw["block_table"]),
        token_to_req_indices=jnp.asarray(kw["token_to_req_indices"]),
        kv_slot_mapping=jnp.asarray(kw["kv_slot_mapping"]),
        fp8_cache=jnp.asarray(kw["fp8_cache"]),
        scale_cache=jnp.asarray(kw["scale_cache"]),
        rms_weight=jnp.asarray(kw["norm_weight"]),
        cos_sin_cache=jnp.asarray(kw["cos_sin_cache"]),
        block_size=kw["block_size"], head_dim=kw["head_dim"],
        rope_head_dim=kw["rope_head_dim"],
        compress_ratio=kw["compress_ratio"], overlap=kw["overlap"],
        rms_eps=kw["rms_eps"], quant_block=kw["quant_block"])
    return state_cache, fp8_cache, scale_cache


_CASES = [
    (4, True, 8, 0, 1),
    (4, True, 12, 3, 2),
    (128, False, 256, 0, 3),
    (128, False, 200, 4, 4),
]


@pytest.mark.parametrize(
    "compress_ratio,overlap,seq_len,num_pad,seed", _CASES)
def test_compressor_forward_indexer_matches_composition(
        compress_ratio, overlap, seq_len, num_pad, seed):
    kw = _make_inputs(compress_ratio, overlap, seq_len=seq_len,
                      num_pad=num_pad, seed=seed)
    exp_state, exp_fp8, exp_scale = _reference(kw)

    act_state, act_fp8, act_scale = compressor_forward_indexer(**_to_jax(kw))

    np.testing.assert_allclose(
        np.asarray(act_state), np.asarray(exp_state), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(
        np.asarray(act_fp8).astype(np.float32),
        np.asarray(exp_fp8).astype(np.float32))
    np.testing.assert_allclose(
        np.asarray(act_scale), np.asarray(exp_scale), rtol=1e-6, atol=1e-6)

    assert np.any(np.asarray(act_state) != 0.0)
    assert np.any(np.asarray(act_fp8).astype(np.float32) != 0.0)


def test_compressor_forward_indexer_eval_shape():
    kw = _make_inputs(4, True, seq_len=8)
    jkw = _to_jax(kw)

    def fn(hidden_states, wkv_wgate, ape, norm_weight, cos_sin_cache,
           positions, slot_mapping, block_table, token_to_req_indices,
           kv_slot_mapping, state_cache, fp8_cache, scale_cache):
        return compressor_forward_indexer(
            hidden_states=hidden_states, wkv_wgate=wkv_wgate, ape=ape,
            norm_weight=norm_weight, cos_sin_cache=cos_sin_cache,
            positions=positions, slot_mapping=slot_mapping,
            block_table=block_table,
            token_to_req_indices=token_to_req_indices,
            kv_slot_mapping=kv_slot_mapping, state_cache=state_cache,
            fp8_cache=fp8_cache, scale_cache=scale_cache,
            block_size=kw["block_size"], head_dim=kw["head_dim"],
            rope_head_dim=kw["rope_head_dim"],
            compress_ratio=kw["compress_ratio"], overlap=kw["overlap"],
            rms_eps=kw["rms_eps"], quant_block=kw["quant_block"])

    out_state, out_fp8, out_scale = jax.eval_shape(
        fn, jkw["hidden_states"], jkw["wkv_wgate"], jkw["ape"],
        jkw["norm_weight"], jkw["cos_sin_cache"], jkw["positions"],
        jkw["slot_mapping"], jkw["block_table"], jkw["token_to_req_indices"],
        jkw["kv_slot_mapping"], jkw["state_cache"], jkw["fp8_cache"],
        jkw["scale_cache"])
    assert out_state.shape == kw["state_cache"].shape
    assert out_fp8.shape == kw["fp8_cache"].shape
    assert out_fp8.dtype == jnp.float8_e4m3fn
    assert out_scale.shape == kw["scale_cache"].shape


@requires_tpu
def test_compressor_forward_indexer_runs_on_tpu():
    kw = _make_inputs(128, False, seq_len=256, num_pad=4, seed=7)
    exp_state, exp_fp8, exp_scale = _reference(kw)

    act_state, act_fp8, act_scale = compressor_forward_indexer(**_to_jax(kw))
    act_fp8.block_until_ready()
    assert act_fp8.devices().pop().platform == "tpu"

    np.testing.assert_allclose(
        np.asarray(act_state), np.asarray(exp_state), rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(
        np.asarray(act_fp8).astype(np.float32),
        np.asarray(exp_fp8).astype(np.float32))
    np.testing.assert_allclose(
        np.asarray(act_scale), np.asarray(exp_scale), rtol=1e-5, atol=1e-5)
