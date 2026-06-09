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
"""Tests for the DeepSeek-V4 KV compressor state cache (Milestone 1).

Covers ``save_partial_states``:
  * a NumPy ground truth mirroring the GPU Triton kernel,
  * randomized numerical-correctness checks (incl. padding tokens),
  * a lowering-only check via ``jax.eval_shape`` (no device needed), and
  * an execution check guarded to actually run on a TPU.

Run on CPU (Mac): TPU-guarded tests skip, the rest validate the math.
Run on the TPU VM: everything executes for real.

    .venv/bin/python -m pytest \
        tests/kernels/experimental/deepseek_v4/compress_store_test.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.experimental.deepseek_v4.compress_store import (
    save_partial_states)

requires_tpu = pytest.mark.skipif(
    jax.devices()[0].platform != "tpu",
    reason="requires a TPU backend",
)


def save_partial_states_ref(
    kv: np.ndarray,
    score: np.ndarray,
    ape: np.ndarray,
    positions: np.ndarray,
    state_cache: np.ndarray,
    slot_mapping: np.ndarray,
    compress_ratio: int,
) -> np.ndarray:
    """Naive NumPy ground truth for ``save_partial_states``.

    One token at a time, mirroring ``_save_partial_states_kernel``: padding
    tokens (``slot < 0``) are skipped; otherwise the slot row is
    ``[kv | score + ape[position % compress_ratio]]``.
    """
    out = state_cache.copy()
    _, block_size, two_w = out.shape
    state_width = two_w // 2
    num_tokens = kv.shape[0]
    for i in range(num_tokens):
        slot = int(slot_mapping[i])
        if slot < 0:
            continue
        block_idx = slot // block_size
        pos_in_block = slot % block_size
        ape_row = int(positions[i]) % compress_ratio
        out[block_idx, pos_in_block, :state_width] = kv[i]
        out[block_idx, pos_in_block, state_width:] = score[i] + ape[ape_row]
    return out


def _make_inputs(
    num_tokens: int,
    num_blocks: int,
    block_size: int,
    state_width: int,
    compress_ratio: int,
    num_pad: int = 0,
    seed: int = 0,
):
    """Build random inputs with unique valid slots and ``num_pad`` -1 pads."""
    rng = np.random.default_rng(seed)
    kv = rng.standard_normal((num_tokens, state_width), dtype=np.float32)
    score = rng.standard_normal((num_tokens, state_width), dtype=np.float32)
    ape = rng.standard_normal((compress_ratio, state_width), dtype=np.float32)
    positions = rng.integers(
        0, compress_ratio * 8, size=(num_tokens,), dtype=np.int32)

    num_slots = num_blocks * block_size
    assert num_tokens - num_pad <= num_slots, "not enough slots for valid tokens"
    # Unique slots for valid tokens so scatter order is unambiguous.
    perm = rng.permutation(num_slots)[: num_tokens - num_pad]
    slot_mapping = np.full((num_tokens,), -1, dtype=np.int32)
    valid_token_idx = rng.permutation(num_tokens)[: num_tokens - num_pad]
    slot_mapping[valid_token_idx] = perm.astype(np.int32)

    # Start from a non-zero cache to prove untouched rows are preserved.
    state_cache = rng.standard_normal(
        (num_blocks, block_size, 2 * state_width), dtype=np.float32)
    return kv, score, ape, positions, state_cache, slot_mapping


_CASES = [
    # num_tokens, num_blocks, block_size, state_width, compress_ratio, num_pad
    (6, 4, 16, 8, 4, 0),
    (1, 2, 16, 8, 1, 0),       # single-token decode
    (8, 4, 16, 16, 128, 2),    # C128 layer, with padding
    (20, 8, 32, 32, 4, 5),     # prefill-ish, many pads
    (3, 1, 4, 8, 4, 3),        # all-padding batch (no writes)
]


@pytest.mark.parametrize(
    "num_tokens,num_blocks,block_size,state_width,compress_ratio,num_pad",
    _CASES,
)
def test_save_partial_states_matches_reference(
    num_tokens, num_blocks, block_size, state_width, compress_ratio, num_pad
):
    kv, score, ape, positions, state_cache, slot_mapping = _make_inputs(
        num_tokens, num_blocks, block_size, state_width, compress_ratio,
        num_pad=num_pad, seed=num_tokens + compress_ratio)

    expected = save_partial_states_ref(
        kv, score, ape, positions, state_cache, slot_mapping, compress_ratio)

    actual = save_partial_states(
        jnp.asarray(kv),
        jnp.asarray(score),
        jnp.asarray(ape),
        jnp.asarray(positions),
        jnp.asarray(state_cache),
        jnp.asarray(slot_mapping),
        compress_ratio=compress_ratio,
    )
    np.testing.assert_allclose(
        np.asarray(actual), expected, rtol=1e-6, atol=1e-6)


def test_save_partial_states_eval_shape():
    """Lowering-only check: traces without executing (no device required)."""
    num_tokens, num_blocks, block_size, state_width, compress_ratio = (
        6, 4, 16, 8, 4)
    kv, score, ape, positions, state_cache, slot_mapping = _make_inputs(
        num_tokens, num_blocks, block_size, state_width, compress_ratio)

    out = jax.eval_shape(
        lambda *a: save_partial_states(*a, compress_ratio=compress_ratio),
        jnp.asarray(kv),
        jnp.asarray(score),
        jnp.asarray(ape),
        jnp.asarray(positions),
        jnp.asarray(state_cache),
        jnp.asarray(slot_mapping),
    )
    assert out.shape == (num_blocks, block_size, 2 * state_width)
    assert out.dtype == jnp.float32


@requires_tpu
def test_save_partial_states_runs_on_tpu():
    """Executes on TPU and confirms the backend really is TPU."""
    num_tokens, num_blocks, block_size, state_width, compress_ratio = (
        20, 8, 32, 32, 4)
    kv, score, ape, positions, state_cache, slot_mapping = _make_inputs(
        num_tokens, num_blocks, block_size, state_width, compress_ratio,
        num_pad=4)
    expected = save_partial_states_ref(
        kv, score, ape, positions, state_cache, slot_mapping, compress_ratio)

    actual = save_partial_states(
        jnp.asarray(kv),
        jnp.asarray(score),
        jnp.asarray(ape),
        jnp.asarray(positions),
        jnp.asarray(state_cache),
        jnp.asarray(slot_mapping),
        compress_ratio=compress_ratio,
    )
    actual.block_until_ready()
    assert actual.devices().pop().platform == "tpu"
    np.testing.assert_allclose(
        np.asarray(actual), expected, rtol=1e-6, atol=1e-6)
