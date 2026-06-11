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
"""Tier-1 differential tests for the DeepSeek-V4 KV compressor boundary stage.

Instead of trusting a hand-written NumPy ground truth (see
``compress_norm_rope_test.py`` / ``compress_norm_rope_indexer_test.py``), this
module compares the JAX kernels against the *real* vLLM GPU Triton kernels run
under ``TRITON_INTERPRET=1`` (CPU):

  * ``compress_norm_rope_store`` vs
    ``_fused_kv_compress_norm_rope_insert_sparse_attn`` (head=512), and
  * ``compress_norm_rope_store_indexer`` vs
    ``_fused_kv_compress_norm_rope_insert_indexer_attn`` (head=128).

Both paths consume identical inputs; their byte-packed (GPU) and paged (TPU)
outputs are decoded back to fp32 ``[num_boundary_tokens, head_dim]`` and
compared.

The JAX kernels use ``quantize_tensor`` (absmax / ``finfo.max`` scales). The
GPU Triton oracle still uses UE8M0 power-of-two scales and bf16 round-trips,
so decoded outputs are compared semantically (per-token cosine + rel-L2), not
bit-for-bit. Scale-exponent equality is not expected and those checks are
skipped.

GPU-oracle tests are marked ``gpu_oracle`` and skip automatically when triton
/ torch / the vLLM kernel are unavailable. Oracle-free invariant tests at the
bottom always run.

    # Unit tests only (no GPU oracle):
    .venv/bin/python -m pytest tests/kernels/experimental/deepseek_v4/ \
        --ignore=tests/kernels/experimental/deepseek_v4/compress_norm_rope_diff_test.py

    # GPU oracle only (needs vllm on PYTHONPATH, interpret mode = CPU):
    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
        tests/kernels/experimental/deepseek_v4/compress_norm_rope_diff_test.py \
        -m gpu_oracle -v
"""

import os

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from tpu_inference.kernels.experimental.deepseek_v4.compress_norm_rope import (
    compress_norm_rope_store, compress_norm_rope_store_indexer,
    indexer_packed_width, interleaved_rope, sparse_packed_width,
    unpack_indexer_kv_cache, unpack_sparse_kv_cache)
from tpu_inference.layers.common.quantization import quantize_tensor

# Triton (imported lazily in ``_load_gpu_kernels``) must run in interpret mode
# so the GPU kernel executes on CPU. Set before any triton import happens.
os.environ.setdefault("TRITON_INTERPRET", "1")


def _load_gpu_kernels():
    """Import the real vLLM Triton kernels (interpret mode).

    Returns ``(triton, sparse_attn_kernel, indexer_kernel)`` or
    ``(None, None, None)`` if triton / torch / vLLM are unavailable.
    """
    try:
        import torch  # noqa: F401
        import triton

        from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache \
            import (_fused_kv_compress_norm_rope_insert_indexer_attn
                    as indexer_kernel)
        from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache \
            import (_fused_kv_compress_norm_rope_insert_sparse_attn
                    as sparse_kernel)

        # vLLM hands out a no-op ``triton`` placeholder when it decides triton
        # is disabled (e.g. no active GPU driver), in which case ``@triton.jit``
        # returns the bare function. A real JIT kernel is subscriptable
        # (``kernel[grid]``); a placeholder function is not. Skip rather than
        # fail cryptically. Set ``CUDA_VISIBLE_DEVICES=""`` to keep real triton
        # (see vllm/triton_utils/importing.py).
        if not hasattr(sparse_kernel, "__getitem__"):
            return None, None, None
        return triton, sparse_kernel, indexer_kernel
    except Exception:  # noqa: BLE001 - any failure → skip the differential test
        return None, None, None


_TRITON, _GPU_KERNEL, _GPU_KERNEL_INDEXER = _load_gpu_kernels()
requires_gpu_kernel = pytest.mark.skipif(
    _GPU_KERNEL is None,
    reason="triton / torch / vLLM sparse-attn compressor kernel unavailable",
)
requires_gpu_kernel_indexer = pytest.mark.skipif(
    _GPU_KERNEL_INDEXER is None,
    reason="triton / torch / vLLM indexer compressor kernel unavailable",
)


# ============================================================================
# Shared inputs
# ============================================================================
def _make_inputs(
    compress_ratio, overlap, head_dim=512, rope_head_dim=64, quant_block=64,
    num_reqs=2, seq_len=None, num_pad=0, seed=0,
):
    """Build one batch consumed identically by the GPU and JAX kernels.

    The compressor ``state_cache`` is random (the boundary stage only reads it
    via ``block_table`` + ``positions``, exactly like stage 1's writes), and a
    single KV-cache geometry (``kv_blk`` tokens/block) is shared so a boundary
    token's slot decodes the same way on both sides.
    """
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

    return dict(
        state_cache=state_cache, positions=positions,
        slot_mapping=slot_mapping, block_table=block_table,
        token_to_req_indices=token_to_req_indices,
        kv_slot_mapping=kv_slot_mapping, rms_weight=rms_weight,
        cos_sin_cache=cos_sin_cache, block_size=block_size, head_dim=head_dim,
        rope_head_dim=rope_head_dim, compress_ratio=compress_ratio,
        overlap=overlap, rms_eps=1e-6, quant_block=quant_block,
        kv_blk=kv_blk, kv_num_blocks=kv_num_blocks)


def _boundary_slots(kw):
    """KV slots of tokens that the kernels actually write (in token order)."""
    cr = kw["compress_ratio"]
    pos = kw["positions"]
    store = (((pos + 1) % cr) == 0) & (kw["slot_mapping"] >= 0) \
        & (kw["kv_slot_mapping"] >= 0)
    return kw["kv_slot_mapping"][store].astype(np.int64)


# ============================================================================
# Decoders: byte-packed (GPU) and three-tensor (TPU) → fp32 [n, head_dim]
# ============================================================================
def _decode_gpu(byte_cache, slots, kv_blk, token_stride, scale_dim, nope_dim,
                rope_head_dim, quant_block):
    """Decode the GPU byte-packed cache for the given KV slots.

    Per cache block the bytes are segmented as the kernel writes them:
    ``[kv_blk * token_stride]`` token data, then ``[kv_blk * scale_dim]``
    UE8M0 scale bytes. Each token's ``token_stride`` bytes are
    ``[nope_dim fp8 | rope_head_dim bf16]``; its scales are ``exponent + 127``.
    """
    n_nb = nope_dim // quant_block
    flat = np.asarray(byte_cache).reshape(byte_cache.shape[0], -1)
    out = np.zeros((len(slots), nope_dim + rope_head_dim), np.float32)
    for i, slot in enumerate(slots):
        b, p = divmod(int(slot), kv_blk)
        block = flat[b]
        td = block[p * token_stride:p * token_stride + token_stride]
        nope = td[:nope_dim].view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        rope = td[nope_dim:nope_dim + rope_head_dim * 2].view(
            ml_dtypes.bfloat16).astype(np.float32)
        scales_region = block[kv_blk * token_stride:]
        sc = scales_region[p * scale_dim:p * scale_dim + scale_dim]
        scale = (2.0 ** (sc[:n_nb].astype(np.int32) - 127)).astype(np.float32)
        out[i, :nope_dim] = nope * np.repeat(scale, quant_block)
        out[i, nope_dim:] = rope
    return out


def _decode_tpu(kv_cache, slots, kv_blk, quant_block, nope_dim,
                rope_head_dim):
    """Decode the JAX packed sparse KV cache for the given KV slots."""
    nope_c, rope_c, scale_c = unpack_sparse_kv_cache(
        kv_cache, nope_dim, rope_head_dim, quant_block)
    nope = np.asarray(nope_c).astype(np.float32)
    rope = np.asarray(rope_c).astype(np.float32)
    scale = np.asarray(scale_c).astype(np.float32)
    out = np.zeros((len(slots), nope_dim + rope_head_dim), np.float32)
    for i, slot in enumerate(slots):
        b, p = divmod(int(slot), kv_blk)
        out[i, :nope_dim] = nope[b, p] * np.repeat(scale[b, p], quant_block)
        out[i, nope_dim:] = rope[b, p]
    return out


def _decode_gpu_indexer(byte_cache, slots, kv_blk, token_stride, scale_dim):
    """Decode the GPU byte-packed indexer cache (whole-head fp8 + fp32 scale).

    Per cache block: ``[kv_blk * token_stride]`` fp8 token data, then
    ``[kv_blk * scale_dim]`` scale bytes. Each token's scale is a single
    power-of-two float32 (not a UE8M0 exponent byte like the sparse path).
    """
    head_dim = token_stride
    flat = np.asarray(byte_cache).reshape(byte_cache.shape[0], -1)
    out = np.zeros((len(slots), head_dim), np.float32)
    for i, slot in enumerate(slots):
        b, p = divmod(int(slot), kv_blk)
        block = flat[b]
        td = block[p * token_stride:p * token_stride + token_stride]
        fp8 = td.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        scales_region = block[kv_blk * token_stride:]
        scale = scales_region[p * scale_dim:p * scale_dim + 4].view(
            np.float32)[0]
        out[i] = fp8 * scale
    return out


def _decode_tpu_indexer(kv_cache, slots, kv_blk, head_dim, quant_block):
    """Decode the JAX packed indexer cache (fp8 + single fp32 scale)."""
    fp8_c, scale_c = unpack_indexer_kv_cache(kv_cache, head_dim, quant_block)
    fp8 = np.asarray(fp8_c).astype(np.float32)
    scale = np.asarray(scale_c).astype(np.float32)
    out = np.zeros((len(slots), head_dim), np.float32)
    for i, slot in enumerate(slots):
        b, p = divmod(int(slot), kv_blk)
        out[i] = fp8[b, p] * scale[b, p, 0]
    return out


# ============================================================================
# Runners
# ============================================================================
def _run_gpu(kw):
    import torch
    triton, kernel = _TRITON, _GPU_KERNEL

    head_dim = kw["head_dim"]
    rope_head_dim = kw["rope_head_dim"]
    quant_block = kw["quant_block"]
    coff = 1 + int(kw["overlap"])
    state_width = coff * head_dim
    nope_dim = head_dim - rope_head_dim
    token_stride = nope_dim + rope_head_dim * 2          # 448 + 128 = 576
    scale_dim = nope_dim // quant_block + 1              # 7 + 1 pad = 8
    kv_blk = kw["kv_blk"]
    num_tokens = kw["positions"].shape[0]

    state_cache = torch.tensor(kw["state_cache"], dtype=torch.float32)
    token_to_req = torch.tensor(
        kw["token_to_req_indices"], dtype=torch.int32)
    positions = torch.tensor(kw["positions"], dtype=torch.int32)
    slot_mapping = torch.tensor(kw["slot_mapping"], dtype=torch.int32)
    block_table = torch.tensor(kw["block_table"], dtype=torch.int32)
    rms_w = torch.tensor(kw["rms_weight"], dtype=torch.float32)
    cos_sin = torch.tensor(kw["cos_sin_cache"], dtype=torch.float32)
    kv_slot = torch.tensor(kw["kv_slot_mapping"], dtype=torch.int32)
    byte_cache = torch.zeros(
        (kw["kv_num_blocks"], kv_blk, token_stride + scale_dim),
        dtype=torch.uint8)

    kernel[(num_tokens,)](
        state_cache, state_cache.stride(0), state_cache.stride(1),
        token_to_req, positions, slot_mapping,
        block_table, block_table.stride(0), kw["block_size"],
        rms_w, float(kw["rms_eps"]),
        cos_sin, cos_sin.stride(0),
        byte_cache, kv_slot, byte_cache.shape[1],
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=kw["compress_ratio"],
        OVERLAP=kw["overlap"],
        ROPE_HEAD_DIM=rope_head_dim,
        FP8_MAX=448.0,
        QUANT_BLOCK=quant_block,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=byte_cache.stride(0),
        num_warps=4,
    )
    return byte_cache.cpu().numpy(), token_stride, scale_dim


def _run_tpu(kw):
    head_dim = kw["head_dim"]
    rope_head_dim = kw["rope_head_dim"]
    quant_block = kw["quant_block"]
    nope_dim = head_dim - rope_head_dim
    packed_width = sparse_packed_width(nope_dim, rope_head_dim, quant_block)
    kv_cache = np.zeros(
        (kw["kv_num_blocks"], kw["kv_blk"], packed_width), dtype=np.uint8)

    return compress_norm_rope_store(
        state_cache=jax.numpy.asarray(kw["state_cache"]),
        positions=jax.numpy.asarray(kw["positions"]),
        slot_mapping=jax.numpy.asarray(kw["slot_mapping"]),
        block_table=jax.numpy.asarray(kw["block_table"]),
        token_to_req_indices=jax.numpy.asarray(kw["token_to_req_indices"]),
        kv_slot_mapping=jax.numpy.asarray(kw["kv_slot_mapping"]),
        kv_cache=jax.numpy.asarray(kv_cache),
        rms_weight=jax.numpy.asarray(kw["rms_weight"]),
        cos_sin_cache=jax.numpy.asarray(kw["cos_sin_cache"]),
        block_size=kw["block_size"], head_dim=head_dim,
        rope_head_dim=rope_head_dim, compress_ratio=kw["compress_ratio"],
        overlap=kw["overlap"], rms_eps=kw["rms_eps"], quant_block=quant_block)


def _run_gpu_indexer(kw):
    import torch
    triton, kernel = _TRITON, _GPU_KERNEL_INDEXER

    head_dim = kw["head_dim"]
    rope_head_dim = kw["rope_head_dim"]
    quant_block = kw["quant_block"]
    coff = 1 + int(kw["overlap"])
    state_width = coff * head_dim
    token_stride = head_dim                             # 128 fp8 bytes/token
    scale_dim = 4                                       # one float32 scale
    kv_blk = kw["kv_blk"]
    num_tokens = kw["positions"].shape[0]

    state_cache = torch.tensor(kw["state_cache"], dtype=torch.float32)
    token_to_req = torch.tensor(
        kw["token_to_req_indices"], dtype=torch.int32)
    positions = torch.tensor(kw["positions"], dtype=torch.int32)
    slot_mapping = torch.tensor(kw["slot_mapping"], dtype=torch.int32)
    block_table = torch.tensor(kw["block_table"], dtype=torch.int32)
    rms_w = torch.tensor(kw["rms_weight"], dtype=torch.float32)
    cos_sin = torch.tensor(kw["cos_sin_cache"], dtype=torch.float32)
    kv_slot = torch.tensor(kw["kv_slot_mapping"], dtype=torch.int32)
    byte_cache = torch.zeros(
        (kw["kv_num_blocks"], kv_blk, token_stride + scale_dim),
        dtype=torch.uint8)

    kernel[(num_tokens,)](
        state_cache, state_cache.stride(0), state_cache.stride(1),
        token_to_req, positions, slot_mapping,
        block_table, block_table.stride(0), kw["block_size"],
        rms_w, float(kw["rms_eps"]),
        cos_sin, cos_sin.stride(0),
        byte_cache, kv_slot, byte_cache.shape[1],
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=kw["compress_ratio"],
        OVERLAP=kw["overlap"],
        ROPE_HEAD_DIM=rope_head_dim,
        FP8_MAX=448.0,
        QUANT_BLOCK=quant_block,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=byte_cache.stride(0),
        num_warps=1,
    )
    return byte_cache.cpu().numpy(), token_stride, scale_dim


def _run_tpu_indexer(kw):
    head_dim = kw["head_dim"]
    quant_block = kw["quant_block"]
    packed_width = indexer_packed_width(head_dim, quant_block)
    kv_cache = np.zeros(
        (kw["kv_num_blocks"], kw["kv_blk"], packed_width), dtype=np.uint8)

    return compress_norm_rope_store_indexer(
        state_cache=jax.numpy.asarray(kw["state_cache"]),
        positions=jax.numpy.asarray(kw["positions"]),
        slot_mapping=jax.numpy.asarray(kw["slot_mapping"]),
        block_table=jax.numpy.asarray(kw["block_table"]),
        token_to_req_indices=jax.numpy.asarray(kw["token_to_req_indices"]),
        kv_slot_mapping=jax.numpy.asarray(kw["kv_slot_mapping"]),
        kv_cache=jax.numpy.asarray(kv_cache),
        rms_weight=jax.numpy.asarray(kw["rms_weight"]),
        cos_sin_cache=jax.numpy.asarray(kw["cos_sin_cache"]),
        block_size=kw["block_size"], head_dim=head_dim,
        rope_head_dim=kw["rope_head_dim"], compress_ratio=kw["compress_ratio"],
        overlap=kw["overlap"], rms_eps=kw["rms_eps"], quant_block=quant_block)


def _assert_token_close(gpu, tpu, name, min_cos, max_rel_l2):
    """Assert two ``[tokens, dim]`` reconstructions agree per token.

    Per-*element* relative error is misleading for block-quantized data: all
    elements in a quant block share one absmax-derived scale, so a single
    fp8-code difference on an element far below its block's absmax yields a
    huge relative error even when the vector is essentially identical. A wiring
    or index bug, by contrast, structurally corrupts the whole vector. So we
    compare per-token cosine similarity and relative L2 norm, which tolerate
    sparse 1-2 code fp8/bf16 rounding differences between the TPU (JAX) and CPU
    (Triton interpret) backends while still catching real disagreement.
    """
    gn = np.linalg.norm(gpu, axis=1)
    tn = np.linalg.norm(tpu, axis=1)
    cos = np.sum(gpu * tpu, axis=1) / (gn * tn + 1e-12)
    rel_l2 = np.linalg.norm(gpu - tpu, axis=1) / (tn + 1e-12)
    worst_cos = float(cos.min())
    worst_l2 = float(rel_l2.max())
    assert worst_cos >= min_cos and worst_l2 <= max_rel_l2, (
        f"{name}: worst cosine={worst_cos:.5f} (min {min_cos}); "
        f"worst rel-L2={worst_l2:.4f} (max {max_rel_l2})")


# ============================================================================
# Differential tests (skip without the GPU kernel)
# ============================================================================
# compress_ratio, overlap, seq_len, num_pad, seed
_DIFF_CASES = [
    (4, True, 8, 0, 1),
    (4, True, 12, 2, 2),
    (128, False, 256, 0, 3),
    (128, False, 200, 5, 4),
]


@requires_gpu_kernel
@pytest.mark.gpu_oracle
@pytest.mark.parametrize(
    "compress_ratio,overlap,seq_len,num_pad,seed", _DIFF_CASES)
def test_matches_gpu_triton_sparse_kernel(
        compress_ratio, overlap, seq_len, num_pad, seed):
    kw = _make_inputs(compress_ratio, overlap, seq_len=seq_len,
                      num_pad=num_pad, seed=seed)
    slots = _boundary_slots(kw)
    assert slots.size > 0, "test case has no boundary tokens"

    byte_cache, token_stride, scale_dim = _run_gpu(kw)
    tpu_kv = _run_tpu(kw)

    nope_dim = kw["head_dim"] - kw["rope_head_dim"]
    gpu = _decode_gpu(
        byte_cache, slots, kw["kv_blk"], token_stride, scale_dim, nope_dim,
        kw["rope_head_dim"], kw["quant_block"])
    tpu = _decode_tpu(
        tpu_kv, slots, kw["kv_blk"], kw["quant_block"], nope_dim,
        kw["rope_head_dim"])

    # The nope head is fp8 (3 mantissa bits) and Triton's interpret-mode fp8
    # cast rounds differently from JAX's by up to ~1 ULP/element (see the
    # spike's 2.5e-1 max err), perturbing per-token cosine to ~0.997. The
    # bf16 rope tail (8 mantissa bits) is far less sensitive, so we hold it
    # tight. Exact-exponent + tight-rope + correct-structure together is the
    # real correctness signal; a wiring bug tanks cosine far below 0.99.
    _assert_token_close(gpu[:, :nope_dim], tpu[:, :nope_dim], "nope/fp8",
                        min_cos=0.99, max_rel_l2=0.15)
    _assert_token_close(gpu[:, nope_dim:], tpu[:, nope_dim:], "rope/bf16",
                        min_cos=0.999, max_rel_l2=0.03)


_INDEXER_DIFF_CASES = [
    (4, True, 8, 0, 1),
    (4, True, 12, 2, 2),
    (128, False, 256, 0, 3),
    (128, False, 200, 5, 4),
]


@requires_gpu_kernel_indexer
@pytest.mark.gpu_oracle
@pytest.mark.parametrize(
    "compress_ratio,overlap,seq_len,num_pad,seed", _INDEXER_DIFF_CASES)
def test_matches_gpu_triton_indexer_kernel(
        compress_ratio, overlap, seq_len, num_pad, seed):
    kw = _make_inputs(
        compress_ratio, overlap, head_dim=128, quant_block=128,
        seq_len=seq_len, num_pad=num_pad, seed=seed)
    slots = _boundary_slots(kw)
    assert slots.size > 0, "test case has no boundary tokens"

    byte_cache, token_stride, scale_dim = _run_gpu_indexer(kw)
    tpu_kv = _run_tpu_indexer(kw)

    gpu = _decode_gpu_indexer(
        byte_cache, slots, kw["kv_blk"], token_stride, scale_dim)
    tpu = _decode_tpu_indexer(
        tpu_kv, slots, kw["kv_blk"], kw["head_dim"], kw["quant_block"])

    # Whole head is fp8 (rope tail included); interpret-mode rounding applies
    # to all 128 dims, so use the same relaxed tolerance as the sparse nope.
    _assert_token_close(gpu, tpu, "indexer/fp8",
                        min_cos=0.99, max_rel_l2=0.15)


@pytest.mark.skip(
    reason="JAX quantize_tensor scales differ from GPU UE8M0 exponents")
@requires_gpu_kernel
@pytest.mark.gpu_oracle
def test_gpu_and_tpu_sparse_scales_are_identical_exponents():
    """GPU UE8M0 exponents vs JAX quantize_tensor scales (not comparable)."""
    kw = _make_inputs(128, False, seq_len=256, seed=11)
    slots = _boundary_slots(kw)
    nope_dim = kw["head_dim"] - kw["rope_head_dim"]
    qb = kw["quant_block"]
    n_nb = nope_dim // qb

    byte_cache, token_stride, scale_dim = _run_gpu(kw)
    _, _, scale_c = unpack_sparse_kv_cache(
        _run_tpu(kw), nope_dim, kw["rope_head_dim"], qb)
    scale_cache = np.asarray(scale_c).astype(np.float32)

    flat = byte_cache.reshape(byte_cache.shape[0], -1)
    for slot in slots:
        b, p = divmod(int(slot), kw["kv_blk"])
        sc = flat[b][kw["kv_blk"] * token_stride:][
            p * scale_dim:p * scale_dim + scale_dim]
        gpu_exp = sc[:n_nb].astype(np.int32) - 127
        tpu_exp = np.round(np.log2(scale_cache[b, p])).astype(np.int32)
        np.testing.assert_array_equal(gpu_exp, tpu_exp)


@pytest.mark.skip(
    reason="JAX quantize_tensor scales differ from GPU UE8M0 exponents")
@requires_gpu_kernel_indexer
@pytest.mark.gpu_oracle
def test_gpu_and_tpu_indexer_scales_are_identical_exponents():
    """GPU UE8M0 scale vs JAX quantize_tensor scale (not comparable)."""
    kw = _make_inputs(
        128, False, head_dim=128, quant_block=128, seq_len=256, seed=12)
    slots = _boundary_slots(kw)

    byte_cache, token_stride, scale_dim = _run_gpu_indexer(kw)
    _, scale_c = unpack_indexer_kv_cache(
        _run_tpu_indexer(kw), kw["head_dim"], kw["quant_block"])
    scale_cache = np.asarray(scale_c).astype(np.float32)

    flat = byte_cache.reshape(byte_cache.shape[0], -1)
    for slot in slots:
        b, p = divmod(int(slot), kw["kv_blk"])
        sc = flat[b][kw["kv_blk"] * token_stride:][
            p * scale_dim:p * scale_dim + scale_dim]
        gpu_exp = int(np.round(np.log2(sc.view(np.float32)[0])))
        tpu_exp = int(np.round(np.log2(scale_cache[b, p, 0])))
        assert gpu_exp == tpu_exp


# ============================================================================
# Oracle-free invariants (always run)
# ============================================================================
def test_interleaved_rope_preserves_norm():
    """A proper rotation preserves the norm of the rotated tail exactly and
    leaves the nope head untouched."""
    rng = np.random.default_rng(0)
    head_dim, rope_head_dim = 512, 64
    x = rng.standard_normal((7, head_dim)).astype(np.float32)
    theta = rng.standard_normal((7, rope_head_dim // 2)).astype(np.float32)
    cos_sin = np.concatenate([np.cos(theta), np.sin(theta)], axis=-1)

    out = np.asarray(interleaved_rope(
        jax.numpy.asarray(x), jax.numpy.asarray(cos_sin), rope_head_dim))

    nope_dim = head_dim - rope_head_dim
    np.testing.assert_allclose(out[:, :nope_dim], x[:, :nope_dim], atol=1e-5)
    np.testing.assert_allclose(
        np.linalg.norm(out[:, nope_dim:], axis=-1),
        np.linalg.norm(x[:, nope_dim:], axis=-1), rtol=1e-5, atol=1e-5)


def test_fp8_block_scale_is_positive():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((5, 448)) * rng.uniform(0.01, 50)).astype(
        np.float32)
    _, scale = quantize_tensor(
        jnp.float8_e4m3fn, jnp.asarray(x), axis=-1, block_size=64)
    scale = np.asarray(scale).astype(np.float32)
    assert np.all(scale > 0)


def test_fp8_values_within_range():
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((5, 448)) * 100).astype(np.float32)
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    q, _ = quantize_tensor(
        jnp.float8_e4m3fn, jnp.asarray(x), axis=-1, block_size=64)
    q = np.asarray(q).astype(np.float32)
    assert np.all(np.abs(q) <= fp8_max)
    assert np.all(np.isfinite(q))
