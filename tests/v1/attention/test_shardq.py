# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ token split and exchanges (vllm/v1/attention/ops/shardq.py)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.ops import shardq
from vllm.v1.attention.ops.shardq import (
    ShardQMetadata,
    allocate_shardq_buffers,
    build_shardq_split,
    shard_q,
    unshard_out,
    zigzag_split,
)


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("lens", [[4096], [65, 300, 7000], [1000, 64, 64, 129]])
def test_zigzag_split_partitions_tokens(world, lens):
    qsl = np.r_[0, np.cumsum(lens)]
    token_ids, num_tokens_per_rank = zigzag_split(np.array(lens), world)
    per_rank = np.split(token_ids, np.cumsum(num_tokens_per_rank)[:-1])
    merged = np.sort(np.concatenate(per_rank))
    np.testing.assert_array_equal(merged, np.arange(qsl[-1]))
    for idx in per_rank:
        assert np.all(np.diff(idx) > 0)
    # Each rank takes one early and one late chunk of every request, so its
    # summed causal positions stay within a couple of tokens per request of
    # the mean.
    positions = [
        sum(int((idx[(idx >= a) & (idx < b)] - a).sum()) for a, b in zip(qsl, qsl[1:]))
        for idx in per_rank
    ]
    assert max(positions) - min(positions) <= sum(4 * n for n in lens)


@pytest.mark.parametrize("num_spec", [0, 2])
def test_build_shardq_split_after_indexer_decodes(monkeypatch, num_spec):
    """The MLA and indexer builders derive the same split from the batch:
    requests of at most 1 + num_spec tokens stay head-sharded, and the tokens
    after them are split once they reach sparse_mla_shard_q_min_tokens and a
    request is longer than the top-k."""
    monkeypatch.setattr(
        shardq, "get_tp_group", lambda: SimpleNamespace(world_size=4, rank_in_group=1)
    )
    query_lens = [1 + num_spec] * 3 + [100, 5000, 4000]
    metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=[q + 300 for q in query_lens], query_lens=query_lens),
        block_size=16,
        device=torch.device("cpu"),
    )
    attention_config = SimpleNamespace(sparse_mla_shard_q_min_tokens=9100)
    hf_config = SimpleNamespace(index_topk=5299)
    vllm_config = SimpleNamespace(
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=num_spec) if num_spec else None
        ),
        attention_config=attention_config,
        model_config=SimpleNamespace(hf_text_config=hf_config),
    )
    split = build_shardq_split(metadata, vllm_config)
    assert split is not None
    assert split.num_decode_tokens == 3 * (1 + num_spec)
    assert split.num_tokens_per_rank.sum() == 9100
    hf_config.index_topk = 5300
    assert build_shardq_split(metadata, vllm_config) is None
    hf_config.index_topk = 5299
    attention_config.sparse_mla_shard_q_min_tokens = 9101
    assert build_shardq_split(metadata, vllm_config) is None
    # Five tokens split into eight chunks, three of them empty, leaving rank 2
    # without tokens.
    attention_config.sparse_mla_shard_q_min_tokens = 1
    short = create_common_attn_metadata(
        BatchSpec(seq_lens=[5300], query_lens=[5]),
        block_size=16,
        device=torch.device("cpu"),
    )
    assert build_shardq_split(short, vllm_config) is None


def _round_trip(rank, world, port, use_sm, q_dtype, num_decode_tokens):
    torch.accelerator.set_device_index(rank)
    device = torch.device(current_platform.device_type, rank)
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
            backend=current_platform.dist_backend,
        )
        ensure_model_parallel_initialized(world, 1)
        torch.manual_seed(0)
        lens = [700, 64, 3001]
        heads, dim = 4, 576
        num_tokens = num_decode_tokens + sum(lens)
        vllm_config.scheduler_config.max_num_batched_tokens = num_tokens
        if use_sm:
            allocate_shardq_buffers(
                vllm_config, heads, dim, 512, q_dtype, torch.bfloat16
            )
        all_heads = world * heads
        full = torch.randn(num_tokens, all_heads, dim, device=device).to(q_dtype)
        token_ids, num_tokens_per_rank = zigzag_split(np.array(lens), world)
        shardq = ShardQMetadata(
            num_decode_tokens=num_decode_tokens,
            num_tokens_per_rank=num_tokens_per_rank.tolist(),
            token_ids=torch.from_numpy(token_ids).to(device),
            rank_offsets=torch.from_numpy(np.r_[0, np.cumsum(num_tokens_per_rank)]).to(
                device
            ),
        )
        local_heads = full[:, rank * heads : (rank + 1) * heads].contiguous()
        for _ in range(2):  # The second pass reuses the symmetric buffers.
            decode_q, sharded = shard_q(local_heads, shardq)
            # Bitwise, as the exchanges only move bytes (and fp8 lacks
            # torch.equal).
            assert torch.equal(
                decode_q.view(torch.uint8),
                local_heads[:num_decode_tokens].view(torch.uint8),
            )
            expected = full[num_decode_tokens:][shardq.local_token_ids]
            assert torch.equal(sharded.view(torch.uint8), expected.view(torch.uint8))
            # Attention outputs are a head-padded slice; unshard reads its
            # strides.
            padded = torch.zeros(sharded.shape[0], all_heads + 8, 512, device=device)
            padded[:, :all_heads] = sharded[..., :512].float()
            out = padded.bfloat16()[:, :all_heads]
            restored = unshard_out(out, shardq, decode_q[..., :512].bfloat16())
            assert torch.equal(restored, local_heads[..., :512].bfloat16())
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("use_sm", [False, True])
@pytest.mark.parametrize("num_decode_tokens", [0, 33])
def test_shard_q_round_trip(use_sm, q_dtype, num_decode_tokens):
    """The exchanges move each rank's tokens to it with all heads and back,
    through symmetric memory or NCCL, leaving the decode tokens head-sharded."""
    world = 2
    if torch.accelerator.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    mp.spawn(
        _round_trip,
        args=(world, get_open_port(), use_sm, q_dtype, num_decode_tokens),
        nprocs=world,
    )


def _v_up_exchange(rank, world, port, use_sm, exchange_value, num_decode_tokens):
    torch.accelerator.set_device_index(rank)
    device = torch.device(current_platform.device_type, rank)
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
            backend=current_platform.dist_backend,
        )
        ensure_model_parallel_initialized(world, 1)
        torch.manual_seed(0)
        lens = [700, 64, 3001]
        heads, dim, latent, v_dim = 4, 576, 512, 128
        num_tokens = num_decode_tokens + sum(lens)
        vllm_config.scheduler_config.max_num_batched_tokens = num_tokens
        if use_sm:
            allocate_shardq_buffers(
                vllm_config,
                heads,
                dim,
                latent,
                torch.bfloat16,
                torch.bfloat16,
                wuv_scratch_dim=v_dim if exchange_value else None,
            )
        all_heads = world * heads
        # Integer-valued bf16: the fp32-accumulated bmm and the fp64 reference
        # both stay exact, so the two orders must agree bitwise.
        full = torch.randint(
            -8, 8, (num_tokens - num_decode_tokens, all_heads, latent), device=device
        ).bfloat16()
        full_decode = torch.randint(
            -8, 8, (num_decode_tokens, all_heads, latent), device=device
        ).bfloat16()
        w_uv = torch.randint(
            -4, 4, (all_heads, latent, v_dim), device=device
        ).bfloat16()
        w_local = w_uv[rank * heads : (rank + 1) * heads]
        token_ids, num_tokens_per_rank = zigzag_split(np.array(lens), world)
        shardq = ShardQMetadata(
            num_decode_tokens=num_decode_tokens,
            num_tokens_per_rank=num_tokens_per_rank.tolist(),
            token_ids=torch.from_numpy(token_ids).to(device),
            rank_offsets=torch.from_numpy(np.r_[0, np.cumsum(num_tokens_per_rank)]).to(
                device
            ),
        )
        local = full[shardq.local_token_ids]
        decode_local = full_decode[:, rank * heads : (rank + 1) * heads].contiguous()
        if exchange_value:
            # Up-project every head on the token shard, then exchange v-space.
            v = local.new_empty((local.shape[0], all_heads, v_dim))
            torch.bmm(local.transpose(0, 1), w_uv, out=v.transpose(0, 1))
            decode_v = None
            if num_decode_tokens:
                decode_v = decode_local.new_empty((num_decode_tokens, heads, v_dim))
                torch.bmm(
                    decode_local.transpose(0, 1), w_local, out=decode_v.transpose(0, 1)
                )
            out = unshard_out(v, shardq, decode_v)
        else:
            # Exchange the latent, then up-project this rank's heads.
            restored = unshard_out(local, shardq, decode_local)
            x = restored.view(num_tokens, heads, latent).transpose(0, 1)
            out = restored.new_empty((num_tokens, heads, v_dim))
            torch.bmm(x, w_local, out=out.transpose(0, 1))
        expected = torch.empty((num_tokens, heads, v_dim), dtype=torch.float64)
        expected[num_decode_tokens:] = torch.einsum(
            "thl,hlv->thv",
            full[:, rank * heads : (rank + 1) * heads].double().cpu(),
            w_local.double().cpu(),
        )
        if num_decode_tokens:
            expected[:num_decode_tokens] = torch.einsum(
                "nhl,hlv->nhv", decode_local.double().cpu(), w_local.double().cpu()
            )
        assert out.shape == expected.shape
        assert torch.equal(out, expected.bfloat16().to(device))
        if use_sm and exchange_value:
            # A later small step exchanges the latent through the same slot
            # the value exchange packed its narrower rows into.
            restored = unshard_out(local, shardq, decode_local)
            x = restored.view(num_tokens, heads, latent).transpose(0, 1)
            out_latent = restored.new_empty((num_tokens, heads, v_dim))
            torch.bmm(x, w_local, out=out_latent.transpose(0, 1))
            assert torch.equal(out_latent, expected.bfloat16().to(device))
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("use_sm", [False, True])
@pytest.mark.parametrize("exchange_value", [False, True])
@pytest.mark.parametrize("num_decode_tokens", [0, 33])
def test_shardq_v_up_exchange_order(use_sm, exchange_value, num_decode_tokens):
    """Both ShardQ orders produce the same o_proj input: exchanging the
    latent and up-projecting the local heads, or up-projecting all heads on
    the token shard and exchanging the v-space output (the narrower exchange
    the sparse-MLA prefill uses)."""
    world = 2
    if torch.accelerator.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    mp.spawn(
        _v_up_exchange,
        args=(world, get_open_port(), use_sm, exchange_value, num_decode_tokens),
        nprocs=world,
    )
