# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import numpy as np
import os
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc

from moe_infinity.utils import ArcherConfig


def _debug_enabled() -> bool:
    return os.environ.get("MOE_INFINITY_DEBUG", "0").lower() not in {"", "0", "false", "no"}


def _debug_log(message: str) -> None:
    if _debug_enabled():
        print(f"[moe-inf-debug] {message}", flush=True)


def _call_expert_dispatcher(method, *args, **kwargs):
    global _expert_dispatcher
    func = getattr(_expert_dispatcher, method)
    return func(*args, **kwargs)


class DistributedExpertExecutor:
    def __init__(self, archer_config: ArcherConfig):
        self.archer_config = archer_config

    def set_expert_dispatcher(self, expert_dispatcher):
        global _expert_dispatcher
        _expert_dispatcher = expert_dispatcher
        self.expert_dispatcher = expert_dispatcher

    def set_device_map_manager(self, device_map_manager):
        self.device_map_manager = device_map_manager

    def dispatch_local(
        self, layer_id, hidden_states, router_mask, router_weights
    ):
        _debug_log(
            "executor-dispatch-local-start "
            f"layer={layer_id} hidden_shape={tuple(hidden_states.shape)} hidden_dtype={hidden_states.dtype} "
            f"hidden_device={hidden_states.device} router_mask_shape={tuple(router_mask.shape)} "
            f"router_mask_dtype={router_mask.dtype} router_weights_shape={tuple(router_weights.shape)} "
            f"router_weights_dtype={router_weights.dtype}"
        )
        num_expert = router_mask.shape[-1]
        _debug_log(f"executor-expert-count-start layer={layer_id} num_expert={num_expert}")
        expert_count = (
            torch.sum(router_mask.view((-1, num_expert)), dim=0)
            .cpu()
            .numpy()
            .flatten()
        )
        _debug_log(
            "executor-expert-count-finish "
            f"layer={layer_id} active={(expert_count > 0).sum()} "
            f"max_tokens_per_expert={int(expert_count.max()) if expert_count.size else 0}"
        )

        expert_list = (
            np.arange(num_expert).astype(int)[expert_count > 0].tolist()
        )
        expected_wait_cnt = len(expert_list)

        _debug_log(
            "executor-set-inputs-start "
            f"layer={layer_id} expected_wait_cnt={expected_wait_cnt} expert_list={expert_list}"
        )
        self.expert_dispatcher.set_inputs(
            hidden_states, router_mask.bool(), router_weights
        )
        _debug_log(f"executor-set-inputs-finish layer={layer_id}")
        self.expert_dispatcher.set_expected_queue(expected_wait_cnt)
        _debug_log(f"executor-set-expected-finish layer={layer_id} expected_wait_cnt={expected_wait_cnt}")

        total_gpus = torch.cuda.device_count()
        for expert_id in expert_list:
            gpu_id = expert_id % total_gpus
            _debug_log(
                "executor-enqueue-start "
                f"layer={layer_id} expert={expert_id} gpu_id={gpu_id} "
                f"tokens={int(expert_count[expert_id])}"
            )
            self.expert_dispatcher.enqueue_expert(
                layer_id, expert_id, gpu_id, False
            )
            _debug_log(f"executor-enqueue-finish layer={layer_id} expert={expert_id} gpu_id={gpu_id}")
        _debug_log(f"executor-notify-fetch-start layer={layer_id}")
        self.expert_dispatcher.notify_fetch_start()
        _debug_log(f"executor-notify-fetch-finish layer={layer_id}")

    def wait_dispatch_local(self):
        _debug_log("executor-wait-start")
        result = self.expert_dispatcher.wait_expert()
        _debug_log(
            "executor-wait-finish "
            f"is_tensor={isinstance(result, torch.Tensor)} "
            f"shape={tuple(result.shape) if isinstance(result, torch.Tensor) else '-'} "
            f"dtype={result.dtype if isinstance(result, torch.Tensor) else '-'} "
            f"device={result.device if isinstance(result, torch.Tensor) else '-'}"
        )
        if isinstance(result, torch.Tensor) and result.is_cuda:
            # The native dispatcher writes the returned tensor on its own CUDA
            # stream. Make the wait API honor that work before the next layer
            # consumes or overwrites dispatcher state.
            torch.cuda.synchronize(result.device)

        return result

    def dispatch(self, hidden_states, router_mask, layer_id):
        num_expert = router_mask.shape[-1]
        expert_count = (
            torch.sum(router_mask.view((-1, num_expert)), dim=0)
            .cpu()
            .numpy()
            .flatten()
        )

        expert_list = (
            np.arange(num_expert).astype(int)[expert_count > 0].tolist()
        )

        device_list = self.device_map_manager.get_target_device(expert_list)
        visited_ranks = set()
        rank_wait_cnt = {r: 0 for r in range(dist.get_world_size())}
        for k, device_meta in enumerate(device_list):
            rank, gpu_id, expert_id = device_meta
            visited_ranks.add(rank)
            rank_wait_cnt[rank] += 1

        futures = []
        for rank in visited_ranks:
            if rank != dist.get_rank():
                future = rpc.rpc_async(
                    f"worker_{rank}",
                    _call_expert_dispatcher,
                    args=("set_inputs", hidden_states.cpu(), router_mask.cpu()),
                )
                futures.append(future)
                future = rpc.rpc_async(
                    f"worker_{rank}",
                    _call_expert_dispatcher,
                    args=("set_expected_queue", rank_wait_cnt[rank]),
                )
                futures.append(future)
            else:
                self.expert_dispatcher.set_inputs(hidden_states, router_mask)
                self.expert_dispatcher.set_expected_queue(rank_wait_cnt[rank])

        # wait for all futures
        for future in futures:
            future.wait()

        futures = []
        for k, device_meta in enumerate(device_list):
            rank, gpu_id, expert_id = device_meta
            if rank == dist.get_rank():
                self.expert_dispatcher.enqueue_expert(
                    layer_id, expert_id, gpu_id, False
                )
            else:
                future = rpc.rpc_async(
                    f"worker_{rank}",
                    _call_expert_dispatcher,
                    args=("enqueue_expert", layer_id, expert_id, gpu_id, True),
                )
                futures.append(future)

        # wait for all futures
        for future in futures:
            future.wait()

        result_list = []
        for rank in visited_ranks:
            if rank != dist.get_rank():
                result = rpc.rpc_sync(
                    f"worker_{rank}",
                    _call_expert_dispatcher,
                    args=("wait_expert",),
                )
                result_list += result
            else:
                result = self.expert_dispatcher.wait_expert()
                result_list += result

        return result_list
