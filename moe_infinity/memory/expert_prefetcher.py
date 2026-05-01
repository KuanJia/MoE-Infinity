# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team


import numpy as np
import os
from transformers import PretrainedConfig

from moe_infinity.utils import parse_moe_param


def _debug_enabled() -> bool:
    return os.environ.get("MOE_INFINITY_DEBUG", "0").lower() not in {"", "0", "false", "no"}


def _debug_log(message: str) -> None:
    if _debug_enabled():
        print(f"[moe-inf-debug] {message}", flush=True)


class ExpertPrefetcher(object):
    cache_file_rd = None
    first_k_dense_replace: int = 0

    def __init__(self, config: PretrainedConfig):
        print(config)
        self.num_layers, self.num_experts, self.num_encoder_layers = (
            parse_moe_param(config)
        )

    def set_archer_engine(self, archer_engine):
        global _expert_prefetcher
        _expert_prefetcher = archer_engine
        self.archer_engine = archer_engine

    def prefetch_experts_list(self, layer_id, expert_list):
        _debug_log(f"prefetch-list-start layer={layer_id} experts={list(expert_list)}")
        tensor_ids = []
        for j in expert_list:
            tensor_ids.append(self.expert_tensor_map[(layer_id, j)])
        _debug_log(f"prefetch-list-tensors layer={layer_id} tensor_ids={tensor_ids}")
        for tensor_id in tensor_ids:
            gpu_id = self.archer_engine.get_node_default_device([tensor_id])
            _debug_log(f"prefetch-enqueue-start layer={layer_id} tensor_id={tensor_id} gpu_id={gpu_id}")
            self.archer_engine.enqueue_prefetch(tensor_id, gpu_id)
            _debug_log(f"prefetch-enqueue-finish layer={layer_id} tensor_id={tensor_id} gpu_id={gpu_id}")
        _debug_log(f"prefetch-list-finish layer={layer_id} count={len(tensor_ids)}")

    def fetch_experts_lock_cache(self, layer_id, expert_list):
        tensor_ids = []
        for j in expert_list:
            tensor_ids.append(self.expert_tensor_map[(layer_id, j)])
        self.archer_engine.replace_cache_candidates(tensor_ids)

    def prefetch_experts(self, layer_id, expert_matrix):
        expert_list = []
        # print("expert_tensor_map", self.expert_tensor_map)
        for i in range(layer_id, self.num_layers):
            for j in range(self.num_experts):
                if expert_matrix[i, j] > 0:
                    expert_list.append(
                        (self.expert_tensor_map[(i, j)], expert_matrix[i, j])
                    )
        ordered_expert_list = sorted(
            expert_list, key=lambda x: x[1], reverse=True
        )
        tensor_ids = [x[0] for x in ordered_expert_list]
        assert len(np.unique(tensor_ids)) == len(tensor_ids)
        self.archer_engine.replace_cache_candidates(tensor_ids)
        for tensor_id in tensor_ids:
            gpu_id = self.archer_engine.get_node_default_device([tensor_id])
            self.archer_engine.enqueue_prefetch(tensor_id, gpu_id)
