import os

import nvtx
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP

import moe_infinity._store as prefetch_lib


def _debug_enabled() -> bool:
    return os.environ.get("MOE_INFINITY_DEBUG", "0").lower() not in {"", "0", "false", "no"}


def _sync_debug_enabled() -> bool:
    return os.environ.get("MOE_INFINITY_SYNC_DEBUG", "0").lower() not in {"", "0", "false", "no"}


def _debug_log(message: str) -> None:
    if _debug_enabled():
        print(f"[moe-inf-debug] {message}", flush=True)


def _debug_sync(label: str, device: torch.device | None = None) -> None:
    if not _sync_debug_enabled() or not torch.cuda.is_available():
        return
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    else:
        torch.cuda.synchronize()
    _debug_log(f"sync label={label}")


def _get_qwen3_num_experts(config) -> int:
    for name in ("num_experts", "n_routed_experts", "num_local_experts"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise AttributeError(
        "Qwen3 config is missing expert-count attributes: "
        "num_experts / n_routed_experts / num_local_experts"
    )


class Qwen3MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = _get_qwen3_num_experts(config)
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for _ in range(self.num_experts)
            ]
        )

        # runtime-injected handles (set by model_offload patching)
        self.lib = None
        self.expert_executor = None
        self.layer_id = None

    @nvtx.annotate("Qwen3Prepare", color="blue")
    def __prepare_expert_route(self, hidden_states):
        # router_logits: (batch * sequence_length, n_experts)
        _debug_log(
            "qwen3-prepare-start "
            f"layer={self.layer_id} hidden_shape={tuple(hidden_states.shape)} "
            f"hidden_dtype={hidden_states.dtype} hidden_device={hidden_states.device}"
        )
        router_logits = self.gate(hidden_states)
        _debug_sync(f"qwen3-gate layer={self.layer_id}", hidden_states.device)
        _debug_log(
            "qwen3-topk-start "
            f"layer={self.layer_id} logits_shape={tuple(router_logits.shape)} "
            f"logits_dtype={router_logits.dtype} logits_device={router_logits.device}"
        )

        router_mask, routing_weights_mask = self.lib.topk_softmax(router_logits)
        _debug_sync(f"qwen3-topk layer={self.layer_id}", router_logits.device)
        _debug_log(
            "qwen3-topk-finish "
            f"layer={self.layer_id} router_mask_shape={tuple(router_mask.shape)} "
            f"router_mask_dtype={router_mask.dtype} routing_weights_shape={tuple(routing_weights_mask.shape)} "
            f"routing_weights_dtype={routing_weights_mask.dtype}"
        )
        # routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # routing_weights, selected_experts = torch.topk(
        #     routing_weights, self.top_k, dim=-1
        # )
        # if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
        #     routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # # we cast back to the input dtype
        # routing_weights = routing_weights.to(hidden_states.dtype)

        # # print(f"hidden_states shape: {hidden_states.shape}")
        # # print(f"routing_weights shape: {routing_weights.shape}")

        # # Compute sparse mask via scatter
        # B, E = routing_weights.shape[0], self.num_experts
        # router_mask = torch.zeros(
        #     B, E, dtype=torch.bool, device=selected_experts.device
        # )
        # router_mask.scatter_(1, selected_experts, True)

        # routing_weights_mask = torch.zeros(
        #     B, E, dtype=routing_weights.dtype, device=routing_weights.device
        # )
        # routing_weights_mask.scatter_add_(1, selected_experts, routing_weights)
        # assert (routing_weights_mask_t == routing_weights_mask).all(), "routing_weights_mask_t and routing_weights_mask should be equal, max diff: {}".format(
        #     (routing_weights_mask_t - routing_weights_mask).abs().max()
        # )
        # assert (router_mask_t == router_mask).all(), "router_mask_t and router_mask should be equal, max diff: {}".format(
        #     (router_mask_t - router_mask).abs().max()
        # )
        return router_logits, router_mask, routing_weights_mask

    @nvtx.annotate("Qwen3MoEBlock", color="blue")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        _debug_log(
            "qwen3-block-start "
            f"layer={self.layer_id} batch={batch_size} seq={sequence_length} hidden_dim={hidden_dim} "
            f"dtype={hidden_states.dtype} device={hidden_states.device}"
        )
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, router_mask, routing_weights_mask = (
            self.__prepare_expert_route(hidden_states)
        )
        # router_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        # routing_weights_mask = (
        #     routing_weights[:, :, None] * router_mask
        # ).permute(0, 2, 1)
        # routing_weights_mask = torch.sum(routing_weights_mask, dim=-1)
        # router_mask = router_mask.permute(0, 2, 1)
        # router_mask = torch.any(router_mask, dim=-1)

        # print(f"router_mask shape: {router_mask.shape}")
        # print(f"routing_weights_mask shape: {routing_weights_mask.shape}")

        # # use logical or to merge last dimension
        # for i in range(self.top_k):
        #     router_mask[:, :, 0] = torch.logical_or(
        #         router_mask[:, :, 0], router_mask[:, :, i]
        #     )
        # router_mask = router_mask[:, :, 0]

        _debug_log(
            "qwen3-dispatch-start "
            f"layer={self.layer_id} flat_tokens={hidden_states.shape[0]} "
            f"router_mask_shape={tuple(router_mask.shape)} weights_shape={tuple(routing_weights_mask.shape)}"
        )
        self.expert_executor.dispatch_local(
            self.layer_id, hidden_states, router_mask, routing_weights_mask
        )
        _debug_log(f"qwen3-wait-start layer={self.layer_id}")
        final_hidden_states = self.expert_executor.wait_dispatch_local()
        _debug_sync(f"qwen3-dispatch layer={self.layer_id}", hidden_states.device)
        _debug_log(
            "qwen3-wait-finish "
            f"layer={self.layer_id} final_shape={tuple(final_hidden_states.shape)} "
            f"final_dtype={final_hidden_states.dtype} final_device={final_hidden_states.device}"
        )
        # final_hidden_states = torch.zeros(
        #     (batch_size * sequence_length, hidden_dim),
        #     dtype=hidden_states.dtype,
        #     device=hidden_states.device,
        # )

        # results = self.expert_executor.wait_dispatch_local()
        # for output, _, idx, _ in results:
        #     token_indices = router_mask[:, idx].bool()
        #     final_hidden_states[token_indices, :] += (
        #         output.to(routing_weights_mask.device)
        #         * routing_weights_mask[token_indices, idx][:, None]
        #     )

        final_hidden_states = final_hidden_states.view(
            batch_size, sequence_length, hidden_dim
        ).to(hidden_states.dtype)

        return final_hidden_states, router_logits
        """
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits
        """
