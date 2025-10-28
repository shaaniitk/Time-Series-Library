# models/celestial_modules/postprocessing.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import CelestialPGATConfig

class PostProcessingModule(nn.Module):
    def __init__(self, config: CelestialPGATConfig):
        super().__init__()
        self.config = config

        if config.enable_adaptive_topk and (config.d_model % config.num_graph_nodes) == 0:
            node_dim = config.d_model // config.num_graph_nodes
            self.node_score = nn.Sequential(
                nn.Linear(node_dim, max(1, node_dim // 2)), nn.GELU(),
                nn.Linear(max(1, node_dim // 2), 1)
            )
            self.topk_projection = nn.Linear(config.adaptive_topk_k * node_dim, config.d_model)
        else:
            self.node_score, self.topk_projection = None, None

        if config.use_stochastic_control:
            self.register_buffer("_stoch_step", torch.tensor(0, dtype=torch.long), persistent=True)

    def forward(self, graph_features, global_step=None):
        # 1. Adaptive TopK Pooling
        if self.config.enable_adaptive_topk and self.node_score is not None:
            bsz, seqlen, dmodel = graph_features.shape
            node_dim = dmodel // self.config.num_graph_nodes
            per_node = graph_features.view(bsz, seqlen, self.config.num_graph_nodes, node_dim)
            scores = self.node_score(per_node).squeeze(-1)
            
            # Differentiable soft selection
            attention_weights = F.softmax(scores / self.config.adaptive_topk_temperature, dim=-1)
            topk_attention, topk_idx = torch.topk(attention_weights, k=self.config.adaptive_topk_k, dim=2)
            idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, node_dim)
            topk_nodes = torch.gather(per_node, dim=2, index=idx_expanded)
            
            pooled = topk_nodes.reshape(bsz, seqlen, self.config.adaptive_topk_k * node_dim)
            graph_features = self.topk_projection(pooled)

        # 2. Stochastic Control
        if self.config.use_stochastic_control and self.training:
            step = global_step if global_step is not None else self._stoch_step.item()
            progress = min(1.0, step / self.config.stochastic_decay_steps)
            temp = (1.0 - progress) * self.config.stochastic_temperature_start + progress * self.config.stochastic_temperature_end
            noise = torch.randn_like(graph_features) * (self.config.stochastic_noise_std * temp)
            graph_features = graph_features + noise
            if global_step is None:
                self._stoch_step += 1

        return graph_features