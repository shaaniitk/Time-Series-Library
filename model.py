# File: 2_model.py

import torch
import torch.nn as nn
from components import PGAT_CrossAttn_Layer, TemporalAttention, StandardDecoder, ProbabilisticDecoder

class InitialEmbedding(nn.Module):
    """Handles embedding of all initial features (x and t) for all node types."""
    def __init__(self, config):
        super().__init__()
        d_model = config.d_model
        self.wave_x_embed = nn.Linear(config.wave_feature_dim, d_model)
        self.target_x_embed = nn.Linear(config.target_feature_dim, d_model)
        self.wave_t_embed = nn.Linear(config.topo_feature_dim, d_model)
        self.target_t_embed = nn.Linear(config.topo_feature_dim, d_model)
        self.transition_x_embed = nn.Embedding(config.num_transitions, d_model)
        self.transition_t_embed = nn.Embedding(config.num_transitions, d_model)

    def forward(self, wave_x, target_x, graph):
        x_dict = {
            'wave': self.wave_x_embed(wave_x),
            'target': self.target_x_embed(target_x),
            'transition': self.transition_x_embed.weight.unsqueeze(0).expand(wave_x.size(0), -1, -1)
        }
        t_dict = {
            'wave': self.wave_t_embed(graph.wave.t).unsqueeze(0).expand(wave_x.size(0), -1, -1),
            'target': self.target_t_embed(graph.target.t).unsqueeze(0).expand(wave_x.size(0), -1, -1),
            'transition': self.transition_t_embed.weight.unsqueeze(0).expand(wave_x.size(0), -1, -1)
        }
        return x_dict, t_dict

class SOTA_Temporal_PGAT(nn.Module):
    def __init__(self, config, mode='bayesian'):
        super().__init__()
        self.mode = mode
        
        self.embedding = InitialEmbedding(config)
        self.spatial_encoder = PGAT_CrossAttn_Layer(config.d_model)
        self.temporal_encoder = TemporalAttention(config.d_model, config.n_heads, config.dropout)
        
        if self.mode == 'standard':
            self.decoder = StandardDecoder(config.d_model)
        else:
            self.decoder = ProbabilisticDecoder(config.d_model)

    def forward(self, wave_window, target_window, graph):
        batch_size, seq_len, num_targets, _ = target_window.shape
        
        historical_event_messages = []
        for t in range(seq_len):
            x_dict_t, t_dict_t = self.embedding(wave_window[:, t], target_window[:, t], graph)
            x_dict_out, _ = self.spatial_encoder(x_dict_t, t_dict_t, graph.edge_index_dict)
            historical_event_messages.append(x_dict_out['transition'].mean(dim=1)) # Avg transitions to get a single event vector
        
        historical_events = torch.stack(historical_event_messages, dim=1)
        
        final_target_state = x_dict_out['target']
        query = final_target_state.view(batch_size * num_targets, 1, -1)
        
        history = historical_events.unsqueeze(1).expand(-1, num_targets, -1, -1).reshape(batch_size * num_targets, seq_len, -1)
        
        temporal_context, _ = self.temporal_encoder(query, history)
        final_embedding = temporal_context.view(batch_size, num_targets, -1)

        return self.decoder(final_embedding)