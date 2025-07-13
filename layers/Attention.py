from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelation, AdaptiveAutoCorrelationLayer

def get_attention_layer(configs):
    if configs.attention_type == 'adaptive_autocorrelation':
        return AdaptiveAutoCorrelationLayer(
            AdaptiveAutoCorrelation(
                False, configs.factor, 
                attention_dropout=configs.dropout,
                output_attention=False,
                adaptive_k=True,
                multi_scale=True,
                scales=[1, 2, 4]
            ),
            configs.d_model, 
            configs.n_heads
        )
    else:
        raise ValueError(f'Unknown attention_type: {configs.attention_type}')
