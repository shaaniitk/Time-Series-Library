from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelation, AdaptiveAutoCorrelationLayer

def get_attention_layer(configs):
    attention_type = getattr(configs, 'attention_type', 'autocorrelation')
    
    if attention_type == 'adaptive_autocorrelation':
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
        # Default to standard AutoCorrelation
        from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
        return AutoCorrelationLayer(
            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
            configs.d_model, configs.n_heads
        )
