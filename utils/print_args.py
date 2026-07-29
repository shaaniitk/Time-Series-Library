def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    print(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    print(f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    print(f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    print()

    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        print("\033[1m" + "Forecasting Task" + "\033[0m")
        print(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Label Len:":<20}{args.label_len:<20}')
        print(f'  {"Pred Len:":<20}{args.pred_len:<20}{"Seasonal Patterns:":<20}{args.seasonal_patterns:<20}')
        print(f'  {"Inverse:":<20}{args.inverse:<20}')
        print()

    if args.task_name == 'imputation':
        print("\033[1m" + "Imputation Task" + "\033[0m")
        print(f'  {"Mask Rate:":<20}{args.mask_rate:<20}')
        print()

    if args.task_name == 'anomaly_detection':
        print("\033[1m" + "Anomaly Detection Task" + "\033[0m")
        print(f'  {"Anomaly Ratio:":<20}{args.anomaly_ratio:<20}')
        print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"Top k:":<20}{args.top_k:<20}{"Num Kernels:":<20}{args.num_kernels:<20}')
    print(f'  {"Enc In:":<20}{args.enc_in:<20}{"Dec In:":<20}{args.dec_in:<20}')
    print(f'  {"C Out:":<20}{args.c_out:<20}{"d model:":<20}{args.d_model:<20}')
    print(f'  {"n heads:":<20}{args.n_heads:<20}{"e layers:":<20}{args.e_layers:<20}')
    d_ff_value = getattr(args, "d_ff", "-")
    if getattr(args, "model", None) == "TemporalFusionTransformer":
        d_ff_value = f"{d_ff_value} (ignored)"
    print(f'  {"d layers:":<20}{args.d_layers:<20}{"d FF:":<20}{str(d_ff_value):<20}')
    print(f'  {"Moving Avg:":<20}{args.moving_avg:<20}{"Factor:":<20}{args.factor:<20}')
    print(f'  {"Distil:":<20}{args.distil:<20}{"Dropout:":<20}{args.dropout:<20}')
    print(f'  {"Embed:":<20}{args.embed:<20}{"Activation:":<20}{args.activation:<20}')
    print()

    if hasattr(args, 'tft_use_lag_attention'):
        print("\033[1m" + "TFT Upgrades" + "\033[0m")
        print(f'  {"TFT Profile:":<20}{getattr(args, "tft_profile", "extended_safe"):<20}{"TFT Digest:":<20}{str(getattr(args, "tft_config_digest", "-")):<20}')
        print(f'  {"TFT SwiGLU:":<20}{args.tft_use_swiglu!s:<20}{"TFT Full Attn:":<20}{args.tft_full_attention!s:<20}')
        print(f'  {"TFT Dual Attn:":<20}{args.tft_dual_attention_fusion!s:<20}{"TFT Cross Mix:":<20}{args.tft_cross_variable_mixing!s:<20}')
        print(f'  {"VSN Bypass:":<20}{args.tft_vsn_residual_bypass!s:<20}{"Custom Known:":<20}{args.tft_allow_custom_known!s:<20}')
        print(f'  {"Explicit XAttn:":<20}{args.tft_use_explicit_cross_attention!s:<20}{"XAttn Type:":<20}{args.tft_cross_attention_type:<20}')
        print(f'  {"Pos Bias:":<20}{args.tft_attention_position_bias:<20}{"Attn Backend:":<20}{args.tft_attention_backend:<20}')
        print(f'  {"RoPE Base:":<20}{args.tft_rope_base:<20}{"ALiBi Scale:":<20}{args.tft_alibi_scale:<20}')
        print(f'  {"Use RevIN:":<20}{args.tft_use_revin!s:<20}{"Quantile Head:":<20}{args.tft_use_quantile_head!s:<20}')
        print(f'  {"Temporal BB:":<20}{args.tft_temporal_backbone:<20}{"BB Layers:":<20}{args.tft_temporal_backbone_layers:<20}')
        print(f'  {"BB Layer Scope:":<20}{str(getattr(args, "tft_temporal_backbone_layers_scope", "-")):<20}')
        print(f'  {"BB Kernel:":<20}{args.tft_temporal_kernel_size:<20}{"BB Hidden:":<20}{str(args.tft_temporal_hidden_size):<20}')
        print(f'  {"Lag Attention:":<20}{args.tft_use_lag_attention!s:<20}{"Lag Scales:":<20}{str(args.tft_lag_scales):<20}')
        print(f'  {"Higher Order:":<20}{args.tft_use_higher_order!s:<20}{"Interaction Ord:":<20}{args.tft_interaction_order:<20}')
        print(f'  {"Interaction Rank:":<20}{str(args.tft_interaction_rank):<20}{"Regime MoE:":<20}{args.tft_use_regime_moe!s:<20}')
        print(f'  {"MoE Experts:":<20}{args.tft_num_moe_experts:<20}{"MoE Top-k:":<20}{args.tft_moe_top_k:<20}')
        print(f'  {"MoE Regimes:":<20}{args.tft_num_regimes:<20}{"MoE Hidden:":<20}{str(args.tft_moe_hidden_size):<20}')
        print(f'  {"MoE Aux Coeff:":<20}{args.tft_moe_aux_loss_coeff:<20}{"Payload Stack:":<20}{args.tft_payload_stack_layers!s:<20}')
        print(f'  {"RevIN Affine:":<20}{args.tft_revin_affine!s:<20}{"Quantiles:":<20}{str(args.tft_output_quantiles):<20}')
        fft_branch = getattr(args, 'tft_use_fft_branch', False)
        if fft_branch:
            print(f'  {"FFT Branch:":<20}{fft_branch!s:<20}{"FFT Modes:":<20}{getattr(args, "tft_fft_modes", 32):<20}')
            print(f'  {"FFT Select:":<20}{getattr(args, "tft_fft_mode_select", "low"):<20}')
        sd_rate = getattr(args, 'tft_stochastic_depth_rate', 0.0)
        gc = getattr(args, 'tft_gradient_checkpointing', False)
        if sd_rate > 0.0 or gc:
            print(f'  {"Stoch Depth:":<20}{sd_rate:<20}{"Grad Ckpt:":<20}{gc!s:<20}')
        tc = getattr(args, 'tft_use_temporal_compression', False)
        if tc:
            print(f'  {"Temporal Compress:":<20}{tc!s:<20}{"TC Stride:":<20}{getattr(args, "tft_tc_stride", 2):<20}')
            print(f'  {"TC Threshold:":<20}{getattr(args, "tft_tc_threshold", 256):<20}')
        vsn_sh = getattr(args, 'tft_vsn_n_selection_heads', 1)
        if vsn_sh > 1:
            print(f'  {"VSN Sel Heads:":<20}{vsn_sh:<20}')
        mlp_qp = getattr(args, 'tft_mlp_quantile_projection', False)
        if mlp_qp:
            print(f'  {"MLP Quantile Proj:":<20}{mlp_qp!s:<20}{"Q Proj FF Size:":<20}{getattr(args, "tft_quantile_projection_ff_size", 0):<20}')
        print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    print(f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{args.loss:<20}')
    print(f'  {"Lradj:":<20}{args.lradj:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    print()

    print("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
    print(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{args.p_hidden_layers:<20}') 
    print()
