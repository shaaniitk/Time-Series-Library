# Migration Plan to Modular Folder

## Overview
This document outlines the step-by-step plan to migrate components from the `layers/` folder to `layers/modular/`, addressing duplicates by selecting the most robust implementations and planning integration for components not yet in modular/. This includes re-evaluation of utils/modular_components/ for additional duplicates and new implementations.

## Step 1: List All Components and Subcomponents
- **Attention**: MultiHeadAttention, AutoCorrelation, FourierAttention, SparseAttention, OptimizedAutoCorrelationAttention, AdaptiveAutoCorrelationAttention, EnhancedAutoCorrelationLayer, MemoryEfficientAttention, etc.
- **Decomposition**: MovingAverage, LearnableDecomposition, WaveletDecomposition.
- **Encoder**: StandardEncoder, EnhancedEncoder, HierarchicalEncoder.
- **Decoder**: StandardDecoder, EnhancedDecoder.
- **Fusion**: HierarchicalFusion.
- **Layers**: Various embedding, normalization, conv blocks.
- **Losses**: MSE, MAE, BayesianLoss, QuantileLoss, CrossEntropy, Huber, NegativeLogLikelihood.
- **Sampling**: Deterministic, Bayesian, MonteCarlo.
- **Output Heads**: StandardHead, QuantileHead.

## Step 2: Identify Duplicates and Robust Implementations
- **Autocorrelation**: Duplicates in AutoCorrelation.py, EnhancedAutoCorrelation.py, OptimizedAutoCorrelationAttention, AdaptiveAutoCorrelationAttention, EnhancedAutoCorrelationLayer. Robust: EnhancedAutoCorrelationLayer (complete with projections, adaptive features).
- **Normalization/Embedding**: Duplicates in Embed.py, Normalization.py. Robust: Normalization.py (flexible).
- **Attention Mechanisms**: Duplicates in Attention.py, CustomMultiHeadAttention.py, MultiHeadAttention, SparseAttention, MemoryEfficientAttention. Robust: MemoryEfficientAttention (optimized with checkpointing).
- **Conv and FFN**: Duplicates in Conv_Blocks.py, GatedMoEFFN.py. Robust: GatedMoEFFN.py (general).
- **Wavelet Decomposition**: Duplicates in DWT_Decomposition.py, MultiWaveletCorrelation.py. Robust: MultiWaveletCorrelation.py (stable).
- **Losses**: New implementations like MSELoss, MAELoss, CrossEntropyLoss, HuberLoss, NegativeLogLikelihood may duplicate existing; Robust: HuberLoss (robust to outliers).
- Replace modular/ versions if these are better, prioritizing optimized and adaptive versions from utils/.

## Step 3: List Non-Modular Components and Migration Plan
- **Embed.py**: Migrate to modular/layers/embed.py with registry.
- **Normalization.py**: Integrate into modular/normalization/.
- **Various EncDec.py (e.g., Transformer_EncDec.py)**: Abstract and move to modular/encoder/ and decoder/.
- **BayesianLayers.py**: Add to modular/layers/bayesian/.
- **Conv_Blocks.py**: Migrate to modular/layers/conv/.
- **New from utils/modular_components/**: attentions.py (MultiHeadAttention, AutoCorrelationAttention, SparseAttention), losses.py (MSELoss, MAELoss, etc.), advanced_attentions.py (OptimizedAutoCorrelationAttention, etc.). Migrate to modular/attention/ and modular/losses/ with registry integration.
- For each: Create base class, register, update models to use modular versions.

## Step 4: Migration Steps
1. Backup existing layers/ and utils/.
2. For duplicates, copy robust version to modular/, delete others.
3. For non-modular and new utils/ components, create subfolders in modular/, implement with registry.
4. Update imports in models/ and tests/.
5. Run tests to validate.
6. Deprecate old files.

## Step 5: Post-Migration
- Update documentation.
- Run full test suite.
- Monitor for issues.