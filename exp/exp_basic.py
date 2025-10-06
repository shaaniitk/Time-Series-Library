import os
import torch
import models as models_pkg

# GCLI modular model
from models.modular_autoformer import ModularAutoformer

# Import HF models and factory (optional; may not always be present)
try:
    from models.HFEnhancedAutoformer import HFEnhancedAutoformer  # type: ignore
    from models.HFAdvancedFactory import (  # type: ignore
        HFBayesianEnhancedAutoformer,
        HFHierarchicalEnhancedAutoformer,
        HFQuantileEnhancedAutoformer,
        HFFullEnhancedAutoformer,
        create_hf_model_from_config,
    )
except Exception:  # pragma: no cover - optional dependency path
    HFEnhancedAutoformer = None  # type: ignore
    HFBayesianEnhancedAutoformer = None  # type: ignore
    HFHierarchicalEnhancedAutoformer = None  # type: ignore
    HFQuantileEnhancedAutoformer = None  # type: ignore
    HFFullEnhancedAutoformer = None  # type: ignore
    create_hf_model_from_config = None  # type: ignore


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        # Build model registry lazily to avoid import-time failures for optional models
        self.model_dict = {}

        # Always register the modular entry point explicitly
        self.model_dict['ModularAutoformer'] = ModularAutoformer

        # Optionally register HF models if available
        if HFEnhancedAutoformer is not None:
            self.model_dict['HFEnhancedAutoformer'] = HFEnhancedAutoformer
        if HFBayesianEnhancedAutoformer is not None:
            self.model_dict['HFBayesianEnhancedAutoformer'] = HFBayesianEnhancedAutoformer
        if HFHierarchicalEnhancedAutoformer is not None:
            self.model_dict['HFHierarchicalEnhancedAutoformer'] = HFHierarchicalEnhancedAutoformer
        if HFQuantileEnhancedAutoformer is not None:
            self.model_dict['HFQuantileEnhancedAutoformer'] = HFQuantileEnhancedAutoformer
        if HFFullEnhancedAutoformer is not None:
            self.model_dict['HFFullEnhancedAutoformer'] = HFFullEnhancedAutoformer

        # Try to register standard models via models package lazy access
        optional_model_names = [
            'TimesNet', 'Autoformer', 'EnhancedAutoformer', 'BayesianEnhancedAutoformer',
            'HierarchicalEnhancedAutoformer', 'Transformer', 'Nonstationary_Transformer', 'DLinear',
            'FEDformer', 'Informer', 'LightTS', 'Reformer', 'ETSformer', 'Pyraformer', 'PatchTST', 'MICN',
            'SOTA_Temporal_PGAT', 'Enhanced_SOTA_PGAT',
            'Crossformer', 'FiLM', 'iTransformer', 'Koopa', 'TiDE', 'FreTS', 'MambaSimple', 'TimeMixer',
            'TSMixer', 'SegRNN', 'TemporalFusionTransformer', 'SCINet', 'PAttn', 'TimeXer', 'WPMixer',
            'MultiPatchFormer'
        ]
        for name in optional_model_names:
            try:
                cls = getattr(models_pkg, name)
                self.model_dict[name] = cls
            except Exception:
                # Skip models that are unavailable in this environment
                pass
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
