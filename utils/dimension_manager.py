import dataclasses
from typing import List

@dataclasses.dataclass
class DimensionManager:
    """
    A stateless configuration object that is the single source of truth for all
    data and model dimensions. It derives all necessary dimension info from
    high-level configuration.
    """
    # --- Core Inputs (from config/args) ---
    mode: str
    target_features: List[str]
    all_features: List[str]
    loss_function: str
    quantiles: List[float] = dataclasses.field(default_factory=list)

    # --- Derived Properties ---
    def __post_init__(self):
        """Calculate derived properties after initialization."""
        self.covariate_features = [f for f in self.all_features if f not in self.target_features]
        self.num_targets = len(self.target_features)
        self.num_covariates = len(self.covariate_features)

        # --- Model Input/Output Dimensions ---
        if self.mode == 'S':
            self.enc_in = self.dec_in = self.c_out_evaluation = self.num_targets
        elif self.mode == 'MS':
            self.enc_in = self.num_targets + self.num_covariates; self.dec_in = self.num_targets
            self.c_out_evaluation = self.num_targets
        elif self.mode == 'M':
            self.enc_in = self.dec_in = self.num_targets + self.num_covariates
            self.c_out_evaluation = self.num_targets + self.num_covariates
        else:
            raise ValueError(f"Unknown feature mode: {self.mode}")

        # --- Adjust model's final layer dimension for the loss function ---
        self.c_out_model = self.c_out_evaluation
        if self.loss_function in ['quantile', 'pinball'] and self.quantiles:
            self.c_out_model *= len(self.quantiles)

    def __repr__(self):
        return (
            f"DimensionManager(Mode='{self.mode}', Loss='{self.loss_function}')\n"
            f"  - Features: {self.num_targets} targets, {self.num_covariates} covariates\n"
            f"  - Model Dims: In={self.enc_in}, Out(Layer)={self.c_out_model}, Out(Eval)={self.c_out_evaluation}"
        )