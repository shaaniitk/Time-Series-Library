import torch
from unified_component_registry import unified_registry
from utils.modular_components.implementations.losses import LossConfig
from utils.modular_components.implementations.outputs import OutputConfig

# List
comps = unified_registry.list_all_components()
print('losses:', comps.get('loss', []))
print('outputs:', comps.get('output', []))

# Loss: MSE
MSE = unified_registry.get_component('loss', 'mse_loss')
msel = MSE(LossConfig())
print('mse_value:', float(msel.compute_loss(torch.randn(2,4), torch.randn(2,4))))

# Output: Forecast
FH = unified_registry.get_component('output', 'forecasting_head')
fh = FH(OutputConfig(d_model=16, output_dim=2, horizon=3))
hs = torch.randn(2,5,16)
y = fh(hs)
print('forecast_shape:', tuple(y.shape))
