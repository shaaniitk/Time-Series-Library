import torch
from tools.unified_component_registry import unified_registry

# List
comps = unified_registry.list_all_components()
print('losses:', comps.get('loss', []))
print('outputs:', comps.get('output', []))

# Loss: MSE (standard)
MSE = unified_registry.get_component('loss', 'mse')
msel = MSE()
print('mse_value:', float(msel(torch.randn(2,4), torch.randn(2,4))))

# Output: Forecasting head (if available)
try:
	FH = unified_registry.get_component('output', 'forecasting_head')
	fh = FH(d_model=16, output_dim=2, horizon=3)
	hs = torch.randn(2,5,16)
	y = fh(hs)
	print('forecast_shape:', tuple(y.shape))
except Exception:
	print('forecasting_head not available')
