from .base import BaseDecomposition
from .moving_average import MovingAverageDecomposition
from ..config_schemas import ComponentConfig

class SeriesDecomposition(BaseDecomposition):
    """Series decomposition used in Autoformer"""
    def __init__(self, config: ComponentConfig):
        super().__init__(config)
        self.kernel_size = config.custom_params.get('kernel_size', 25)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.moving_avg = MovingAverageDecomposition(config)
    def forward(self, x):
        return self.moving_avg(x)
