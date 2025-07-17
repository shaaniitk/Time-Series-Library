
from .series_decomposition import MovingAverage
from .base import BaseDecomposition

class StableSeriesDecomposition(BaseDecomposition):
    """
    Series decomposition with a stability fix to ensure the kernel size is always odd.
    """
    def __init__(self, kernel_size):
        super(StableSeriesDecomposition, self).__init__()
        # Fix: Ensure odd kernel size for stability
        stable_kernel_size = kernel_size + (1 - kernel_size % 2)
        self.moving_avg = MovingAverage(stable_kernel_size, stride=1)

    def forward(self, x):
        """
        Decomposes the input time series into seasonal and trend components.
        """
        moving_mean = self.moving_avg(x)
        seasonal = x - moving_mean
        return seasonal, moving_mean
