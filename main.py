import numpy as np
import scipy.stats as stats
import scipy.special as special
import scipy.signal as signal
import random
from typing import Callable, List, Tuple, Union

class ParametricTimeSeriesGenerator:
    def __init__(self, 
                 base_functions: List[Callable],
                 param_ranges: List[Tuple[Tuple[float, float], ...]],
                 noise_type: str = 'normal',
                 noise_params: Tuple = (0, 0.1),
                 seed: Union[int, None] = None):
        """
        Initialize the generator.

        Args:
            base_functions: List of functions to superpose.
            param_ranges: List of parameter ranges for each function.
            noise_type: Type of noise ('normal', 'uniform', 'laplace', etc.).
            noise_params: Parameters for noise distribution.
            seed: Random seed for reproducibility.
        """
        self.base_functions = base_functions
        self.param_ranges = param_ranges
        self.noise_type = noise_type
        self.noise_params = noise_params
        self.random = np.random.RandomState(seed)

    def _sample_parameters(self, param_range: Tuple[Tuple[float, float], ...]) -> List[float]:
        return [self.random.uniform(low, high) for (low, high) in param_range]

    def _generate_noise(self, size: int) -> np.ndarray:
        if self.noise_type == 'normal':
            return self.random.normal(*self.noise_params, size=size)
        elif self.noise_type == 'uniform':
            return self.random.uniform(*self.noise_params, size=size)
        elif self.noise_type == 'laplace':
            return self.random.laplace(*self.noise_params, size=size)
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")

    def generate(self, t: np.ndarray) -> np.ndarray:
        """
        Generate the time series over a given time array.

        Args:
            t: Time array.

        Returns:
            np.ndarray: Time series values.
        """
        y = np.zeros_like(t)
        for func, param_range in zip(self.base_functions, self.param_ranges):
            params = self._sample_parameters(param_range)
            y += func(t, *params)

        noise = self._generate_noise(len(t))
        return y + noise

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define some base functions
    def sinusoid(t, amplitude, frequency, phase):
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)

    def gaussian(t, amplitude, mean, std_dev):
        return amplitude * np.exp(-((t - mean) ** 2) / (2 * std_dev ** 2))

    t = np.linspace(0, 10, 1000)

    generator = ParametricTimeSeriesGenerator(
        base_functions=[sinusoid, gaussian],
        param_ranges=[((0.8, 1.2), (0.9, 1.1), (0, np.pi)),  # for sinusoid
                      ((0.5, 1.5), (4, 6), (0.3, 0.6))],     # for gaussian
        noise_type='laplace',
        noise_params=(0, 0.05),
        seed=42
    )

    y = generator.generate(t)

    plt.plot(t, y)
    plt.title("Generated Parametric Time Series")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()
