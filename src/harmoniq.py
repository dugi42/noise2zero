import numpy as np
import scipy.signal as signal
import pandas as pd
import logging
from typing import Callable, List, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class BaseFunctionFactory:
    """
    Factory class for generating base functions used in parametric time series generation.
    """
    @staticmethod
    def sinusoid(t: np.ndarray, amplitude: float, frequency: float, phase: float) -> np.ndarray:
        """Generates a sinusoidal wave."""
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)

    @staticmethod
    def gaussian(t: np.ndarray, amplitude: float, mean: float, std_dev: float) -> np.ndarray:
        """Generates a Gaussian curve."""
        return amplitude * np.exp(-((t - mean) ** 2) / (2 * std_dev ** 2))

    @staticmethod
    def sawtooth_wave(t: np.ndarray, amplitude: float, frequency: float, width: float) -> np.ndarray:
        """Generates a sawtooth wave."""
        return amplitude * signal.sawtooth(2 * np.pi * frequency * t, width)

    @staticmethod
    def exponential_decay(t: np.ndarray, amplitude: float, decay_rate: float) -> np.ndarray:
        """Generates an exponential decay curve."""
        return amplitude * np.exp(-decay_rate * t)

    @staticmethod
    def square_wave(t: np.ndarray, amplitude: float, frequency: float) -> np.ndarray:
        """Generates a square wave."""
        return amplitude * signal.square(2 * np.pi * frequency * t)

    @staticmethod
    def get_function(name: str) -> Callable:
        """
        Retrieves a base function by name.
        
        Args:
            name (str): Name of the base function.
        
        Returns:
            Callable: The corresponding function.
        
        Raises:
            ValueError: If the function name is not recognized.
        """
        functions = {
            'sinusoid': BaseFunctionFactory.sinusoid,
            'gaussian': BaseFunctionFactory.gaussian,
            'sawtooth_wave': BaseFunctionFactory.sawtooth_wave,
            'exponential_decay': BaseFunctionFactory.exponential_decay,
            'square_wave': BaseFunctionFactory.square_wave
        }
        if name not in functions:
            raise ValueError(f"Unknown base function: {name}")
        return functions[name]

    @staticmethod
    def load_functions_from_config(config: dict) -> List[Callable]:
        """
        Loads base functions from a configuration dictionary.
        
        Args:
            config (dict): Configuration dictionary containing base function names.
        
        Returns:
            List[Callable]: List of base functions.
        """
        base_function_names = config['base_functions']
        return [BaseFunctionFactory.get_function(name) for name in base_function_names]

class ParametricTimeSeriesGenerator:
    """
    Class for generating parametric time series with noise and variations.
    """
    def __init__(self, 
                 base_functions: List[Callable],
                 param_ranges: List[Tuple[Tuple[float, float], ...]],
                 noise_type: str = 'normal',
                 noise_params: Tuple = (0, 0.1),
                 seed: Union[int, None] = None):
        """
        Initializes the generator.
        
        Args:
            base_functions (List[Callable]): List of base functions.
            param_ranges (List[Tuple[Tuple[float, float], ...]]): Parameter ranges for each function.
            noise_type (str): Type of noise ('normal', 'uniform', 'laplace').
            noise_params (Tuple): Parameters for the noise distribution.
            seed (Union[int, None]): Random seed for reproducibility.
        """
        self.base_functions = base_functions
        self.param_ranges = param_ranges
        self.noise_type = noise_type
        self.noise_params = noise_params
        self.random = np.random.RandomState(seed)
        logging.info(f"Initialized generator with {len(base_functions)} base functions.")

    def _sample_parameters(self, param_range: Tuple[Tuple[float, float], ...]) -> List[float]:
        """
        Samples parameters from the given range.
        
        Args:
            param_range (Tuple[Tuple[float, float], ...]): Parameter range.
        
        Returns:
            List[float]: Sampled parameters.
        """
        params = [self.random.uniform(low, high) for (low, high) in param_range]
        logging.debug(f"Sampled parameters: {params}")
        return params

    def _perturb_parameters(self, params: List[float], variation_scale: float = 0.05) -> List[float]:
        """
        Perturbs parameters by adding random noise.
        
        Args:
            params (List[float]): Original parameters.
            variation_scale (float): Scale of the variation.
        
        Returns:
            List[float]: Perturbed parameters.
        """
        perturbed = [param + self.random.normal(0, variation_scale * abs(param)) for param in params]
        logging.debug(f"Perturbed parameters: {perturbed}")
        return perturbed

    def _generate_noise(self, size: int) -> np.ndarray:
        """
        Generates noise of the specified type and size.
        
        Args:
            size (int): Number of noise samples.
        
        Returns:
            np.ndarray: Generated noise.
        
        Raises:
            ValueError: If the noise type is unsupported.
        """
        logging.info(f"Generating {size} noise samples with type {self.noise_type} and params {self.noise_params}.")
        if self.noise_type == 'normal':
            return self.random.normal(*self.noise_params, size=size)
        elif self.noise_type == 'uniform':
            return self.random.uniform(*self.noise_params, size=size)
        elif self.noise_type == 'laplace':
            return self.random.laplace(*self.noise_params, size=size)
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")

    def generate(self, t_start: float, t_end: float, t_points: int, repetitions: int = 5, variation_scale: float = 0.05, to_utc: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a parametric time series.
        
        Args:
            t_start (float): Start time.
            t_end (float): End time.
            t_points (int): Number of time points per repetition.
            repetitions (int): Number of repetitions.
            variation_scale (float): Scale of parameter variation.
            to_utc (bool): Whether to convert time to UTC.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Time and value arrays.
        """
        logging.info(f"Generating time series from {t_start} to {t_end} with {t_points} points per motive, {repetitions} repetitions.")
        t = np.linspace(t_start, t_end, t_points)
        y_full = []

        initial_params = [self._sample_parameters(param_range) for param_range in self.param_ranges]

        for rep in range(repetitions):
            logging.info(f"Generating repetition {rep+1}/{repetitions}")
            y = np.zeros_like(t)
            for idx, (func, params) in enumerate(zip(self.base_functions, initial_params)):
                perturbed_params = self._perturb_parameters(params, variation_scale)
                contribution = func(t, *perturbed_params)
                y += contribution
                logging.debug(f"Added perturbed contribution from function {idx}.")
            noise = self._generate_noise(len(t))
            y += noise
            y_full.append(y)

        y_concat = np.concatenate(y_full)
        t_full = np.linspace(t_start, t_end * repetitions, t_points * repetitions)

        if to_utc:
            start_time = pd.Timestamp.utcnow()
            time_deltas = pd.to_timedelta(t_full, unit='s')
            t_full = start_time + time_deltas

        return t_full, y_concat
