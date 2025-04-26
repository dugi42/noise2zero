import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src.harmoniq import BaseFunctionFactory, ParametricTimeSeriesGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate parametric time series from config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file.")
    parser.add_argument("--output", type=str, required=False, help="Path to output Parquet file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    base_functions = BaseFunctionFactory.load_functions_from_config(config)

    param_ranges = config['param_ranges']
    noise_type = config['noise_type']
    noise_params = tuple(config['noise_params'])
    seed = config.get('seed', None)
    repetitions = config['repetitions']
    variation_scale = config['variation_scale']
    to_utc = config.get('to_utc', False)

    t_start = config['time']['start']
    t_end = config['time']['end']
    t_points = config['time']['points']

    generator = ParametricTimeSeriesGenerator(
        base_functions=base_functions,
        param_ranges=param_ranges,
        noise_type=noise_type,
        noise_params=noise_params,
        seed=seed
    )

    t_full, y = generator.generate(t_start=t_start, t_end=t_end, t_points=t_points, repetitions=repetitions, variation_scale=variation_scale, to_utc=to_utc)

    df = pd.DataFrame({'time': t_full, 'value': y})

    if args.output:
        print(f"Saving generated time series to {args.output}")
        df.to_parquet(args.output, index=False)

    plt.plot(t_full, y)
    plt.title("Generated Parametric Time Series with Config")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()
