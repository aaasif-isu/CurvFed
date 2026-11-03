"""Single unified entry point for CurvFed experiments.

Usage:
    # Run using parameters in config.yaml:
    ./curvfed/bin/python run.py

    # Run using a custom YAML file:
    ./curvfed/bin/python run.py --config my_experiment.yaml
"""

from scripts.run_experiment import main

if __name__ == "__main__":
    main()
