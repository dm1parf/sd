import os
import sys
cwd = os.getcwd()  # Linux fix
if cwd not in sys.path:
    sys.path.append(cwd)
from utils.config import ConfigManager


if __name__ == "__main__":
    config_path = os.path.join("scripts", "planner_config.ini")
    config = ConfigManager(config_path)
    planner = config.get_planner()
    start_experiment = config.get_start_experiment()

    if start_experiment:
        planner.run_experiment_series(start_experiment)
    else:
        planner.run_experiment_series()
