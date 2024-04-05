import os
from utils.config import ConfigManager


if __name__ == "__main__":
    config_path = os.path.join("scripts", "planner_config.ini")
    config = ConfigManager(config_path)
    planner = config.get_planner()

    planner.run_experiment_series()
