from utils.planner import ExperimentPlanner


if __name__ == "__main__":
    planner = ExperimentPlanner("scripts/planner_config.ini",
                                "scripts/experiment_config.ini",
                                "scripts.experiment_pipeline",
                                "statistics",
                                "..index.txt"
                                )

    planner.run_experiment_series()
