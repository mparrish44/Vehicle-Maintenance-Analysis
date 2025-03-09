# load_data.py
import pandas as pd
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_data(config):
    fleet_df = pd.read_csv(
        config["DATA_PATHS"]["FLEET_DATA"],
        encoding=config["ENCODING"],
        low_memory=False,
        dtype={"Equipment_Number": str},
    ).sample(n=config["SAMPLE_SIZES"]["FLEET"], random_state=config["RANDOM_STATE"])

    logistics_df = pd.read_csv(
        config["DATA_PATHS"]["LOGISTICS_DATA"],
        dtype={"Vehicle_ID": str},
    ).sample(n=config["SAMPLE_SIZES"]["LOGISTICS"], random_state=config["RANDOM_STATE"])

    return fleet_df, logistics_df


if __name__ == "__main__":
    config = load_config()
    fleet_df, logistics_df = load_data(config)
    print("Fleet Data Loaded:", fleet_df.head())
    print("Logistics Data Loaded:", logistics_df.head())
