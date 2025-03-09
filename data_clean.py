# data_clean.py
import pandas as pd
from load_data import load_data, load_config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def clean_and_save_data(config):
    """
    Loads, cleans, and saves data to a CSV file.

    Args:
        config (dict): Configuration dictionary.
    """
    logging.info("Starting data loading...")
    fleet_df, logistics_df = load_data(config)
    logging.info("Data loaded successfully.")

    logging.info("Starting data cleaning...")
    fleet_df.rename(columns={"Equipment_Number": "Vehicle_ID", "Model_Year": "Year_of_Manufacture"}, inplace=True)
    fleet_df["Year_of_Manufacture"] = pd.to_numeric(fleet_df["Year_of_Manufacture"], errors="coerce")
    logistics_df["Year_of_Manufacture"] = pd.to_numeric(logistics_df["Year_of_Manufacture"], errors="coerce")
    fleet_df.dropna(subset=["Year_of_Manufacture"], inplace=True)

    FLEET_REFERENCE_YEAR = 2025
    fleet_df["Vehicle_Age"] = FLEET_REFERENCE_YEAR - fleet_df["Year_of_Manufacture"]
    fleet_df["Acquisition_Mileage"] = pd.to_numeric(fleet_df["Acquisition_Mileage"], errors="coerce")
    fleet_df["Total_Miles"] = pd.to_numeric(fleet_df["Total_Miles"], errors="coerce")
    logistics_df["Last_Maintenance_Date"] = pd.to_datetime(logistics_df["Last_Maintenance_Date"], format="%m/%d/%y",
                                                           errors="coerce")

    fleet_df["Vehicle_ID"] = fleet_df["Vehicle_ID"].astype(str)
    logistics_df["Vehicle_ID"] = logistics_df["Vehicle_ID"].astype(str)
    merged_df = pd.merge(logistics_df, fleet_df, on="Vehicle_ID", how="left")
    merged_df.dropna(subset=["Last_Maintenance_Date"], inplace=True)

    merged_df["Days_Since_Last_Maintenance"] = (pd.Timestamp.today() - merged_df["Last_Maintenance_Date"]).dt.days
    merged_df.loc[:, "Days_Since_Last_Maintenance"] = merged_df["Days_Since_Last_Maintenance"].fillna(
        merged_df["Days_Since_Last_Maintenance"].median())

    merged_df.to_csv(config["OUTPUT_PATHS"]["CLEANED_DATA"], index=False)
    logging.info(f"Cleaned data saved to {config['OUTPUT_PATHS']['CLEANED_DATA']}.")


if __name__ == "__main__":
    config = load_config()
    clean_and_save_data(config)
