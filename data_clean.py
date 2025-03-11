# data_clean.py
'''
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
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from load_data import load_data, load_config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def clean_and_save_data(config):
    """
    Loads, cleans, saves the data to a CSV file, and generates basic visualizations.

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

    # Save the cleaned data
    merged_df.to_csv(config["OUTPUT_PATHS"]["CLEANED_DATA"], index=False)
    logging.info(f"Cleaned data saved to {config['OUTPUT_PATHS']['CLEANED_DATA']}.")

    # Create visualizations
    create_visualizations(fleet_df, merged_df, config)
    create_eda_visualizations(fleet_df, merged_df, config)


def create_visualizations(fleet_df, merged_df, config):
    """
    Creates and saves the initial visualizations based on fleet and merged data.

    Args:
        fleet_df (pd.DataFrame): Dataframe containing fleet data.
        merged_df (pd.DataFrame): Dataframe containing merged fleet and logistics data.
        config (dict): Configuration dictionary.
    """
    # Define directory to save visualizations
    visualization_dir = Path(config["OUTPUT_PATHS"].get("VIZ_DIR", "./visualizations"))
    visualization_dir.mkdir(parents=True, exist_ok=True)

    # 1. Distribution of Vehicle Ages
    plt.figure(figsize=(10, 6))
    sns.histplot(data=fleet_df, x="Vehicle_Age", bins=20, kde=True, color="blue")
    plt.title("Distribution of Vehicle Ages")
    plt.xlabel("Vehicle Age (years)")
    plt.ylabel("Count")
    vehicle_age_plot_path = visualization_dir / "vehicle_age_distribution.png"
    plt.savefig(vehicle_age_plot_path)
    plt.close()
    logging.info(f"Vehicle Age Distribution plot saved to {vehicle_age_plot_path}.")

    # 2. Days Since Last Maintenance Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=merged_df, x="Days_Since_Last_Maintenance", bins=30, kde=True, color="green")
    plt.title("Distribution of Days Since Last Maintenance")
    plt.xlabel("Days Since Last Maintenance")
    plt.ylabel("Count")
    maintenance_days_plot_path = visualization_dir / "days_since_maintenance_distribution.png"
    plt.savefig(maintenance_days_plot_path)
    plt.close()
    logging.info(f"Days Since Last Maintenance plot saved to {maintenance_days_plot_path}.")


def create_eda_visualizations(fleet_df, merged_df, config):
    """
    Creates and saves EDA visualizations for the datasets.

    Args:
        fleet_df (pd.DataFrame): Cleaned fleet dataframe.
        merged_df (pd.DataFrame): Merged dataframe with logistic and fleet info.
        config (dict): Configuration dictionary.
    """
    # Define the directory to save visualizations
    visualization_dir = Path(config["OUTPUT_PATHS"].get("VIZ_DIR", "./visualizations"))
    visualization_dir.mkdir(parents=True, exist_ok=True)

    # 1. Correlation Heatmap for Numerical Features
    plt.figure(figsize=(12, 8))
    numeric_columns = ['Vehicle_Age', 'Acquisition_Mileage', 'Total_Miles', 'Days_Since_Last_Maintenance']
    correlation_matrix = merged_df[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numerical Features")
    correlation_plot_path = visualization_dir / "correlation_heatmap.png"
    plt.savefig(correlation_plot_path)
    plt.close()
    logging.info(f"Correlation heatmap saved to {correlation_plot_path}.")

    # 2. Acquisition Mileage vs. Total Miles Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=fleet_df, x="Acquisition_Mileage", y="Total_Miles", color="purple", alpha=0.6)
    plt.title("Acquisition Mileage vs. Total Miles")
    plt.xlabel("Acquisition Mileage")
    plt.ylabel("Total Miles")
    scatter_plot_path = visualization_dir / "acquisition_vs_total_miles.png"
    plt.savefig(scatter_plot_path)
    plt.close()
    logging.info(f"Scatter plot of Acquisition Mileage vs. Total Miles saved to {scatter_plot_path}.")

    # 3. Vehicle Age Distribution by Maintenance Status (Grouped Histogram)
    plt.figure(figsize=(10, 6))
    if "Maintenance_Required" in merged_df.columns:
        sns.histplot(data=merged_df, x="Vehicle_Age", hue="Maintenance_Required", bins=15, kde=True, palette="Set2")
        plt.title("Vehicle Age Distribution by Maintenance Status")
        plt.xlabel("Vehicle Age (years)")
        plt.ylabel("Count")
        maintenance_age_plot_path = visualization_dir / "vehicle_age_by_maintenance_status.png"
        plt.savefig(maintenance_age_plot_path)
        plt.close()
        logging.info(f"Vehicle Age by Maintenance Status plot saved to {maintenance_age_plot_path}.")
    else:
        logging.warning("Maintenance_Required column not found, skipping Vehicle Age by Maintenance Status plot.")

    # 4. Maintenance Count Over Time
    plt.figure(figsize=(12, 6))
    if "Last_Maintenance_Date" in merged_df.columns:
        merged_df["Last_Maintenance_YearMonth"] = merged_df["Last_Maintenance_Date"].dt.to_period("M")
        maintenance_counts = merged_df["Last_Maintenance_YearMonth"].value_counts().sort_index()
        maintenance_counts.plot(kind="line", color="red")
        plt.title("Maintenance Count Over Time (Monthly)")
        plt.xlabel("Year-Month")
        plt.ylabel("Count of Maintenance Records")
        time_series_plot_path = visualization_dir / "maintenance_count_over_time.png"
        plt.savefig(time_series_plot_path)
        plt.close()
        logging.info(f"Maintenance count over time plot saved to {time_series_plot_path}.")
    else:
        logging.warning("Last_Maintenance_Date column not found, skipping Maintenance Count Over Time plot.")


if __name__ == "__main__":
    config = load_config()
    clean_and_save_data(config)
