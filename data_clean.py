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
    vehicle_age_plot_path = visualization_dir / "Vehicle_Age_distribution.png"
    plt.savefig(vehicle_age_plot_path)
    plt.close()
    logging.info(f"Vehicle Age Distribution plot saved to {vehicle_age_plot_path}.")

    # 2. Days Since Last Maintenance Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=merged_df, x="Days_Since_Last_Maintenance", bins=30, kde=True, color="green")
    plt.title("Distribution of Days Since Last Maintenance")
    plt.xlabel("Days Since Last Maintenance")
    plt.ylabel("Count")
    maintenance_days_plot_path = visualization_dir / "Days_Since_Maintenance_distribution.png"
    plt.savefig(maintenance_days_plot_path)
    plt.close()
    logging.info(f"Days Since Last Maintenance plot saved to {maintenance_days_plot_path}.")

    # 3. Bar Plot of Missing Values
    plt.figure(figsize=(12, 8))

    # Calculate missing values for fleet_df and merged_df
    missing_values_fleet = fleet_df.isnull().sum()
    missing_values_merged = merged_df.isnull().sum()

    # Combine into a single DataFrame for visualization
    missing_values = pd.DataFrame({
        "Fleet Data": missing_values_fleet,
        "Merged Data": missing_values_merged
    }).stack().reset_index()
    missing_values.columns = ["Feature", "Dataset", "Missing Values"]
    missing_values = missing_values[missing_values["Missing Values"] > 0].sort_values(by="Missing Values",
                                                                                      ascending=False)

    # Plot the missing values bar plot using seaborn
    sns.barplot(x="Missing Values", y="Feature", hue="Dataset", data=missing_values, palette="viridis")
    plt.title("Missing Values Count by Feature in Fleet and Merged Data")
    plt.xlabel("Number of Missing Values")
    plt.ylabel("Feature Name")
    missing_values_plot_path = visualization_dir / "Missing_Values_bar_plot.png"
    plt.savefig(missing_values_plot_path)
    plt.close()
    logging.info(f"Missing values bar plot saved to {missing_values_plot_path}.")

    # 4. Scatter Plot of Acquisition Mileage vs. Total Miles
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=fleet_df, x="Acquisition_Mileage", y="Total_Miles", color="purple", alpha=0.6)
    plt.title("Acquisition Mileage vs. Total Miles")
    plt.xlabel("Acquisition Mileage")
    plt.ylabel("Total Miles")
    scatter_plot_path = visualization_dir / "acquisition_vs_total_miles.png"
    plt.savefig(scatter_plot_path)
    plt.close()
    logging.info(f"Scatter plot of Acquisition Mileage vs. Total Miles saved to {scatter_plot_path}.")

    # 5. Vehicle Age Distribution by Maintenance Status (Grouped Histogram)
    plt.figure(figsize=(10, 6))
    if "Maintenance_Required" in merged_df.columns:
        sns.histplot(data=merged_df, x="Vehicle_Age", hue="Maintenance_Required", bins=15, kde=True, palette="Set2")
        plt.title("Vehicle Age Distribution by Maintenance Status")
        plt.xlabel("Vehicle Age (years)")
        plt.ylabel("Count")
        maintenance_age_plot_path = visualization_dir / "Vehicle_Age_By_Maintenance_Status.png"
        plt.savefig(maintenance_age_plot_path)
        plt.close()
        logging.info(f"Vehicle Age by Maintenance Status plot saved to {maintenance_age_plot_path}.")
    else:
        logging.warning("Maintenance_Required column not found, skipping Vehicle Age by Maintenance Status plot.")

    # 6. Maintenance Count Over Time
    plt.figure(figsize=(12, 6))
    if "Last_Maintenance_Date" in merged_df.columns:
        merged_df["Last_Maintenance_YearMonth"] = merged_df["Last_Maintenance_Date"].dt.to_period("M")
        maintenance_counts = merged_df["Last_Maintenance_YearMonth"].value_counts().sort_index()
        maintenance_counts.plot(kind="line", color="red")
        plt.title("Maintenance Count Over Time (Monthly)")
        plt.xlabel("Year-Month")
        plt.ylabel("Count of Maintenance Records")
        time_series_plot_path = visualization_dir / "Maintenance_Count_Over_Time.png"
        plt.savefig(time_series_plot_path)
        plt.close()
        logging.info(f"Maintenance count over time plot saved to {time_series_plot_path}.")
    else:
        logging.warning("Last_Maintenance_Date column not found, skipping Maintenance Count Over Time plot.")

    # 7. Box Plots for Outlier Detection
    boxplot_columns = ["Vehicle_Age", "Acquisition_Mileage", "Total_Miles", "Days_Since_Last_Maintenance"]

    for col in boxplot_columns:
        # Fleet dataset
        if col in fleet_df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=fleet_df, y=col, color='skyblue')
            plt.title(f"Box Plot of {col} (Fleet Data)")
            plt.ylabel(col)
            output_path = visualization_dir / f"Box_Plot_{col}_Fleet.png"
            plt.savefig(output_path)
            plt.close()
            logging.info(f"Box plot for {col} in fleet data saved to {output_path}.")

        # Merged dataset
        if col in merged_df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=merged_df, y=col, color='lightgreen')
            plt.title(f"Box Plot of {col} (Merged Data)")
            plt.ylabel(col)
            output_path = visualization_dir / f"Box_Plot_{col}_Merged.png"
            plt.savefig(output_path)
            plt.close()
            logging.info(f"Box plot for {col} in merged data saved to {output_path}.")


if __name__ == "__main__":
    config = load_config()
    clean_and_save_data(config)
