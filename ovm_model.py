import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import logging
from data_clean import clean_and_save_data
from load_data import load_config
from pathlib import Path
import warnings
import mlflow
import mlflow.sklearn
import sys
import sklearn
import subprocess
import datetime

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def train_and_evaluate_model(config):
    """
    Loads cleaned data, trains and evaluates machine learning models.

    Args:
        config (dict): Configuration dictionary.
    """
    nowdate = datetime.date.today()
    experiment_name = "Vehicle Maintenance Model, experiment run on " + str(nowdate)
    experiment = mlflow.set_experiment(experiment_name)
    run_name = "Run started at " + datetime.datetime.now().strftime("%H:%M")

    logging.info("Triggering data cleaning and saving process...")
    clean_and_save_data(config)
    logging.info("Data cleaning and saving process completed.")

    logging.info("Loading cleaned data for model training...")
    cleaned_df = pd.read_csv(config["OUTPUT_PATHS"]["CLEANED_DATA"], low_memory=False)
    logging.info("Cleaned data loaded successfully.")

    columns_to_keep = [
        "Vehicle_Age", "Usage_Hours", "Total_Miles", "Failure_History", "Predictive_Score",
        "Impact_on_Efficiency", "Downtime_Maintenance", "Days_Since_Last_Maintenance",
        "Weather_Conditions", "Road_Conditions", "Maintenance_Required"
    ]
    cleaned_df = cleaned_df[columns_to_keep].copy()

    categorical_features = ["Weather_Conditions", "Road_Conditions"]
    valid_categorical_features = [col for col in categorical_features if col in cleaned_df.columns
                                  and cleaned_df[col].notna().any()]
    if valid_categorical_features:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_categories = encoder.fit_transform(cleaned_df[valid_categorical_features])
        encoded_feature_names = encoder.get_feature_names_out(valid_categorical_features)
        encoded_df = pd.DataFrame(encoded_categories, columns=encoded_feature_names)
        cleaned_df.drop(columns=valid_categorical_features, inplace=True)
        cleaned_df = pd.concat([cleaned_df, encoded_df], axis=1)

    imputer = SimpleImputer(strategy="median")
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    cleaned_df[numeric_cols] = imputer.fit_transform(cleaned_df[numeric_cols])

    X = cleaned_df.drop(columns=["Maintenance_Required"])
    y = cleaned_df["Maintenance_Required"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=config["RANDOM_STATE"], stratify=y)

    smote = SMOTE(random_state=config["RANDOM_STATE"])
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Create input_example
    input_example = X_test_scaled[:1]

    models = {
        "Logistic Regression": LogisticRegression(**config["MODEL_PARAMS"]["LOGISTIC_REGRESSION"],
                                                  random_state=config["RANDOM_STATE"]),
        "Random Forest": RandomForestClassifier(**config["MODEL_PARAMS"]["RANDOM_FOREST"],
                                                random_state=config["RANDOM_STATE"]),
        "XGBoost": XGBClassifier(**config["MODEL_PARAMS"]["XGBOOST"]),
    }
    trained_models = {}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_artifact("config.yaml", artifact_path="Configuration")
            mlflow.log_artifact(config["OUTPUT_PATHS"]["CLEANED_DATA"], artifact_path="Data")
            joblib.dump(scaler, "scaler.joblib")
            mlflow.log_artifact("scaler.joblib", artifact_path="Scalers")

            commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
            mlflow.log_param("git_commit_hash", commit_hash)
            mlflow.log_param("python_version", sys.version)
            mlflow.log_param("sklearn_version", sklearn.__version__)
            mlflow.log_param("test_size", config["TRAIN_TEST_SPLIT"]["TEST_SIZE"])

            for model_name, model in models.items():
                logging.info(f"{model_name} Start...")
                model.fit(X_train_scaled, y_train_resampled)
                trained_models[model_name] = model

                model_path = Path(config["OUTPUT_PATHS"]["MODEL_DIR"]) / f"{model_name.replace(' ', '_')}.pkl"
                joblib.dump(model, model_path)
                logging.info(f"{model_name} saved to {model_path}")

                if model_name == "XGBoost":
                    model.save_model(
                        str(Path(config["OUTPUT_PATHS"]["MODEL_DIR"]) / f"{model_name.replace(' ', '_')}.json"))
                    mlflow.log_artifact(
                        str(Path(config["OUTPUT_PATHS"]["MODEL_DIR"]) / f"{model_name.replace(' ', '_')}.json"),
                        artifact_path="Models")

                else:
                    mlflow.sklearn.log_model(model, artifact_path=f"Models/{model_name.replace(' ', '_')}",
                                             input_example=input_example)

                y_pred = model.predict(X_test_scaled)
                print(f"\n{model_name} Report:")
                print(" Accuracy Score:", model.score(X_test_scaled, y_test))
                print(" Classification Report:\n", classification_report(y_test, y_pred))
                print(" Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision_weighted", precision)
                mlflow.log_metric("recall_weighted", recall)
                mlflow.log_metric("f1_weighted", f1)

                logging.info(f"{model_name} Completed")


if __name__ == "__main__":
    config = load_config()
    Path(config["OUTPUT_PATHS"]["MODEL_DIR"]).mkdir(parents=True, exist_ok=True)
    train_and_evaluate_model(config)
