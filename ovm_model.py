import pandas as pd
import numpy as np
import joblib
import warnings
import mlflow
import mlflow.sklearn
import sys
import sklearn
import subprocess
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
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

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def plot_learning_curve(estimator, X, y, scoring="accuracy", output_path=None):
    """
    Plots the learning curve for a model.

    Args:
        estimator: The model (e.g., XGBClassifier) to evaluate.
        X: Input features/data.
        y: Target labels.
        scoring: The scoring metric to use (e.g., "accuracy").
        output_path (str): Optional; path to save the plot.
    """

    # Generate learning curve data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["RANDOM_STATE"])
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=skf, scoring=scoring, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    # Calculate mean and standard deviation
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="green", label="Cross-Validation Score")

    # Add the error bands
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1, color="blue"
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1, color="green"
    )

    # Graph labels and layout
    plt.title("Learning Curve")
    plt.xlabel("Training Samples")
    plt.ylabel(scoring)
    plt.legend(loc="best")
    plt.grid()

    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
        print(f"Learning curve saved to: {output_path}")
    else:
        plt.show()


def plot_feature_importance(model, feature_names, output_path=None, top_n=10):
    """
    Plots and saves feature importance for a given model.

    Args:
        model: The trained machine learning model.
        feature_names: List of feature names.
        output_path (str): Optional; path to save the plot.
        top_n (int): Number of top features to display (default: 10).
    """
    try:
        # Determine feature importance based on model type
        importances = None
        if hasattr(model, "feature_importances_"):  # Tree-based models (e.g., Random Forest, XGBoost)
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):  # Linear models (e.g., Logistic Regression)
            if len(model.coef_.shape) > 1:  # Multiclass classification case
                importances = model.coef_[0]  # Use coefficients for the first class
            else:
                importances = model.coef_
            importances = np.abs(importances)  # Use absolute values for coefficients
        else:
            logging.warning(f"{type(model).__name__} does not support feature importance plotting.")
            return

        # If feature importance is still None, skip plotting
        if importances is None:
            logging.warning(f"No feature importance available for {type(model).__name__}.")
            return

        # Create a DataFrame to sort and rank features by importance
        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # Plot the top N features
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=feature_importance.head(top_n),
            x="Importance",
            y="Feature",
            palette="viridis"
        )
        plt.title(f"Top {top_n} Feature Importances ({type(model).__name__})", fontsize=14)
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()

        # Save or display the plot
        if output_path:
            plt.savefig(output_path)
            logging.info(f"Feature importance plot saved to {output_path}")
        else:
            plt.show()

        # Clean up the plot after saving or showing
        plt.close()

    except Exception as e:
        logging.error(f"Error plotting feature importance: {e}")


def train_and_evaluate_model(config):
    Path('visualizations').mkdir(parents=True, exist_ok=True)
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
    # logging.info(f"Configuration Loaded: {config}")
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config["RANDOM_STATE"], stratify=y
    )

    logging.info(f"Original class distribution: {y_train.value_counts()}")
    smote = SMOTE(random_state=config["RANDOM_STATE"])
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logging.info(f"Resampled class distribution: {pd.Series(y_train_resampled).value_counts()}")

    # Visualizing Class Distribution After SMOTE
    plt.figure(figsize=(8, 6))
    class_counts = pd.Series(y_train_resampled).value_counts()

    # Fix for categorical parsing warning: Treat labels explicitly as integers and then cast to strings.
    class_indices = class_counts.index.astype(int)  # Ensure numeric labels
    class_labels = [f"Class {c}" for c in class_indices]  # Create readable class labels

    scaler = StandardScaler()
    scaler.fit(X_train)
    scaled_columns = X_train.columns  # Preserve column names for scaled data
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_resampled), columns=scaled_columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=scaled_columns)

    # Create input_example
    input_example = X_test_scaled[:1]

    # Initialize y_pred_dict to store all predictions
    y_pred_dict = {}

    # Define machine learning models
    models = {
        "Logistic Regression": LogisticRegression(**config["MODEL_PARAMS"]["LOGISTIC_REGRESSION"],
                                                  random_state=config["RANDOM_STATE"]),
        "Random Forest": RandomForestClassifier(**config["MODEL_PARAMS"]["RANDOM_FOREST"],
                                                random_state=config["RANDOM_STATE"]),
        "XGBoost": XGBClassifier(**config["MODEL_PARAMS"]["XGBOOST"]),
    }

    trained_models = {}

    # Ensure any previous MLFlow run is properly closed before starting a new one
    if mlflow.active_run():
        mlflow.end_run()  # End the active run to avoid conflicts

    # Start MLFlow experiment
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        with mlflow.start_run(run_name=run_name):
            # Log initial configuration and setup details
            mlflow.log_artifact("config.yaml", artifact_path="Configuration")
            mlflow.log_artifact(config["OUTPUT_PATHS"]["CLEANED_DATA"], artifact_path="Data")
            joblib.dump(scaler, "scaler.joblib")
            mlflow.log_artifact("scaler.joblib", artifact_path="Scalers")
            commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                                  stderr=subprocess.DEVNULL).decode("ascii").strip()
            mlflow.log_param("git_commit_hash", commit_hash)
            mlflow.log_param("python_version", sys.version)
            mlflow.log_param("sklearn_version", sklearn.__version__)
            mlflow.log_param("test_size", config["TRAIN_TEST_SPLIT"]["TEST_SIZE"])

            # STEP 1: **Raw Dataset Class Distribution Visualization**
            try:
                # Ensure the directory for visualizations exists
                output_path = Path(config["OUTPUT_PATHS"]["VIZ_DIR"])
                output_path.mkdir(parents=True, exist_ok=True)

                # Check if the target variable exists
                if 'Maintenance_Required' not in cleaned_df.columns:
                    logging.error("Error: 'Maintenance_Required' column not found in DataFrame.")
                else:
                    logging.info("Generating raw class distribution visualization.")
                    # Count occurrences of each class
                    class_counts_raw = cleaned_df['Maintenance_Required'].value_counts()
                    class_indices_raw = class_counts_raw.index.astype(int)
                    class_labels_raw = [f"Class {c}" for c in class_indices_raw]

                    # Plot raw dataset distribution
                    plt.figure(figsize=(8, 6))
                    sns.barplot(x=class_labels_raw, y=class_counts_raw.values, palette="plasma")
                    plt.title("Class Distribution in Raw Dataset", fontsize=14)
                    plt.xlabel("Classes", fontsize=12)
                    plt.ylabel("Frequency", fontsize=12)
                    plt.xticks(fontsize=10)
                    plt.yticks(fontsize=10)
                    plt.tight_layout()

                    # Save the visualization
                    raw_class_dist_path = output_path / "class_distribution_raw.png"
                    plt.savefig(raw_class_dist_path)
                    plt.close()
                    logging.info(f"Raw dataset class distribution plot saved to {raw_class_dist_path}")

                    # Log as an MLflow artifact
                    mlflow.log_artifact(str(raw_class_dist_path), artifact_path="Visualizations")

            except Exception as e:
                logging.error(f"Failed to generate raw dataset distribution visualization: {e}")

            # STEP 2: **SMOTE Visualization Code**
            logging.info("Generating SMOTE class distribution visualization.")
            plt.figure(figsize=(8, 6))
            class_counts = pd.Series(y_train_resampled).value_counts()

            # Fix for categorical label warnings
            class_indices = class_counts.index.astype(int)
            class_labels = [f"Class {c}" for c in class_indices]

            sns.barplot(
                x=class_labels,  # Proper string labels for the X-axis
                y=class_counts.values,  # Y-axis bar heights
                palette="viridis",  # Color palette
                hue=None,  # Explicitly set hue to None to avoid warnings
                legend=False  # No legend needed
            )
            plt.title("Class Distribution After SMOTE Balancing", fontsize=14)
            plt.xlabel("Classes", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()

            # Save and log the SMOTE class distribution plot as an artifact
            smote_plot_path = Path('visualizations') / "class_distribution_after_smote.png"
            smote_plot_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            plt.savefig(smote_plot_path)
            logging.info(f"SMOTE class distribution plot saved to {smote_plot_path}")
            mlflow.log_artifact(smote_plot_path, artifact_path="Visualizations")
            plt.close()

            # STEP 3: **Start training models**
            for model_name, model in models.items():
                logging.info(f"{model_name} Start...")
                # Train the model on the scaled training data
                model.fit(X_train_scaled, y_train_resampled)
                trained_models[model_name] = model

                # Save the trained model locally
                model_path = Path(config["OUTPUT_PATHS"]["MODEL_DIR"]) / f"{model_name.replace(' ', '_')}.pkl"
                joblib.dump(model, model_path)
                logging.info(f"{model_name} saved to {model_path}")

                # Log the model in MLflow
                if model_name == "XGBoost":
                    # Save XGBoost-specific JSON model
                    model.save_model(
                        str(Path(config["OUTPUT_PATHS"]["MODEL_DIR"]) / f"{model_name.replace(' ', '_')}.json"))
                    mlflow.log_artifact(
                        str(Path(config["OUTPUT_PATHS"]["MODEL_DIR"]) / f"{model_name.replace(' ', '_')}.json"),
                        artifact_path="Models")
                else:
                    mlflow.sklearn.log_model(model, artifact_path=f"Models/{model_name.replace(' ', '_')}",
                                             input_example=input_example)

                # Generate feature importance visualizations, if applicable
                if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
                    # Use the general plot_feature_importance function for feature visualization
                    feature_importance_output_path = Path(
                        'visualizations') / f"{model_name.replace(' ', '_')}_feature_importance.png"
                    plot_feature_importance(
                        model, feature_names=scaled_columns,
                        output_path=feature_importance_output_path
                    )
                    mlflow.log_artifact(
                        feature_importance_output_path, artifact_path="Visualizations"
                    )
                else:
                    logging.info(f"{model_name} does not support feature importance plotting.")

                # **Plot and save the learning curve for all models**
                learning_curve_output_path = Path(
                    'visualizations') / f"{model_name.replace(' ', '_')}_learning_curve.png"
                try:
                    plot_learning_curve(
                        estimator=model,
                        X=X_train_scaled.values,
                        y=y_train_resampled,
                        scoring="accuracy",
                        output_path=learning_curve_output_path
                    )
                    mlflow.log_artifact(
                        str(learning_curve_output_path), artifact_path="Visualizations"
                    )
                    logging.info(f"Learning curve plotted and saved for {model_name} at {learning_curve_output_path}")
                except Exception as e:
                    logging.error(f"Failed to plot learning curve for {model_name}: {e}")

                # Model evaluation and logging metrics
                y_pred = model.predict(X_test_scaled)
                y_pred_dict[model_name] = y_pred
                accuracy = model.score(X_test_scaled, y_test)

                print(f"\n{model_name} Report:")
                print(" Accuracy Score:", accuracy)
                logging.info(" Classification Report:\n" + str(classification_report(y_test, y_pred)))
                print(" Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

                # Confusion Matrix Visualization
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
                plt.title(f'{model_name} Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                confusion_matrix_path = Path('visualizations') / f"{model_name.replace(' ', '_')}_confusion_matrix.png"
                plt.savefig(confusion_matrix_path)
                mlflow.log_artifact(confusion_matrix_path, artifact_path="Visualizations")
                logging.info(f"Confusion matrix saved to {confusion_matrix_path}")
                plt.close()

                # Compute metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                # Log metrics to MLFlow
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision_weighted", precision)
                mlflow.log_metric("recall_weighted", recall)
                mlflow.log_metric("f1_weighted", f1)

                logging.info(f"{model_name} Completed")

        return y_pred_dict, trained_models, X, y_train_resampled, y_test, config


if __name__ == "__main__":
    config = load_config()

    # Create necessary directories
    Path(config["OUTPUT_PATHS"]["MODEL_DIR"]).mkdir(parents=True, exist_ok=True)
    # Train and evaluate the model
    train_and_evaluate_model(config)
