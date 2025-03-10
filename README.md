# Vehicle-Maintenance-Analysis

# Vehicle Maintenance Prediction Model

This project develops a machine learning model to predict vehicle maintenance requirements based on various operational and environmental factors.

## Project Overview

The goal of this project is to build a predictive model that can assess the likelihood of a vehicle requiring maintenance. By analyzing factors such as vehicle age, usage hours, mileage, failure history, and environmental conditions, the model aims to provide insights that can optimize maintenance schedules and reduce downtime.

## Files and Directory Structure

├─ config.yaml           # Configuration file for data paths and model parameters
├── data_clean.py        # Script for cleaning and preprocessing the raw data
├── load_data.py         # Script for loading configuration files
├── ovm_model.py         # Script for training and evaluating the machine learning models
├── README.md            # Project documentation
├── mlruns/              # MLflow tracking directory (created during runs)
└── .venv/               # Virtual environment directory


## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2.  **Create a virtual environment:**

    ```bash
    python3 -m venv .venv
    ```

3.  **Activate the virtual environment:**

    * macOS/Linux:

        ```bash
        source .venv/bin/activate
        ```

    * Windows:

        ```bash
        .venv\Scripts\activate
        ```

4.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    (Note: You'll need to create a `requirements.txt` file listing all the required Python packages. For example:)

    ```
    pandas
    numpy
    scikit-learn
    xgboost
    imblearn
    mlflow
    joblib
    pyyaml
    ```

## Configuration

* The project uses a `config.yaml` file to manage configuration parameters. Modify this file to adjust data paths and model hyperparameters.

## Running the Model

1.  **Ensure your virtual environment is activated.**
2.  **Run the `ovm_model.py` script:**

    ```bash
    python ovm_model.py
    ```

3.  **View MLflow runs:**

    ```bash
    mlflow ui
    ```

    Then, open the provided URL in your browser to view the MLflow UI.

## MLflow Tracking

* MLflow is used to track experiments, parameters, metrics, and artifacts.
* Experiments are organized by date, and runs are named with the start time.
* Artifacts, including configuration files, cleaned data, scalers, and models, are organized into folders within each run.

## Model Details

* The project trains and evaluates Logistic Regression, Random Forest, and XGBoost models.
* Data preprocessing includes handling categorical features, imputing missing values, and scaling numeric features.
* SMOTE is used to address class imbalance.
* Model performance is evaluated using accuracy, precision, recall, F1-score, classification reports, and confusion matrices.

## Key Components

* **`config.yaml`:** Stores configuration parameters.
* **`data_clean.py`:** Cleans and preprocesses the raw data.
* **`ovm_model.py`:** Trains and evaluates the machine learning models.
* **MLflow:** Tracks experiments and manages artifacts.
* **Joblib:** Saves and loads scikit-learn models and transformers.

## Future Enhancements

* Implement more advanced feature engineering techniques.
* Explore additional machine learning models and hyperparameter tuning.
* Deploy the model as a web service or API.
* Add more comprehensive error handling and logging.
* Create more data visualization.
```

