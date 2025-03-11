# Vehicle Maintenance Prediction Model

A Machine Learning project designed to predict vehicle maintenance requirements based on a set of features 
like vehicle age, usage hours, historical failures, and environmental conditions. This project uses multiple 
machine learning models and visualizations to ensure insights are transparent and actionable.

## Project Overview

The goal of this project is to:
- Analyze vehicle data to identify the likelihood of maintenance requirements.
- Build scalable machine learning models using techniques like SMOTE for class imbalance and feature scaling.
- Visualize feature importance, class distributions, learning curves, and evaluation metrics for insight generation.
- Implement a reproducible and trackable training pipeline with `MLflow` for experiment tracking and model deployment 
  capabilities.

This project can help businesses enhance their predictive maintenance processes, reduce downtime, 
and optimize operational efficiency.

## **Project Features**
1. **Data Preprocessing**:
    - Handles categorical variables with OneHotEncoding.
    - Imputes missing values using a median strategy.
    - Balances class distribution using SMOTE to overcome data imbalance.

2. **Machine Learning Pipeline**:
    - Supports multiple models such as:
        - Logistic Regression
        - Random Forest
        - XGBoost

    - Automates hyperparameter management via configuration files.

3. **Visualization Capabilities**:
    - Class distribution visualization (raw and after SMOTE).
    - Feature importance for interpretable models.
    - Learning curves to evaluate model performance on increasing data sizes.
    - Confusion Matrix for evaluation metrics.

4. **Experiment Tracking**:
    - Uses `MLflow` to log metrics, visualization artifacts, and model versions.
    - Tracks configuration and experiment conditions for reproducibility.

5. **Model Deployment Readiness**:
    - Trains models on scalable pipelines with saved models for future use.

## **Technologies Used**

- **Backend/Frameworks**:
    - Python 3.9.2
    - Scikit-learn
    - XGBoost
    - SMOTE from `imbalanced-learn`

- **Visualization**:
    - Matplotlib
    - Seaborn

- **Experiment Tracking**:
    - MLflow

- **Data Handling**:
    - Pandas
    - NumPy

- **Others**:
    - Joblib (Model Serialization)
    - Git (Version Control)

## **Installation and Setup Instructions**

### **Prerequisites**

Make sure the following is installed:
- Python (>= 3.9.2)
- Pip or Conda for dependency management
- Git for version control

### **Step-by-Step Setup**

1. **Clone the Repository**:

    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2. **Create and Activate a Virtual Environment**:

    ```bash
       python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
    ```

3. **Install dependencies:**

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

4.  **Set up Configuration**:
    - Modify the configuration file (`config.yaml`) to fit your requirements (e.g., dataset paths, ML parameters, random state, etc.).

5. **Run the Training Script**:

    ```bash
    python ovm_model.py
    ```

6.  **Track Experiment Results**:

    - Start an MLflow server locally using:

    ```bash
    mlflow ui
    ```

- Open the MLflow UI at `http://127.0.0.1:5000` to view logged metrics and visualizations.


## MLflow Tracking

* MLflow is used to track experiments, parameters, metrics, and artifacts.
* Experiments are organized by date, and runs are named with the start time.
* Artifacts, including configuration files, cleaned data, scalers, and models, are organized into folders within each run.

## **Project Structure**

```
project_root/
│
├── config.yaml                     # Configuration file for hyperparameters and paths
├── requirements.txt                # List of Python dependencies
├── train_and_evaluate_model.py     # Main pipeline script to preprocess, train, and evaluate models
├── utils/
│   ├── plot_learning_curve.py      # Helper for learning curve visualizations
│   ├── plot_feature_importance.py  # Helper for feature importance visuals
│   └── ...                         # Additional utilities
├── visualizations/                 # Directory for saved visualizations (e.g., learning curves, confusion matrices)
├── models/                         # Directory for trained models
└── README.md                       # Project documentation (this file)
```

## **How to Use**
1. Place your cleaned dataset in the path specified in `config.yaml` (e.g., `data/cleaned_data.csv`).
2. Modify the `config.yaml` file if necessary to:
    - Adjust model hyperparameters
    - Change train-test split ratios or paths for input/output files
    - Enable/disable certain plots or logs

3. Run the training script (`train_and_evaluate_model.py`) to preprocess the data, train models, evaluate performance, and generate visualizations.

## **Key Outputs**
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrices for each model.
- **Visualizations**:
    - Learning Curves for understanding model performance with more data.
    - Feature Importance to interpret model decisions.
    - Class distribution (before and after balancing).

- **Saved Models**: Serialized models stored in the `models/` directory.
- **MLflow Tracking**: Experiment parameters, metrics, and artifacts logged for reproducibility and analysis.

## **Customization**
- **Add New Models**:
    - Edit the `models` dictionary in `train_and_evaluate_model.py` to add new algorithms.

- **Update Configurations**:
    - Change parameters in `config.yaml` to tune models, data split ratios, or preprocessing steps.

## **Future Enhancements**
1. **Hyperparameter Tuning**:
    - Automate tuning using tools like `GridSearchCV` or `Optuna`.

2. **Add Explainability Techniques**:
    - Integrate SHAP or LIME for deeper model interpretability.

3. **Deploy Models**:
    - Use frameworks like Flask or FastAPI to create APIs for real-time predictions.

4. **Pipeline Automation**:
    - Transition to tools like `Airflow` or `Luigi` for seamless data-to-model pipelines.

## **Contribution**
Contributions are welcome! If you'd like to contribute:
1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Submit a pull request with a detailed explanation of your changes.

## **License**
This project is licensed under the [MIT License](LICENSE).


