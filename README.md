# Churn Prediction Project

This project provides a machine learning solution to predict customer churn using Python, Pandas, and Scikit-learn. The model is trained on the "Telco Customer Churn" dataset and achieves approximately 82% accuracy.

## Project Structure
```
.
├── .gitignore
├── classify_churn.py
├── requirements.txt
└── README.md
```

## Setup and Installation

Follow these steps to set up the project environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Andrii-Kon/churn-prediction-project.git
    cd churn-prediction-project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv venv
    
    # Activate on Windows
    .\venv\Scripts\activate
    
    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset:**
    *   Download the data from [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
    *   Unzip the archive, find the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file, and place it in the project's root directory.
    *   **Rename the file to `churn_data.csv`**.

## Usage

To run the model training and evaluation script, execute the following command from the project's root directory:

```bash
python classify_churn.py
```

The script will output the model's final accuracy on the test data.

## Results
The trained Logistic Regression model achieves a prediction accuracy of approximately **82.0%**.

```
Model Accuracy on Test Data: 0.8197
```
