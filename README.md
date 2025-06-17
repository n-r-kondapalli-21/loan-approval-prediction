# Loan Approval Prediction

This project is a web application that predicts whether a loan application will be approved or not based on the applicant's information. It uses a machine learning model to make the predictions.

## Dataset

The project uses the `loan_data.csv` dataset. This dataset contains information about past loan applications, including applicant details and whether the loan was approved.

**Note:** You should add a more detailed description of the dataset here, including the features (columns) and their meanings.

## Machine Learning Model

A machine learning model is trained on the `loan_data.csv` dataset to predict loan approvals. The trained model artifacts, such as the `label_encoder.joblib`, are used by the web application.

You should provide more details about the model here, such as:
*   The type of model used (e.g., Logistic Regression, Random Forest, etc.)
*   The features used for training.
*   The performance of the model (e.g., accuracy, precision, recall).

## Web Application

The project includes a web application built with Streamlit that allows users to input their information and get a loan approval prediction.

### How to Run the Application

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv loan_env
    source loan_env/bin/activate  # On Windows, use `loan_env\Scripts\activate`
    ```

3.  **Install the dependencies:**
    The required libraries are listed in `loan_env/Lib/site-packages/`. A `requirements.txt` file should be created for easier installation.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: A `requirements.txt` file is not yet present. You can generate one using `pip freeze > requirements.txt` after installing the necessary packages in your virtual environment.*

4.  **Run the Streamlit application:**
    ```bash
    streamlit run loan_app.py
    ```

## Project Structure

```
.
├── loan_env/
│   ├── loan_app.py         # Main Streamlit application file
│   ├── loan_data.csv       # Dataset for training and prediction
│   └── label_encoder.joblib  # Saved label encoder
└── README.md
``` 
