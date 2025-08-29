# Loan Risk Scoring System

A **what-if analysis** tool for predicting loan default probability based on individual borrower status (**Independent** or **Dependent**). This system uses a machine learning model trained on loan default datasets to provide instant risk assessments.

> **Live Demo:**


> **Browser Link:** [Loan Risk Scoring System](https://loan-risk-scoring-app-ztqeqbhh9cnf8ktcjdusxy.streamlit.app/)

-----

## 🚀 Key Features

  * **Independent & Dependent Forms:** Tailored input forms for different borrower statuses.
  * **Instant Risk Prediction:** Provides an immediate probability of loan default.
  * **Risk Categorization:** Classifies risk into **Low**, **Medium**, and **High** categories.
  * **Interactive What-if Analysis:** Allows you to simulate changes in key factors like credit score, down payment, and monthly debt to see how they impact risk.
  * **Data Export:** One-click CSV export of loan scenarios.

-----

## 🛠️ Tech Stack

  * **Language:** Python
  * **Libraries:** `streamlit`, `scikit-learn`, `pandas`, `numpy`, `joblib`

-----

## 📦 Installation

To get the project running on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<your-username>/Loan-Risk-Scoring-System.git
    cd Loan-Risk-Scoring-System
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # Mac/Linux
    source .venv/bin/activate
    # Windows
    .venv\Scripts\Activate.ps1
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

-----

## ▶ Run Locally

1.  **Train the model:** Run `train_model.py` to process your data and save the machine learning model.
    ```bash
    python train_model.py
    ```
2.  **Launch the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    This will open the app in your browser.

-----

## 💡 How to Use

1.  Select your borrower type (**Independent** or **Dependent**).
2.  Enter your financial details and click **Submit**.
3.  View your **default probability** and **risk label**.
4.  Use the **What-if analysis** section to experiment with different loan scenarios.
5.  Click the **Export** button to download a CSV of your results.

-----

## 📂 Project Structure

```
Loan-Risk-Scoring-System/
├─ app.py                # Main Streamlit application
├─ train_model.py        # Model training script
├─ models/
│  └─ risk_model.pkl     # Saved machine learning model
├─ data/
│  └─ data_clean.csv     # Cleaned dataset (used for training)
├─ requirements.txt      # Python dependencies
└─ README.md             # This file
```

-----

## 🛣️ Roadmap

  * Add a monthly payment breakdown.
  * Provide personalized financial advice based on risk factors.
  * Allow side-by-side comparison of multiple loan profiles.

-----

## 🧰 Troubleshooting

  * **`ModuleNotFoundError`**: Ensure you have installed all dependencies from `requirements.txt` within your virtual environment.
  * **`Model not found`**: Run `python train_model.py` to train and save the model file (`models/risk_model.pkl`).

-----

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

-----

## 🙏 Acknowledgments

  * **Kaggle** for the loan default datasets.
  * **Streamlit** for the interactive web interface.
  * **scikit-learn** for the machine learning model.
