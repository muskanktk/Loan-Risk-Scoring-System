# Loan Risk Scoring System

Provide a **what-if analysis** for loan approval likelihood based on individual status (**Independent** or **Dependent**):

1. **Collected data** from Kaggle loan default datasets across borrower categories
2. **Trained a machine learning model** to score loan risk
3. **Tested and deployed the model** to provide interactive what-if analysis

**Live demo:** [https://loan-risk-scoring-app-ztqeqbhh9cnf8ktcjdusxy.streamlit.app/](https://loan-risk-scoring-app-ztqeqbhh9cnf8ktcjdusxy.streamlit.app/)

## Features

* **Two borrower forms:** Independent (uses your info) or Dependent (uses guardian/co-signer info)
* **Default probability prediction:** Shows chance of repayment vs. default and assigns a risk label (Low, Medium, High)
* **DTI and LTV analysis:** Explains how debt-to-income ratio and loan-to-value ratio affect your risk
* **What-if analysis:** Simulate changes (extra down payment, lower interest rate, improved credit score) and instantly see risk impact
* **CSV Export:** Download your loan scenario results

## Tech Stack

* **Language:** Python
* **Frameworks/Libraries:** Streamlit, NumPy, pandas, scikit-learn, joblib
* **Tools:** Git, VS Code

## Installation

```bash
# 1) Clone the repository
git clone https://github.com/<your-username>/Loan-Risk-Scoring-System.git
cd Loan-Risk-Scoring-System

# 2) Create & activate a virtual environment
python -m venv venv
# Mac/Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt
```

**Sample `requirements.txt`:**

```
streamlit
pandas
numpy
scikit-learn
joblib
```

---

## Run Locally

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

## How to Use

1. Choose **Independent** or **Dependent**
2. Fill out the loan form with your details (income, debts, property value, down payment, credit score, etc.)
3. Submit → see your **loan risk score** and **probability of default**
4. Expand **What-if analysis** to test scenarios (e.g., add down payment, lower interest rate)
5. Download results as CSV

## Project Structure

```
Loan-Risk-Scoring-System/
├─ app.py               # Streamlit app entry point
├─ requirements.txt     # Python dependencies
├─ train_model.py       # Script to train and save ML model
├─ models/
│   └─ risk_model.pkl   # Trained loan risk model
├─ tests/               # (Optional) test scripts
├─ data/                # Training dataset(s)
├─ README.md            # This file
└─ .env                 # (Optional) secrets/config
```

---

## Hosted App

* **Browser link:** [Loan Risk Scoring System](https://loan-risk-scoring-app-ztqeqbhh9cnf8ktcjdusxy.streamlit.app/)

---

## Roadmap

* [ ] Add visualization of monthly payment breakdown (principal vs. interest)
* [ ] More granular credit / DTI suggestions
* [ ] Support for multiple borrower profiles comparison
* [ ] Expand dataset for improved accuracy

## Sample Output

Example exported scenario:
[loan_scenario (2).csv](https://github.com/user-attachments/files/21991303/loan_scenario.2.csv)

## Troubleshooting

* **`ModuleNotFoundError`** → Run `pip install -r requirements.txt` inside your virtualenv
* **Streamlit won’t start** → Ensure virtual environment is active (`which streamlit` or `venv\Scripts\activate`)
* **Model not found** → Run `python train_model.py` to create `models/risk_model.pkl`

## License

MIT License – see [LICENSE](LICENSE) for details.
