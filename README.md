# Loan Risk Scoring System

Provide a **what-if analysis** for loan approval likelihood based on individual status (**Independent** or **Dependent**):

1. **Collected data** from Kaggle loan default datasets across borrower categories
2. **Trained a machine learning model** to score loan risk
3. **Tested and deployed the model** to provide interactive what-if analysis

**Live demo:** 


## 🚀 Features

✔ Independent & Dependent borrower forms
✔ Instant loan default probability prediction
✔ Clear risk categories (Low / Medium / High)
✔ DTI (0–1) & LTV risk factor analysis
✔ Interactive what-if analysis (down payment, interest rate, credit score, monthly debt)
✔ One-click CSV export of loan scenarios

## 🛠️ Tech Stack

* **Language:** Python
* **Frameworks/Libraries:** Streamlit, NumPy, pandas, scikit-learn, joblib
* **Training helpers:** imbalanced-learn (optional but recommended)

## 📦 Installation

```bash
# 1) Clone the repository
git clone https://github.com/<your-username>/Loan-Risk-Scoring-System.git
cd Loan-Risk-Scoring-System

# 2) Create & activate a virtual environment
python -m venv .venv
# Mac/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

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
imbalanced-learn
```

> If you prefer minimal deps, you can omit `imbalanced-learn`, but it helps with class imbalance.

---

## ▶ Run Locally

```bash
# (A) If you have raw data at data/data.csv:
python make_clean_dataset.py      # writes data/data_clean.csv
python train_model.py             # trains on data/data_clean.csv and saves models/risk_model.pkl

# (B) If train_model.py auto-cleans inside (optional setup):
python train_model.py

# Launch the app
streamlit run app.py
```

## 💡 How to Use

1. Select **Independent** or **Dependent**
2. Enter borrower details (income, monthly debt, property value, down payment, credit score, etc.)
3. Submit → see **default probability** and **risk label**
4. Open **What-if analysis** to simulate changes (extra down payment, lower rate, higher score, lower monthly debt)
5. Export results as CSV


## 📂 Project Structure

```
Loan-Risk-Scoring-System/
├─ app.py                   # Streamlit app (what-if analysis UI)
├─ train_model.py           # Train & save model (uses cleaned data)
├─ clean_loan_data.py       # Data cleaning rules (DTI unit, clipping, sanity)
├─ make_clean_dataset.py    # Reads data/data.csv → writes data/data_clean.csv
├─ models/
│   └─ risk_model.pkl       # Trained model pipeline
├─ data/
│   ├─ data.csv             # Raw dataset (not committed if large)
│   └─ data_clean.csv       # Cleaned dataset used for training
├─ tests/                   # (Optional) sanity tests
├─ requirements.txt         # Dependencies
└─ README.md                # This file
```

## 🧠 Dataset & Modeling Notes

* **Target:** `Default` (0/1)
* **Core features used:**
  `Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, HasCoSigner`
* **DTI unit:** Treated as a **ratio (0–1)** end-to-end (cleaning script auto-converts percent if needed).
* **Class imbalance:** Model uses `class_weight="balanced"`; you can also add calibration or SMOTE for smoother probabilities.

## 🌐 Hosted App

* **Browser link:** [Loan Risk Scoring System](https://loan-risk-scoring-app-ztqeqbhh9cnf8ktcjdusxy.streamlit.app/)


## 🛣️ Roadmap

* [ ] Add monthly payment breakdown (principal vs. interest)
* [ ] Provide personalized credit/DTI improvement tips
* [ ] Compare multiple borrower profiles side-by-side
* [ ] Train on expanded datasets for better accuracy

## 📊 Sample Output

Example CSV export:


## 🧰 Troubleshooting

* **`data/data_clean.csv not found`** → Run `python make_clean_dataset.py` (creates the cleaned file)
* **`ModuleNotFoundError`** → Run `pip install -r requirements.txt` inside your virtualenv
* **Streamlit won’t start** → Ensure virtual environment is active (`source .venv/bin/activate` or `.venv\Scripts\Activate.ps1`)
* **Model not found** → Run `python train_model.py` to create `models/risk_model.pkl`

## License

MIT License – see [LICENSE](LICENSE) for details.
