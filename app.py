import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ================= Page config FIRST =================
st.set_page_config(page_title="Loan Default Risk Scoring System", layout="centered")

# ================= Custom CSS (background, buttons, title) =================
st.markdown(
    """
    <style>
        .stApp { background-color: #e6f2ff; }
        div.stButton > button:first-child {
            background-color: #003366; color: white; border: none; border-radius: 8px;
            padding: 0.5em 1.25em; font-weight: 600;
        }
        div.stButton > button:first-child:hover { background-color: #002244; color: white; }
        h1 { color: #003366 !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ================= Title + Subtitle =================
st.title("Loan Default Risk Scoring System")
st.markdown(
    """
    <p style="font-size:18px; color:#003366; font-weight:300; margin-top:-10px; margin-bottom:20px; text-align:center;">
        Choose the option that best reflects your current status
    </p>
    """,
    unsafe_allow_html=True
)

# ================= Borrower Type Buttons (unique keys) =================
outer_left, outer_mid, outer_right = st.columns([1, 2, 1])
with outer_mid:
    col_left, col_right = st.columns(2)
    with col_left:
        if st.button("Independent", key="btn_independent"):
            st.session_state["borrower_type"] = "Independent"
    with col_right:
        if st.button("Dependent", key="btn_dependent"):
            st.session_state["borrower_type"] = "Dependent"

# Placeholder to render the download button **after** the form
download_placeholder = st.empty()

# ================= Model loader (cached) =================
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "models" / "risk_model.pkl"
    return joblib.load(model_path)

try:
    model = load_model()
except Exception:
    model = None

# ================= Helpers =================
def monthly_payment(principal: float, annual_rate_pct: float, n_months: int) -> float:
    """Principal + interest (no taxes/insurance)."""
    r = (annual_rate_pct / 100.0) / 12.0
    if r == 0:
        return principal / n_months
    return principal * (r * (1 + r) ** n_months) / ((1 + r) ** n_months - 1)

def build_feature_row(age, income, loan_amount, credit_score, months_employed,
                      interest_rate, term_months, dti_value, has_cosigner) -> pd.DataFrame:
    """Single source of truth for the model input row (names must match training)."""
    return pd.DataFrame([{
        "Age": int(age),
        "Income": float(income),
        "LoanAmount": float(loan_amount),
        "CreditScore": float(credit_score),
        "MonthsEmployed": int(months_employed),
        "InterestRate": float(interest_rate),
        "LoanTerm": int(term_months),
        "DTIRatio": float(dti_value),   # percent if you trained that way
        "HasCoSigner": int(has_cosigner)
    }])

# ================= Sidebar (controls) =================
st.sidebar.header("Scoring Options")
# UI computes DTI as ratio (0–1). If training used percent, convert ratio→percent.
dti_is_ratio = st.sidebar.toggle(
    "UI DTI is a ratio 0–1 (convert to % for model)", value=True, key="opt_dti_is_ratio"
)
low_thr = st.sidebar.slider("Low → Medium threshold", 0.00, 0.50, 0.20, 0.01, key="opt_low_thr")
med_thr = st.sidebar.slider("Medium → High threshold", 0.30, 0.90, 0.50, 0.01, key="opt_med_thr")

# ================= Main Form =================
if "borrower_type" in st.session_state:
    bt = st.session_state["borrower_type"]
    st.write(f"Selection: **{bt}**")

    with st.form("borrower_form", clear_on_submit=False):
        st.subheader(f"{bt} Borrower Form")
        left, right = st.columns(2)

        if bt == "Independent":
            with left:
                age = st.number_input("Age", min_value=18, max_value=100, value=25, step=1, key="inp_age_ind")
                annual_income = st.number_input("Annual Income ($)", min_value=0, value=50_000, step=1_000, key="inp_income_ind")
                monthly_debt = st.number_input("Monthly Non-Housing Debt ($)", min_value=0, value=300, step=50, key="inp_debt_ind")
                employment_years = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=2, step=1, key="inp_empyrs_ind")
            with right:
                credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=680, step=1, key="inp_cs_ind")
                property_value = st.number_input("Property Value ($)", min_value=50_000, value=350_000, step=5_000, key="inp_prop_ind")
                down_payment = st.number_input("Down Payment ($)", min_value=0, value=35_000, step=5_000, key="inp_down_ind")
                interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=25.0, value=6.5, step=0.1, key="inp_rate_ind")
                term_months = st.number_input("Term (months)", min_value=60, max_value=480, value=360, step=12, key="inp_term_ind")
        else:
            with left:
                age = st.number_input("Age", min_value=18, max_value=100, value=22, step=1, key="inp_age_dep")
                relation = st.selectbox("Relationship to Guardian / Co-signer", ["Parent", "Guardian", "Co-signer"], key="inp_rel_dep")
                guardian_employment_years = st.number_input("Guardian Employment Length (years)", min_value=0, max_value=50, value=8, step=1, key="inp_empyrs_dep")
                guardian_annual_income = st.number_input("Guardian Annual Income ($)", min_value=0, value=110_000, step=1_000, key="inp_income_dep")
                guardian_monthly_debt = st.number_input("Guardian Monthly Debt ($)", min_value=0, value=400, step=50, key="inp_debt_dep")
            with right:
                guardian_credit_score = st.number_input("Guardian Credit Score", min_value=300, max_value=850, value=720, step=1, key="inp_cs_dep")
                property_value = st.number_input("Property Value ($)", min_value=50_000, value=300_000, step=5_000, key="inp_prop_dep")
                down_payment = st.number_input("Down Payment ($)", min_value=0, value=30_000, step=5_000, key="inp_down_dep")
                interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=25.0, value=6.8, step=0.1, key="inp_rate_dep")
                term_months = st.number_input("Term (months)", min_value=60, max_value=480, value=360, step=12, key="inp_term_dep")

        submitted = st.form_submit_button("Submit", use_container_width=True)

        if submitted:
            # -------- Map inputs based on borrower type --------
            if bt == "Independent":
                income_for_calc = annual_income
                monthly_debt_for_calc = monthly_debt
                used_credit_score = credit_score
                used_employment_years = employment_years
            else:
                income_for_calc = guardian_annual_income
                monthly_debt_for_calc = guardian_monthly_debt
                used_credit_score = guardian_credit_score
                used_employment_years = guardian_employment_years

            # -------- Derived features --------
            loan_amount = max(property_value - down_payment, 0.0)
            piti = monthly_payment(loan_amount, interest_rate, term_months)  # principal+interest only
            gross_monthly_income = max(income_for_calc / 12.0, 1.0)

            dti = (monthly_debt_for_calc + piti) / gross_monthly_income   # ratio 0–1
            ltv = (loan_amount / property_value) if property_value > 0 else 0.0

            # -------- Prediction --------
            if model is None:
                st.warning("Model not found or failed to load. Run `python train_model.py` first.")
            else:
                months_employed = int(used_employment_years * 12)
                has_cosigner = 1 if bt == "Dependent" else 0

                # DTI for model (percent if training used percent)
                dti_for_model = (dti * 100.0) if dti_is_ratio else dti

                X_user = build_feature_row(
                    age=age,
                    income=income_for_calc,
                    loan_amount=loan_amount,
                    credit_score=used_credit_score,
                    months_employed=months_employed,
                    interest_rate=interest_rate,
                    term_months=term_months,
                    dti_value=dti_for_model,
                    has_cosigner=has_cosigner,
                )

                with st.spinner("Scoring your risk..."):
                    try:
                        prob_default = float(model.predict_proba(X_user)[0, 1])
                    except Exception as e:
                        st.exception(e)
                        st.stop()

                # thresholds from sidebar
                if prob_default < low_thr:
                    risk_label = "Low Risk ✅"
                elif prob_default < med_thr:
                    risk_label = "Medium Risk ⚠️"
                else:
                    risk_label = "High Risk ❌"

                # ----- Save baseline to session_state so What-if can update reactively -----
                st.session_state["baseline"] = {
                    "bt": bt,
                    "age": age,
                    "income": income_for_calc,
                    "monthly_debt": monthly_debt_for_calc,
                    "credit_score": used_credit_score,
                    "months_employed": months_employed,
                    "interest_rate": interest_rate,
                    "term_months": term_months,
                    "property_value": property_value,
                    "down_payment": down_payment,
                    "loan_amount": loan_amount,
                    "piti": piti,
                    "gross_monthly_income": gross_monthly_income,
                    "dti_ratio": dti,
                    "dti_for_model": dti_for_model,
                    "ltv": ltv,
                    "has_cosigner": has_cosigner,
                    "prob_default": prob_default,
                    "risk_label": risk_label,
                }

                # Build CSV & flag ready
                result_row = {
                    "BorrowerType": bt, "Age": age, "Income": income_for_calc,
                    "LoanAmount": loan_amount, "CreditScore": used_credit_score,
                    "MonthsEmployed": months_employed, "InterestRate": interest_rate,
                    "LoanTerm": term_months, "DTI_ratio": dti, "DTI_for_model": dti_for_model,
                    "HasCoSigner": has_cosigner, "LTV": ltv,
                    "ProbDefault": prob_default, "RiskLabel": risk_label
                }
                st.session_state["scenario_csv"] = pd.DataFrame([result_row]).to_csv(index=False).encode()
                st.session_state["scenario_ready"] = True

# ======== Render baseline + Suggestions + What-if OUTSIDE the form, reactively ========
if "baseline" in st.session_state:
    base = st.session_state["baseline"]

    st.success("Form Submitted Successfully!")
    st.write(f"**Loan amount:** ${base['loan_amount']:,.0f} · **LTV:** {base['ltv']:.2f} · **DTI:** {base['dti_ratio']:.2f}")
    st.caption(f"(Using credit score: {base['credit_score']}, employment yrs: {base['months_employed'] // 12})")

    st.divider()
    st.write(f"**Predicted Default Probability:** {base['prob_default']:.2%}")
    st.progress(min(max(base["prob_default"], 0.0), 1.0))
    st.subheader(f"Risk Category: {base['risk_label']}")

    # Guidance
    tips = []
    if base["dti_ratio"] > 0.43:
        tips.append("DTI is high; consider reducing monthly debt or increasing down payment.")
    if base["property_value"] and (base["loan_amount"] / base["property_value"]) > 0.80:
        tips.append("LTV > 0.80 may trigger mortgage insurance.")
    if base["credit_score"] < 620:
        tips.append("Low credit score; conventional eligibility may be limited.")
    if tips:
        st.markdown("**Suggestions:**")
        for t in tips:
            st.markdown(f"- {t}")

    # -------- What-if analysis (reactive) --------
    with st.expander("What-if analysis", expanded=True):
        bump_down = st.number_input("Extra down payment ($)", 0, 100_000, 5_000, 1_000, key="whatif_bump_down")
        bump_rate = st.number_input("Rate reduction (percentage points)", 0.0, 5.0, 0.5, 0.1, key="whatif_bump_rate")
        bump_score = st.number_input("Credit score increase", 0, 200, 20, 5, key="whatif_bump_score")

        # Recompute scenario against saved baseline
        loan_amount_B = max(base["property_value"] - (base["down_payment"] + bump_down), 0.0)
        interest_rate_B = max(base["interest_rate"] - bump_rate, 0.0)
        credit_score_B = min(base["credit_score"] + bump_score, 850)

        piti_B = monthly_payment(loan_amount_B, interest_rate_B, base["term_months"])
        dti_B = (base["monthly_debt"] + piti_B) / base["gross_monthly_income"]
        dti_for_model_B = (dti_B * 100.0) if dti_is_ratio else dti_B

        if model is not None:
            X_user_B = build_feature_row(
                age=base["age"],
                income=base["income"],
                loan_amount=loan_amount_B,
                credit_score=credit_score_B,
                months_employed=base["months_employed"],
                interest_rate=interest_rate_B,
                term_months=base["term_months"],
                dti_value=dti_for_model_B,
                has_cosigner=base["has_cosigner"],
            )
            prob_B = float(model.predict_proba(X_user_B)[0, 1]) 

            st.write(
                f"Baseline: **{base['prob_default']:.2%}** → Scenario: **{prob_B:.2%}** "
                f"(**Δ {(prob_B - base['prob_default']):+.2%}**)"
            )
        else:
            st.warning("Model not found. Run `python train_model.py` first.")

# ========== Render the download button OUTSIDE the form ==========
if st.session_state.get("scenario_ready") and "scenario_csv" in st.session_state:
    download_placeholder.download_button(
        "Download scenario CSV",
        st.session_state["scenario_csv"],
        file_name="loan_scenario.csv",
        mime="text/csv",
        key="dl_scenario_csv"
    )
