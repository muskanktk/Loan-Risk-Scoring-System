import streamlit as st


def render_independent_form(left, right):
    st.subheader("Independent Borrower Form")

    with left:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1, key="inp_age_ind")
        annual_income = st.number_input("Annual Income ($)", min_value=0, value=75000, step=1000, key="inp_income_ind")
        monthly_debt = st.number_input("Monthly Non-Housing Debt ($)", min_value=0, value=500, step=50, key="inp_debt_ind")
        employment_years = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5, step=1, key="inp_empyrs_ind")
        num_credit_lines = st.number_input("Number of Open Credit Lines", min_value=0, value=3, step=1, key="inp_ncl_ind")

    with right:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720, step=1, key="inp_cs_ind")
        property_value = st.number_input("Property Value ($)", min_value=50000, value=350000, step=5000, key="inp_prop_ind")
        down_payment = st.number_input("Down Payment ($)", min_value=0, value=70000, step=5000, key="inp_down_ind")
        interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=15.0, value=6.5, step=0.1, key="inp_rate_ind")
        term_months = st.number_input("Term (months)", min_value=60, max_value=480, value=360, step=12, key="inp_term_ind")

    return {
        "age": age,
        "income": annual_income,
        "monthly_debt": monthly_debt,
        "employment_years": employment_years,
        "num_credit_lines": num_credit_lines,
        "credit_score": credit_score,
        "property_value": property_value,
        "down_payment": down_payment,
        "interest_rate": interest_rate,
        "term_months": term_months,
    }
