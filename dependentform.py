import streamlit as st


def render_dependent_form(left, right):
    st.subheader("Dependent Borrower Form")

    with left:
        age = st.number_input("Age", min_value=18, max_value=100, value=22, step=1, key="inp_age_dep")
        relation = st.selectbox("Relationship to Guardian / Co-signer", ["Parent", "Guardian", "Co-signer"], key="inp_rel_dep")
        guardian_employment_years = st.number_input("Guardian Employment Length (years)", min_value=0, max_value=50, value=8, step=1, key="inp_empyrs_dep")
        guardian_annual_income = st.number_input("Guardian Annual Income ($)", min_value=0, value=85000, step=1000, key="inp_income_dep")
        guardian_monthly_debt = st.number_input("Guardian Monthly Debt ($)", min_value=0, value=600, step=50, key="inp_debt_dep")
        num_credit_lines = st.number_input("Number of Open Credit Lines (Guardian)", min_value=0, value=4, step=1, key="inp_ncl_dep")

    with right:
        guardian_credit_score = st.number_input("Guardian Credit Score", min_value=300, max_value=850, value=740, step=1, key="inp_cs_dep")
        property_value = st.number_input("Property Value ($)", min_value=50000, value=300000, step=5000, key="inp_prop_dep")
        down_payment = st.number_input("Down Payment ($)", min_value=0, value=60000, step=5000, key="inp_down_dep")
        interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=15.0, value=6.8, step=0.1, key="inp_rate_dep")
        term_months = st.number_input("Term (months)", min_value=60, max_value=480, value=360, step=12, key="inp_term_dep")

    return {
        "age": age,
        "income": guardian_annual_income,
        "monthly_debt": guardian_monthly_debt,
        "employment_years": guardian_employment_years,
        "num_credit_lines": num_credit_lines,
        "credit_score": guardian_credit_score,
        "property_value": property_value,
        "down_payment": down_payment,
        "interest_rate": interest_rate,
        "term_months": term_months,
        "relation": relation,
    }
