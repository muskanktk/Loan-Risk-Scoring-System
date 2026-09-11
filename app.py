import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from independentform import render_independent_form
from dependentform import render_dependent_form
from landing import render_landing_page
from analysis import render_analysis

# ================= Page config FIRST =================
st.set_page_config(
    page_title="Loan Default Risk Scoring System",
    layout="wide"
)


@st.dialog(" ")
def show_submission_dialog():
    st.markdown(
        """
        <div style="
            color: #003366;
            font-size: 1.1rem;
            font-weight: 600;
            text-align: center;
            padding: 0.75rem 0;
        ">
            Your form was submitted successfully.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ================= Custom CSS =================
st.markdown(
    """
    <style>

        /* ---------- PAGE ---------- */
        .stApp {
            background-color: #e6f2ff;
        }

        /* ---------- WEBSITE BRAND / LOGO ---------- */
        .brand-header {
            position: fixed;
            top: 55px;
            left: 30px;

            display: flex;
            align-items: center;
            gap: 10px;

            width: auto;
            margin: 0;

            z-index: 999;
        }

        .brand-logo {
            width: 52px;
            height: 52px;
            object-fit: contain;
        }

        .brand-name {
            color: #003366;
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1;
            white-space: nowrap;
        }

        .brand-tagline {
            margin-top: 5px;
            color: #1769aa;
            font-size: 0.78rem;
            font-weight: 500;
            line-height: 1.2;
            white-space: nowrap;
        }

        .borrower-card-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin: 0 auto 1rem;
            max-width: 760px;
        }

        .borrower-card {
            display: block;
            overflow: hidden;
            border: 2px solid transparent;
            border-radius: 14px;
            background: #ffffff;
            box-shadow: 0 4px 14px rgba(0, 51, 102, 0.12);
            color: #003366 !important;
            text-decoration: none !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        .borrower-card:hover {
            border-color: #1769aa;
            box-shadow: 0 7px 20px rgba(0, 51, 102, 0.2);
            transform: translateY(-2px);
        }

        .borrower-card img {
            display: block;
            width: 100%;
            aspect-ratio: 16 / 9;
            object-fit: cover;
        }

        .borrower-card-label {
            display: block;
            padding: 0.75rem 1rem;
            font-size: 1.05rem;
            font-weight: 700;
            text-align: center;
        }

        /* ---------- BORROWER FORM WINDOW ---------- */
        div[data-testid="stForm"] {
            position: fixed;
            top: 145px;
            left: 50%;
            z-index: 1000;
            width: min(900px, 92vw);
            max-height: calc(100vh - 165px);
            overflow-y: auto;
            padding: 1.5rem;
            border: 2px solid #1769aa;
            border-radius: 16px;
            background: #ffffff;
            box-shadow: 0 18px 50px rgba(0, 34, 68, 0.28);
            transform: translateX(-50%);
        }

        @media (max-width: 640px) {
            .borrower-card-grid {
                grid-template-columns: 1fr;
            }

            div[data-testid="stForm"] {
                top: 125px;
                max-height: calc(100vh - 140px);
                padding: 1rem;
            }
        }

        /* ---------- BUTTONS ---------- */
        div.stButton > button:first-child {
            background-color: #003366;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5em 1.25em;
            font-weight: 600;
        }

        div.stButton > button:first-child:hover {
            background-color: #002244;
            color: white;
        }

        /* ---------- MAIN TITLE ---------- */
        h1 {
            color: #003366 !important;
            font-size: 2rem !important;
            font-weight: 700 !important;
            text-align: center;
            white-space: nowrap;
            margin-bottom: 0.5rem !important;
        }

        /* ---------- RISK COLORS ---------- */
        .risk-low {
            color: green;
            font-weight: bold;
        }

        .risk-medium {
            color: orange;
            font-weight: bold;
        }

        .risk-high {
            color: red;
            font-weight: bold;
        }

    </style>
    """,
    unsafe_allow_html=True
)

# ================= Page selection =================
selected_borrower = st.query_params.get("borrower_type")
if selected_borrower in {"Independent", "Dependent"}:
    if selected_borrower != st.session_state.get("borrower_type"):
        st.session_state["form_submitted"] = False
    st.session_state["borrower_type"] = selected_borrower

if "borrower_type" not in st.session_state:
    render_landing_page()
    st.stop()

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            display: none !important;
        }

        div[data-testid="stForm"] {
            top: 60px;
            max-height: calc(100vh - 84px);
        }

    </style>
    """,
    unsafe_allow_html=True,
)

if st.session_state.get("form_submitted"):
    st.markdown(
        """
        <style>
            div[data-testid="stForm"] {
                display: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ================= Model loader =================
@st.cache_resource
def load_model():

    model_path = (
        Path(__file__).parent
        / "models"
        / "risk_model.pkl"
    )

    try:

        obj = joblib.load(model_path)

        if isinstance(obj, dict) and "model" in obj:
            return obj

        return {
            "model": obj,
            "feature_names": [
                "Age",
                "Income",
                "LoanAmount",
                "CreditScore",
                "MonthsEmployed",
                "NumCreditLines",
                "InterestRate",
                "LoanTerm",
                "DTIRatio",
                "HasCoSigner"
            ]
        }

    except Exception as e:

        st.error(
            f"Model loading failed: {str(e)}"
        )

        st.info(
            "Please run `python train_model.py` first to train the model."
        )

        return None


model_obj = load_model()

clf = (
    model_obj["model"]
    if model_obj
    else None
)

FEATURE_NAMES = (
    model_obj["feature_names"]
    if model_obj
    else []
)


# ================= Monthly Payment =================
def monthly_payment(
    principal: float,
    annual_rate_pct: float,
    n_months: int
) -> float:

    if principal <= 0:
        return 0.0

    r = (
        annual_rate_pct
        / 100.0
        / 12.0
    )

    if r == 0:
        return principal / n_months

    return principal * (
        r * (1 + r) ** n_months
    ) / (
        (1 + r) ** n_months - 1
    )


# ================= Feature Builder =================
def build_feature_row(
    age,
    income,
    loan_amount,
    credit_score,
    months_employed,
    num_credit_lines,
    interest_rate,
    term_months,
    dti_ratio,
    has_cosigner
):

    row = pd.DataFrame([
        {
            "Age": int(age),
            "Income": float(income),
            "LoanAmount": float(loan_amount),
            "CreditScore": float(credit_score),
            "MonthsEmployed": int(months_employed),
            "NumCreditLines": int(num_credit_lines),
            "InterestRate": float(interest_rate),
            "LoanTerm": int(term_months),
            "DTIRatio": float(dti_ratio),
            "HasCoSigner": int(has_cosigner)
        }
    ])

    if FEATURE_NAMES:
        return row[[*FEATURE_NAMES]]

    return row


# ================= Input Validation =================
def validate_loan_inputs(
    age,
    income,
    loan_amount,
    credit_score,
    employment_years,
    interest_rate,
    property_value,
    down_payment
):

    errors = []

    # Loan-to-value ratio
    if property_value > 0:

        ltv = loan_amount / property_value

        if ltv > 0.95:
            errors.append(
                "❌ Loan-to-value ratio should typically be below 95%"
            )

        if ltv > 1.0:
            errors.append(
                "❌ Loan amount cannot exceed property value"
            )

    # Debt-to-income estimate
    if income > 0:

        monthly_pmt = monthly_payment(
            loan_amount,
            interest_rate,
            360
        )

        estimated_dti = (
            monthly_pmt
            / (income / 12)
        )

        if estimated_dti > 0.5:
            errors.append(
                "⚠️ Estimated debt-to-income ratio appears high (above 50%)"
            )

    # Employment consistency
    if employment_years == 0 and income > 50000:

        errors.append(
            "⚠️ If unemployed, income should typically be lower"
        )

    # Credit score
    if credit_score < 300 or credit_score > 850:

        errors.append(
            "❌ Credit score should be between 300 and 850"
        )

    # Interest rate
    if interest_rate > 15:

        errors.append(
            "⚠️ Interest rate seems unusually high"
        )

    # Age
    if age < 18:

        errors.append(
            "❌ Borrower must be at least 18 years old"
        )

    return errors


# ================= Sidebar =================
st.sidebar.header("Scoring Options")

low_thr = st.sidebar.slider(
    "Low → Medium threshold",
    0.00,
    0.50,
    0.15,
    0.01,
    key="opt_low_thr"
)

med_thr = st.sidebar.slider(
    "Medium → High threshold",
    0.30,
    0.90,
    0.35,
    0.01,
    key="opt_med_thr"
)


# ================= Main Form =================
if "borrower_type" in st.session_state:

    bt = st.session_state["borrower_type"]

    with st.form(
        "borrower_form",
        clear_on_submit=False
    ):

        left, right = st.columns(2)

        if bt == "Independent":
            form_data = render_independent_form(left, right)
        else:
            form_data = render_dependent_form(left, right)


        # ================= Submit =================
        submitted = st.form_submit_button(
            "Submit",
            use_container_width=True
        )

        age = form_data["age"]
        income_for_calc = form_data["income"]
        monthly_debt_for_calc = form_data["monthly_debt"]
        used_credit_score = form_data["credit_score"]
        used_employment_years = form_data["employment_years"]
        property_value = form_data["property_value"]
        down_payment = form_data["down_payment"]
        interest_rate = form_data["interest_rate"]
        term_months = form_data["term_months"]
        num_credit_lines = form_data["num_credit_lines"]

        if submitted:

            # Map inputs based on borrower type
            # ================= Derived Features =================

            loan_amount = max(
                property_value - down_payment,
                0.0
            )

            piti = monthly_payment(
                loan_amount,
                interest_rate,
                term_months
            )

            gross_monthly_income = max(
                income_for_calc / 12.0,
                1.0
            )

            dti = (
                monthly_debt_for_calc + piti
            ) / gross_monthly_income

            ltv = (
                loan_amount / property_value
                if property_value > 0
                else 0.0
            )


            # ================= Validation =================

            validation_errors = validate_loan_inputs(
                age,
                income_for_calc,
                loan_amount,
                used_credit_score,
                used_employment_years,
                interest_rate,
                property_value,
                down_payment
            )

            if validation_errors:

                for error in validation_errors:
                    st.error(error)

                if any(
                    "❌" in error
                    for error in validation_errors
                ):
                    st.stop()


            # ================= Prediction =================

            if clf is None:

                st.warning(
                    "Model not found or failed to load. "
                    "Run `python train_model.py` first."
                )

            else:

                months_employed = int(
                    used_employment_years * 12
                )

                has_cosigner = (
                    1
                    if bt == "Dependent"
                    else 0
                )

                X_user = build_feature_row(
                    age=age,
                    income=income_for_calc,
                    loan_amount=loan_amount,
                    credit_score=used_credit_score,
                    months_employed=months_employed,
                    num_credit_lines=num_credit_lines,
                    interest_rate=interest_rate,
                    term_months=term_months,
                    dti_ratio=dti,
                    has_cosigner=has_cosigner
                )

                with st.spinner(
                    "Scoring your risk..."
                ):

                    try:

                        prob_default = float(
                            clf.predict_proba(
                                X_user
                            )[0, 1]
                        )

                    except Exception as e:

                        st.error(
                            f"Prediction error: {str(e)}"
                        )

                        st.stop()


                # ================= Risk Category =================

                if prob_default < low_thr:

                    risk_label = "Low Risk ✅"
                    risk_class = "risk-low"

                elif prob_default < med_thr:

                    risk_label = "Medium Risk ⚠️"
                    risk_class = "risk-medium"

                else:

                    risk_label = "High Risk ❌"
                    risk_class = "risk-high"


                # ================= Save Baseline =================

                st.session_state["baseline"] = {

                    "bt": bt,

                    "age": age,

                    "income": income_for_calc,

                    "monthly_debt":
                        monthly_debt_for_calc,

                    "credit_score":
                        used_credit_score,

                    "months_employed":
                        months_employed,

                    "interest_rate":
                        interest_rate,

                    "term_months":
                        term_months,

                    "property_value":
                        property_value,

                    "down_payment":
                        down_payment,

                    "num_credit_lines":
                        num_credit_lines,

                    "loan_amount":
                        loan_amount,

                    "piti":
                        piti,

                    "gross_monthly_income":
                        gross_monthly_income,

                    "dti_ratio":
                        dti,

                    "ltv":
                        ltv,

                    "has_cosigner":
                        has_cosigner,

                    "prob_default":
                        prob_default,

                    "risk_label":
                        risk_label,

                    "risk_class":
                        risk_class
                }


                # ================= CSV =================

                result_row = {

                    "BorrowerType": bt,

                    "Age": age,

                    "Income":
                        income_for_calc,

                    "LoanAmount":
                        loan_amount,

                    "CreditScore":
                        used_credit_score,

                    "MonthsEmployed":
                        months_employed,

                    "NumCreditLines":
                        num_credit_lines,

                    "InterestRate":
                        interest_rate,

                    "LoanTerm":
                        term_months,

                    "DTI_ratio":
                        dti,

                    "HasCoSigner":
                        has_cosigner,

                    "LTV":
                        ltv,

                    "ProbDefault":
                        prob_default,

                    "RiskLabel":
                        risk_label
                }

                st.session_state[
                    "scenario_csv"
                ] = pd.DataFrame(
                    [result_row]
                ).to_csv(
                    index=False
                ).encode()

                st.session_state[
                    "scenario_ready"
                ] = True
                st.session_state["form_submitted"] = True
                st.session_state["show_success_message"] = True
                st.session_state["submission_dialog_shown"] = False
                st.rerun()


# =========================================================
# RESULTS
# =========================================================

if "baseline" in st.session_state:

    base = st.session_state["baseline"]

    if (
        st.session_state.get("show_success_message", False)
        and not st.session_state.get("submission_dialog_shown", False)
    ):
        st.session_state["submission_dialog_shown"] = True
        show_submission_dialog()

    render_analysis(
        base=base,
        clf=clf,
        build_feature_row=build_feature_row,
        monthly_payment=monthly_payment,
    )
    st.stop()

    st.write(
        f"**Loan amount:** "
        f"${base['loan_amount']:,.0f} · "
        f"**LTV:** {base['ltv']:.2f} · "
        f"**DTI:** {base['dti_ratio']:.2f}"
    )

    st.caption(
        f"(Using credit score: "
        f"{base['credit_score']}, "
        f"employment yrs: "
        f"{base['months_employed'] // 12})"
    )

    st.divider()

    st.write(
        f"**Predicted Default Probability:** "
        f"{base['prob_default']:.2%}"
    )

    st.progress(
        min(
            max(
                base["prob_default"],
                0.0
            ),
            1.0
        )
    )

    st.markdown(
        f'''
        <p class="{base["risk_class"]}">
            Risk Category:
            {base["risk_label"]}
        </p>
        ''',
        unsafe_allow_html=True
    )


    # ================= Suggestions =================

    tips = []

    if base["dti_ratio"] > 0.43:

        tips.append(
            "DTI is high; consider reducing monthly debt "
            "or increasing down payment."
        )

    if base["ltv"] > 0.80:

        tips.append(
            "LTV > 0.80 may trigger mortgage insurance."
        )

    if base["credit_score"] < 620:

        tips.append(
            "Low credit score; conventional eligibility "
            "may be limited."
        )

    if base["interest_rate"] > 8.0:

        tips.append(
            "Interest rate seems high; consider improving "
            "credit score or shopping around."
        )

    if tips:

        st.markdown(
            "**Suggestions:**"
        )

        for t in tips:
            st.markdown(
                f"- {t}"
            )


    # =====================================================
    # WHAT-IF ANALYSIS
    # =====================================================

    with st.container(border=True):

        st.caption(
            "DTI treated as RATIO (0–1) "
            "in both training and inference."
        )

        c1, c2 = st.columns(2)

        with c1:

            bump_down = st.number_input(
                "Extra down payment ($)",
                0,
                300000,
                20000,
                1000,
                key="whatif_bump_down"
            )

            bump_rate = st.number_input(
                "Rate reduction (percentage points)",
                0.0,
                5.0,
                0.5,
                0.1,
                key="whatif_bump_rate"
            )

        with c2:

            bump_score = st.number_input(
                "Credit score increase (+points)",
                0,
                200,
                20,
                5,
                key="whatif_bump_score"
            )

            debt_cut = st.number_input(
                "Monthly debt reduction ($)",
                0,
                5000,
                100,
                50,
                key="whatif_debt_cut"
            )


        loan_amount_B = max(
            base["property_value"]
            - (
                base["down_payment"]
                + bump_down
            ),
            0.0
        )

        interest_rate_B = max(
            base["interest_rate"]
            - bump_rate,
            0.1
        )

        credit_score_B = min(
            base["credit_score"]
            + bump_score,
            850
        )

        piti_B = monthly_payment(
            loan_amount_B,
            interest_rate_B,
            base["term_months"]
        )

        monthly_debt_B = max(
            base["monthly_debt"]
            - debt_cut,
            0.0
        )

        dti_B = (
            monthly_debt_B + piti_B
        ) / base[
            "gross_monthly_income"
        ]

        ltv_B = (
            loan_amount_B
            / base["property_value"]
            if base["property_value"] > 0
            else 0.0
        )


        if clf is not None:

            X_user_B = build_feature_row(

                age=base["age"],

                income=base["income"],

                loan_amount=
                    loan_amount_B,

                credit_score=
                    credit_score_B,

                months_employed=
                    base["months_employed"],

                num_credit_lines=
                    base["num_credit_lines"],

                interest_rate=
                    interest_rate_B,

                term_months=
                    base["term_months"],

                dti_ratio=
                    dti_B,

                has_cosigner=
                    base["has_cosigner"]
            )

            try:

                prob_B = float(
                    clf.predict_proba(
                        X_user_B
                    )[0, 1]
                )

            except Exception as e:

                st.error(
                    f"What-if analysis failed: {str(e)}"
                )

                prob_B = base[
                    "prob_default"
                ]


            # ================= Metrics =================

            st.markdown(
                "**Scenario metrics:**"
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "P&I (new)",
                f"${piti_B:,.0f}/mo",
                f"${piti_B - base['piti']:+.0f}"
            )

            m2.metric(
                "DTI (new)",
                f"{dti_B:.2f}",
                f"{dti_B - base['dti_ratio']:+.2f}"
            )

            m3.metric(
                "LTV (new)",
                f"{ltv_B:.2f}",
                f"{ltv_B - base['ltv']:+.2f}"
            )

            m4.metric(
                "Credit (new)",
                f"{int(credit_score_B)}",
                f"+{bump_score}"
            )


            delta = (
                prob_B
                - base["prob_default"]
            )

            st.write(
                f"Baseline: "
                f"**{base['prob_default'] * 100:.2f}%** "
                f"→ Scenario: "
                f"**{prob_B * 100:.2f}%** "
                f"(**Δ {delta * 100:+.2f}%**)"
            )

            st.progress(
                min(
                    max(
                        prob_B,
                        0.0
                    ),
                    1.0
                )
            )

        else:

            st.warning(
                "Model not found. "
                "Run `python train_model.py` first."
            )

