import streamlit as st


def render_analysis(
    base,
    clf,
    build_feature_row,
    monthly_payment
):

    # =========================================================
    # PAGE CSS
    # =========================================================

    st.markdown(
        """
        <style>

        /* ===============================================
           PAGE
        =============================================== */

        .stApp {
            background:
                radial-gradient(
                    circle at 5% 20%,
                    rgba(210, 234, 255, 0.75),
                    transparent 24%
                ),
                radial-gradient(
                    circle at 95% 10%,
                    rgba(225, 240, 255, 0.8),
                    transparent 24%
                ),
                #f7fbff;
        }

        .block-container {
            max-width: 1500px !important;
            padding-top: 2rem !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
            padding-bottom: 3rem !important;
        }


        /* ===============================================
           MAIN TITLE
        =============================================== */

        h1 {
            color: #0b3d78 !important;
            font-size: 3rem !important;
            font-weight: 800 !important;
            line-height: 1.05 !important;
            text-align: left !important;
            margin-bottom: 0.25rem !important;
        }


        /* ===============================================
           SUBHEADINGS
        =============================================== */

        h2,
        h3,
        h4 {
            color: #0b3d78 !important;
        }


        /* ===============================================
           CONTAINERS / CARDS
        =============================================== */

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,255,255,0.94);

            border: 1px solid #d7e6f5 !important;

            border-radius: 18px !important;

            box-shadow:
                0 10px 30px
                rgba(26, 67, 110, 0.08);

            padding: 8px;
        }


        /* ===============================================
           INPUTS
        =============================================== */

        div[data-testid="stNumberInput"] label {
            color: #123f73 !important;
            font-weight: 650 !important;
        }

        div[data-baseweb="input"] {
            background: white !important;
            border-radius: 10px !important;
        }

        div[data-baseweb="input"] > div {
            background: white !important;
        }


        /* ===============================================
           METRIC CARDS
        =============================================== */

        div[data-testid="stMetric"] {
            background: #ffffff;

            border: 1px solid #d8e7f5;

            border-radius: 13px;

            padding: 1rem 0.95rem;

            box-shadow:
                0 3px 10px
                rgba(0, 45, 90, 0.04);
        }

        div[data-testid="stMetricLabel"] {
            color: #55779a !important;
            font-size: 0.82rem !important;
        }

        div[data-testid="stMetricValue"] {
            color: #073c79 !important;
            font-weight: 750 !important;
        }


        /* ===============================================
           BUTTONS
        =============================================== */

        div.stButton > button {
            background-color: #073d78;

            color: white;

            border: none;

            border-radius: 10px;

            min-height: 44px;

            font-weight: 650;
        }

        div.stButton > button:hover {
            background-color: #052d5c;
            color: white;
            border: none;
        }


        /* ===============================================
           DOWNLOAD BUTTON
        =============================================== */

        div.stDownloadButton > button {
            background: white;

            color: #073d78;

            border: 1px solid #bcd5ed;

            border-radius: 10px;

            min-height: 45px;

            font-weight: 650;

            box-shadow:
                0 4px 12px
                rgba(0, 45, 90, 0.05);
        }

        div.stDownloadButton > button:hover {
            background: #f3f8ff;
            color: #073d78;
            border-color: #8db9df;
        }


        /* ===============================================
           PROGRESS
        =============================================== */

        div[data-testid="stProgress"] > div > div > div {
            background-color: #07447e;
        }


        /* ===============================================
           ALERTS
        =============================================== */

        div[data-testid="stAlert"] {
            border-radius: 12px;
            background: #eaf4ff !important;
            border: 1px solid #9cc5e8 !important;
            color: #0b3d78 !important;
        }

        div[data-testid="stAlert"] * {
            color: #0b3d78 !important;
        }

        div[data-testid="stAlert"] svg {
            fill: #1769aa !important;
            color: #1769aa !important;
        }


        /* ===============================================
           DIVIDERS
        =============================================== */

        hr {
            border-color: #dbe7f2 !important;
            margin-top: 1rem !important;
            margin-bottom: 1rem !important;
        }


        /* ===============================================
           CAPTIONS
        =============================================== */

        .stCaption {
            color: #6683a0 !important;
        }


        /* ===============================================
           MOBILE
        =============================================== */

        @media (max-width: 900px) {

            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }

            h1 {
                font-size: 2.25rem !important;
            }

        }

        </style>
        """,
        unsafe_allow_html=True
    )


    # =========================================================
    # DEFAULT WHAT-IF VALUES
    # =========================================================

    if "whatif_bump_down" not in st.session_state:
        st.session_state["whatif_bump_down"] = 20000

    if "whatif_bump_rate" not in st.session_state:
        st.session_state["whatif_bump_rate"] = 0.5

    if "whatif_bump_score" not in st.session_state:
        st.session_state["whatif_bump_score"] = 20

    if "whatif_debt_cut" not in st.session_state:
        st.session_state["whatif_debt_cut"] = 100


    # =========================================================
    # RESET
    # =========================================================

    def reset_scenario():

        st.session_state["whatif_bump_down"] = 20000
        st.session_state["whatif_bump_rate"] = 0.5
        st.session_state["whatif_bump_score"] = 20
        st.session_state["whatif_debt_cut"] = 100


    # =========================================================
    # HEADER
    # =========================================================

    header_left, header_right = st.columns(
        [4.2, 1.1]
    )


    with header_left:

        st.title(
            "What-If Analysis"
        )

        st.markdown(
            """
            <p style="
                color:#56789a;
                font-size:1.05rem;
                margin-top:-8px;
                margin-bottom:14px;
            ">
                Adjust different factors to see how they impact your loan risk.
            </p>
            """,
            unsafe_allow_html=True
        )


    with header_right:

        st.write("")

        if (
            st.session_state.get("scenario_ready")
            and
            "scenario_csv" in st.session_state
        ):

            st.download_button(
                "⬇ Download Scenario CSV",
                st.session_state["scenario_csv"],
                file_name="loan_scenario.csv",
                mime="text/csv",
                use_container_width=True,
                key="analysis_download"
            )


    # =========================================================
    # CALCULATE SCENARIO
    # =========================================================

    # We need the values before rendering the right side.
    # The widgets themselves are created inside the left panel.


    # =========================================================
    # MAIN DASHBOARD COLUMNS
    # =========================================================

    left_side, right_side = st.columns(
        [1, 1.08],
        gap="large"
    )


    # =========================================================
    # LEFT PANEL
    # =========================================================

    with left_side:

        with st.container(
            border=True
        ):

            st.subheader(
                "🧮 Scenario Inputs"
            )

            st.caption(
                "Modify the values below to explore different scenarios."
            )

            st.write("")


            # =============================================
            # INPUT ROW 1
            # =============================================

            input1, input2 = st.columns(
                2,
                gap="large"
            )


            with input1:

                bump_down = st.number_input(
                    "Extra Down Payment ($)",
                    min_value=0,
                    max_value=300000,
                    step=1000,
                    key="whatif_bump_down"
                )


            with input2:

                bump_rate = st.number_input(
                    "Interest Rate Reduction (percentage points)",
                    min_value=0.0,
                    max_value=5.0,
                    step=0.1,
                    key="whatif_bump_rate"
                )


            # =============================================
            # INPUT ROW 2
            # =============================================

            input3, input4 = st.columns(
                2,
                gap="large"
            )


            with input3:

                bump_score = st.number_input(
                    "Credit Score Increase (+points)",
                    min_value=0,
                    max_value=200,
                    step=5,
                    key="whatif_bump_score"
                )


            with input4:

                debt_cut = st.number_input(
                    "Monthly Debt Reduction ($)",
                    min_value=0,
                    max_value=5000,
                    step=50,
                    key="whatif_debt_cut"
                )


            # =============================================
            # CURRENT LOAN DETAILS
            # =============================================

            st.divider()

            st.subheader(
                "🏠 Current Loan Details"
            )

            st.caption(
                "Your current financial profile."
            )


            current1, current2 = st.columns(
                2
            )


            current1.metric(
                "💰 Loan Amount",
                f"${base['loan_amount']:,.0f}"
            )


            current2.metric(
                "🎯 Credit Score",
                f"{int(base['credit_score'])}"
            )


            current3, current4 = st.columns(
                2
            )


            current3.metric(
                "📉 Interest Rate",
                f"{base['interest_rate']:.2f}%"
            )


            current4.metric(
                "💼 Years Employed",
                f"{base['months_employed'] / 12:.0f}"
            )


            st.write("")


            # =============================================
            # RESET BUTTON
            # =============================================

            reset_col, _ = st.columns(
                [1.2, 1.8]
            )


            with reset_col:

                st.button(
                    "↻ Reset to Default Values",
                    on_click=reset_scenario,
                    use_container_width=True,
                    key="reset_analysis"
                )


    # =========================================================
    # UPDATED CALCULATIONS
    # =========================================================

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
        monthly_debt_B
        + piti_B
    ) / base["gross_monthly_income"]


    ltv_B = (
        loan_amount_B / base["property_value"]
        if base["property_value"] > 0
        else 0.0
    )


    # =========================================================
    # MODEL PREDICTION
    # =========================================================

    prob_B = base[
        "prob_default"
    ]


    if clf is not None:

        X_user_B = build_feature_row(
            age=
                base["age"],

            income=
                base["income"],

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


    # =========================================================
    # RIGHT PANEL
    # =========================================================

    with right_side:

        with st.container(
            border=True
        ):

            st.subheader(
                "📊 Results"
            )

            st.caption(
                "Based on your current inputs."
            )


            # =================================================
            # PROBABILITY + RISK
            # =================================================

            probability_col, risk_col = st.columns(
                [1.2, 1],
                gap="medium"
            )


            # =============================================
            # PROBABILITY
            # =============================================

            with probability_col:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "**Predicted Default Probability**"
                    )

                    st.markdown(
                        f"""
                        <div style="
                            color:#073d78;
                            font-size:2.7rem;
                            font-weight:800;
                            line-height:1;
                            margin-top:12px;
                            margin-bottom:18px;
                        ">
                            {base["prob_default"]:.2%}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(
                        min(
                            max(
                                base[
                                    "prob_default"
                                ],
                                0.0
                            ),
                            1.0
                        )
                    )


            # =============================================
            # RISK CATEGORY
            # =============================================

            with risk_col:

                clean_risk = (
                    base[
                        "risk_label"
                    ]
                    .replace(
                        "❌",
                        ""
                    )
                    .replace(
                        "⚠️",
                        ""
                    )
                    .replace(
                        "✅",
                        ""
                    )
                    .strip()
                )


                if (
                    "High"
                    in base[
                        "risk_label"
                    ]
                ):

                    st.error(
                        f"### ⚠️ {clean_risk}\n\n"
                        "Based on the current financial profile, "
                        "this loan has a higher predicted risk of default."
                    )


                elif (
                    "Medium"
                    in base[
                        "risk_label"
                    ]
                ):

                    st.warning(
                        f"### ⚠️ {clean_risk}\n\n"
                        "This loan currently falls within a "
                        "moderate predicted risk range."
                    )


                else:

                    st.success(
                        f"### ✅ {clean_risk}\n\n"
                        "This loan currently has a relatively "
                        "lower predicted risk of default."
                    )


            # =================================================
            # CURRENT METRICS
            # =================================================

            st.write("")


            current_m1, current_m2, current_m3, current_m4 = (
                st.columns(
                    4
                )
            )


            current_m1.metric(
                "🏠 Loan-to-Value",
                f"{base['ltv']:.2f}"
            )


            current_m2.metric(
                "🪙 Debt-to-Income",
                f"{base['dti_ratio']:.2f}"
            )


            current_m3.metric(
                "🗓 Payment",
                f"${base['piti']:,.0f}"
            )


            current_m4.metric(
                "🎯 Credit Score",
                f"{int(base['credit_score'])}"
            )


            # =================================================
            # UPDATED SCENARIO
            # =================================================

            st.divider()

            st.subheader(
                "📈 Updated Scenario"
            )


            new_m1, new_m2, new_m3, new_m4 = (
                st.columns(
                    4
                )
            )


            new_m1.metric(
                "💵 New Payment",
                f"${piti_B:,.0f}",
                f"${piti_B - base['piti']:+,.0f}"
            )


            new_m2.metric(
                "🪙 New DTI",
                f"{dti_B:.2f}",
                f"{dti_B - base['dti_ratio']:+.2f}",
                delta_color="inverse"
            )


            new_m3.metric(
                "🏠 New LTV",
                f"{ltv_B:.2f}",
                f"{ltv_B - base['ltv']:+.2f}",
                delta_color="inverse"
            )


            new_m4.metric(
                "🎯 New Credit",
                f"{int(credit_score_B)}",
                f"+{bump_score}"
            )


            # =================================================
            # RISK COMPARISON
            # =================================================

            st.divider()

            st.subheader(
                "📊 Risk Comparison"
            )


            risk_old, risk_new = (
                st.columns(
                    2,
                    gap="medium"
                )
            )


            with risk_old:

                st.metric(
                    "Original Default Risk",
                    f"{base['prob_default']:.2%}"
                )


            delta = (
                prob_B
                - base[
                    "prob_default"
                ]
            )


            with risk_new:

                st.metric(
                    "New Default Risk",
                    f"{prob_B:.2%}",
                    f"{delta * 100:+.2f} percentage points",
                    delta_color="inverse"
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


            # =================================================
            # IMPROVEMENT MESSAGE
            # =================================================

            if (
                prob_B
                <
                base[
                    "prob_default"
                ]
            ):

                difference = (
                    base[
                        "prob_default"
                    ]
                    -
                    prob_B
                )

                st.success(
                    f"✅ Your scenario reduced predicted default risk "
                    f"from {base['prob_default']:.2%} "
                    f"to {prob_B:.2%}. "
                    f"That's an improvement of {difference:.2%}."
                )


            elif (
                prob_B
                >
                base[
                    "prob_default"
                ]
            ):

                difference = (
                    prob_B
                    -
                    base[
                        "prob_default"
                    ]
                )

                st.warning(
                    f"⚠️ Your scenario increased predicted default risk "
                    f"from {base['prob_default']:.2%} "
                    f"to {prob_B:.2%}."
                )


            else:

                st.info(
                    "The scenario did not change "
                    "the predicted default probability."
                )


            # =================================================
            # INSIGHTS
            # =================================================

            st.subheader(
                "💡 Insights"
            )


            if (
                ltv_B
                <= 0.80
            ):

                st.write(
                    "• Your loan-to-value ratio is within "
                    "a relatively strong range."
                )

            else:

                st.write(
                    "• Increasing your down payment may "
                    "improve your loan-to-value ratio."
                )


            if (
                dti_B
                <= 0.43
            ):

                st.write(
                    "• Your debt-to-income ratio is within "
                    "a relatively manageable range."
                )

            else:

                st.write(
                    "• Reducing monthly debt may improve "
                    "your debt-to-income ratio."
                )


            if (
                credit_score_B
                < 620
            ):

                st.write(
                    "• Improving your credit score may "
                    "help reduce predicted loan risk."
                )


            if (
                interest_rate_B
                <
                base[
                    "interest_rate"
                ]
            ):

                st.write(
                    "• A lower interest rate reduces your "
                    "monthly payment and can improve affordability."
                )