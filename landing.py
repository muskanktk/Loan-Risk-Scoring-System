import base64
from pathlib import Path

import streamlit as st


def render_landing_page():
    root = Path(__file__).parent
    logo_data = base64.b64encode((root / "assets" / "logo.png").read_bytes()).decode("ascii")
    independent_card = base64.b64encode(
        (root / "assets" / "independent-card.svg").read_bytes()
    ).decode("ascii")
    dependent_card = base64.b64encode(
        (root / "assets" / "dependent-card.svg").read_bytes()
    ).decode("ascii")

    st.markdown(
        f'''
        <div class="brand-header">
            <img class="brand-logo" src="data:image/png;base64,{logo_data}" alt="Loan Risk logo">
            <div class="brand-name">
                Loan Risk Scoring System
                <div class="brand-tagline">
                    Plan Smarter. Borrow Wisely. Build Your Future.
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.title("Select Your Current Financial Situation")
    st.markdown(
        '''
        <p style="
            color: #003366;
            font-size: 18px;
            font-weight: 300;
            margin-top: -10px;
            margin-bottom: 20px;
            text-align: center;
        ">
            Choose the option that best reflects your current status
        </p>
        ''',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'''
        <div class="borrower-card-grid">
            <a class="borrower-card" href="?borrower_type=Independent">
                <img src="data:image/svg+xml;base64,{independent_card}" alt="Independent borrower">
                <span class="borrower-card-label">Independent</span>
            </a>
            <a class="borrower-card" href="?borrower_type=Dependent">
                <img src="data:image/svg+xml;base64,{dependent_card}" alt="Dependent borrower">
                <span class="borrower-card-label">Dependent</span>
            </a>
        </div>
        ''',
        unsafe_allow_html=True,
    )
