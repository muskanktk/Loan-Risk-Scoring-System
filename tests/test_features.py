from src.features import build_row
def test_build_row_units():
    r = build_row(30, 60000, 200000, 720, 24, 6.5, 360, 0.35, 0)
    assert list(r.columns) == ["Age","Income","LoanAmount","CreditScore","MonthsEmployed",
                               "InterestRate","LoanTerm","DTIRatio","HasCoSigner"]
    assert 34.9 < r["DTIRatio"][0] < 35.1  # ratio→percent check
