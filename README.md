# Loan Risk Scoring System

Provide a what-if anaylsys with the loan amount based on indiuval status **Independent or Dependent**:

1. Collected data from Kaggle for default Loan stats based on categories
2. Trained the model to adjust the program
3. Tested the model to provide a what-if anaylsys
**Live demo:** [[https://ai-mentor-study-assistant-tvv5bhksyvyderelpq8pmh.streamlit.app/](https://ai-mentor-study-assistant-tvv5bhksyvyderelpq8pmh.streamlit.app/)](https://loan-risk-scoring-app-ztqeqbhh9cnf8ktcjdusxy.streamlit.app/)


## 🚀 Features

*  **Two Form based on Status**: Mathces your condition.
*  **Probailioy of payment or following thorugh with your plan**: OpenAI condenses content, adds examples, code, and analogies.
*  **Provding DTI and LTI risks based on them **: Provides the risk being taken the loan based on LTI and DTI based on 
*  **What-if-anylsys**: Chnages made to adjust the probaolby 


## 🛠️ Tech Stack

* **Language:** Python
* **Frameworks/Libraries:** import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

* **Tools:** Git, VS Code / Visual Studio Code


## 📦 Installation

```bash
# 1) Clone the repository
git clone https://github.com/<your-username>/AI-Mentor-Study-Assistant.git
cd AI-Mentor-Study-Assistant

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
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
# Optional if you use other services:
# YOUTUBE_API_KEY=...
```

---

## ▶Run Locally

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

## 💡 How to Use

1. **Select option** Fill form based on your indiuval information.
---

## 📂 Project Structure

```
AI-Mentor-Study-Assistant/
├─ app.py               # Streamlit entry point
├─ requirements.txt     # Python dependencies
├─ README.md            # This file
├─ .env                 # Your API keys (not committed)
├─ data/                # Uploaded PDFs / sample assets
└─ src/
   ├─ summarizer.py     # OpenAI summary helpers
   ├─ audio.py          # TTS (gTTS) utilities
   ├─ quiz.py           # Quiz generation / logic
   ├─ videos.py         # YouTube topic fetchers
   └─ utils.py          # shared helpers (parsing, I/O, etc.)
```

> Your exact files may differ—use this as a template.

---

## 🌐 Hosted App

* **Browser link:** https://loan-risk-scoring-app-ztqeqbhh9cnf8ktcjdusxy.streamlit.app/


## 🛣️ Roadmap

* [ ] Additional voice models / SSML support
* [ ] Richer simulations and spaced-repetition scheduling
* [ ] Per-topic flashcards and cloze deletions
* [ ] Multi-PDF study packs & cross-topic linking

## FAQ

Download the information doe your status in .cvs file 
## Troubleshooting

* **`ModuleNotFoundError`**: Re-run `pip install -r requirements.txt` in the active venv.
* **Streamlit won’t start**: Ensure the venv is active; try `python --version` and `which streamlit`.
* **gTTS network issues**: Ensure your network allows outbound requests; retry or switch networks.

## Contributing

Please open an issue to discuss substantial changes first.

## License
MIT License – see [LICENSE](LICENSE) for details.


## Acknowledgments



---

## Final Product

[loan_scenario (1).csv](https://github.com/user-attachments/files/21991205/loan_scenario.1.csv)
BorrowerType,Age,Income,LoanAmount,CreditScore,MonthsEmployed,InterestRate,LoanTerm,DTI_ratio,DTI_for_model,HasCoSigner,LTV,ProbDefault,RiskLabel
Independent,25,50000,315000,680,24,6.5,360,0.5498434257606818,54.98434257606818,0,0.9,0.9999999740307541,High Risk ❌
