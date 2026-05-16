# Graph-Based SBM + CTGAN Synthetic IoT Attack Generation

This project generates synthetic IoT network traffic using:

- Graph fingerprinting of attack flows
- Degree-Controlled Stochastic Block Model (DC-SBM)
- CTGAN-based synthetic flow generation (SDV)

Dataset:
NF-ToN-IoT

---

# 🚀 Setup Instructions

## 1. Install Python
Use Python 3.10 only.

---

## 2. Clone repository

```bash
git clone https://github.com/fatimabazzoun17/graph_SBM_CTGAN.git
cd graph_SBM_CTGAN

3. Create virtual environment
Windows
python -m venv venv
venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt


▶    Run Pipeline (IMPORTANT ORDER)
Step 1: Fingerprinting attack graphs
python step1code.py
Step 2: DC-SBM synthetic graph generation
python step2code.py
Step 3: CTGAN training + synthetic flow generation
python step3code.py