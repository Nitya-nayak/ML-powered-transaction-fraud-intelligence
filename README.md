# ⚡ TxIntel
### ML-powered transaction fraud intelligence

> Upload any financial dataset — TxIntel automatically detects fraud patterns, 
> analyzes behavioral anomalies, and delivers business-grade intelligence 
> that a real fraud analyst would trust.

---

## 🚀 Live Demo
👉 ## 🚀 Live Demo
👉 [Try TxIntel Live](https://ml-powered-transaction-fraud-intelligence-gjmpstpmlvqjkf3ro4r6.streamlit.app)

---

## 💡 Why TxIntel?

Most fraud detection projects show you an accuracy score and call it done.

**TxIntel is different.**

It works like a real fraud analyst:
- 🔍 Automatically understands your dataset structure
- 🧠 Builds a full ML pipeline without any configuration
- 📊 Delivers insights that actually mean something to a business
- ⚡ Flags behavioral anomalies humans would never manually catch

---

## ✨ What It Does

### 🤖 Auto Schema Detection
Automatically identifies Amount, Time, Location, Device, Transaction columns using fuzzy matching — works with **any** financial dataset, zero hardcoding.

### 📊 5 Business Visualizations
- Fraud vs Normal transaction distribution
- Do fraudsters spend more? (Amount analysis)
- Top 10 highest fraud locations
- Fraud by hour of day
- Fraud by transaction channel (ATM/Online/Branch)

### ⚙️ End-to-End ML Pipeline
- 8-feature engineering (is_weekend, is_night, is_high_amount, amount_to_balance, is_high_risk_location, and more)
- Auto preprocessing (LabelEncoder + OneHotEncoder)
- XGBoost Classifier
- IsolationForest Anomaly Detection

### 🧠 Smart Fraud Insights
- **Behavioral Baseline Anomaly** — every account has a spending personality. Flags transactions 3σ above the account's own average — not just a global threshold
- **Transaction Velocity** — detects rapid-fire transactions under 60 seconds on the same account. A human can't be in two cities in 44 seconds.
- **Risk Scoring** — every transaction gets a fraud probability score (0–100%)
- **Anomaly Flagging** — IsolationForest marks statistically rare transactions as suspicious

### ☁️ Cloud-First Architecture
- Multi-user support via email authentication
- Supabase Storage — every dataset backed up to cloud
- PostgreSQL — full audit trail of uploads, profiles, and analysis

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| ML | XGBoost, IsolationForest |
| Database | Supabase (PostgreSQL) |
| Storage | Supabase Storage |
| Data | Pandas, NumPy |
| Visualization | Plotly |
| Schema Detection | FuzzyWuzzy |
| Preprocessing | Scikit-learn |

---

## 📂 Project Structure

TxIntel/
│
├── app.py                  
│
├── requirements.txt        
│
├── .env.example            
│
├── .gitignore              
│
└── README.md               
