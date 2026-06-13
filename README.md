# RestaurantIQ — AI-Powered Restaurant Intelligence & Business Analytics Platform

> Analyzes restaurant performance, customer preferences, and business success factors using Machine Learning and interactive dashboards — built on multi-country restaurant data to help stakeholders make smarter, data-driven decisions.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

---

## Problem Statement

Restaurants fail at an alarming rate — not because of bad food, but because of blind spots in business intelligence. Owners lack actionable insights on what drives ratings, which cuisines succeed in which cities, and whether their pricing strategy is aligned with customer expectations.

This project builds a full-stack analytics platform that processes multi-country restaurant data to deliver predictive ratings, success scoring, customer preference mapping, and geospatial trend analysis — giving restaurant owners and investors the intelligence they need before making critical decisions.

---

## Dataset

| Property | Detail |
|---|---|
| Total records | Multi-country restaurant transactions |
| Coverage | Multiple countries and cities globally |
| Features | Restaurant name, city, country, cuisines, average cost, ratings, votes, price range, online delivery, table booking |
| Target variable | Aggregate Rating (Regression) |
| Key challenge | Multi-currency cost normalization, sparse ratings, categorical cuisine encoding |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 |
| Web Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| ML Models | Scikit-Learn (Random Forest, Gradient Boosting) |
| Visualization | Plotly, Matplotlib, Seaborn |
| Geospatial | Folium |
| Deployment | Streamlit Community Cloud / Render / Hugging Face Spaces |

---

## Platform Modules

```
RestaurantIQ Platform
        │
        ├── 📊 Dashboard
        │     ├── Key business metrics overview
        │     ├── Interactive KPI cards
        │     └── High-level restaurant performance summary
        │
        ├── 🔍 Data Exploration & Preprocessing
        │     ├── Data cleaning and validation pipeline
        │     ├── Missing value handling
        │     └── Dataset overview and profiling
        │
        ├── 📈 Descriptive Analysis
        │     ├── Cuisine-wise performance breakdown
        │     ├── Rating distribution analysis
        │     ├── Cost analysis by region
        │     └── Country and city-wise insights
        │
        ├── 🗺️ Geospatial Analysis
        │     ├── Interactive restaurant location mapping (Folium)
        │     ├── Market hotspot identification
        │     └── Geographic trend visualization
        │
        ├── ⚙️ Feature Engineering
        │     ├── Automated feature creation
        │     └── Model-ready dataset generation
        │
        ├── 🧠 Customer Preference Analysis
        │     ├── Cuisine preference scoring
        │     ├── User behavior insights
        │     └── Popular restaurant category trends
        │
        ├── 🤖 Predictive Modeling
        │     ├── ML-based aggregate rating prediction
        │     └── Model performance evaluation (R², MAE, RMSE)
        │
        ├── 🏆 Success Score Engine
        │     ├── Restaurant success estimation
        │     └── Business growth indicators
        │
        └── 🔮 AI Predictor
              ├── Real-time restaurant rating prediction
              └── Business performance forecasting
```

---

## ML Workflow

```
Raw Restaurant Records (Multi-country)
        ↓
Data Cleaning & Preprocessing
  ├── Missing value imputation (median for cost, mode for categoricals)
  ├── Currency normalization across countries
  ├── Outlier treatment for cost and votes
  └── Label encoding for cuisines, city, country
        ↓
Feature Engineering
  ├── Price-to-rating ratio     → avg_cost / aggregate_rating
  ├── Popularity index          → votes / max_votes (normalized)
  ├── Delivery premium flag     → online_delivery == Yes
  ├── Booking convenience score → table_booking * price_range
  └── Cuisine diversity count   → count of cuisines offered
        ↓
Model Training
  ├── Random Forest Regressor   — baseline
  └── Gradient Boosting         — optimized
        ↓
Hyperparameter Tuning
  └── GridSearchCV / RandomizedSearchCV (cross-validated)
        ↓
Evaluation
  └── R² Score, MAE, RMSE — tracked per experiment
        ↓
Deployment
  └── Streamlit App — real-time prediction + interactive dashboards
```

---

## Model Results

All five models evaluated using 5-Fold Cross Validation — ranked by R² score (higher is better):

| Rank | Model | R² | RMSE | MAE | CV R² Mean | CV R² Std | Train Time |
|---|---|---|---|---|---|---|---|
| 🏆 1 | **Gradient Boosting** | **0.6066** | **0.3533** | **0.2578** | **0.4858** | 0.0516 | 3.2s |
| 2 | Random Forest | 0.5873 | 0.3619 | 0.2610 | 0.4643 | 0.0488 | 2.9s |
| 3 | Decision Tree | 0.5282 | 0.3869 | 0.2767 | 0.3556 | 0.1032 | 0.1s |
| 4 | Linear Regression | 0.3808 | 0.4433 | 0.3478 | 0.2795 | 0.0817 | 0.7s |
| 5 | Ridge Regression | 0.3808 | 0.4433 | 0.3478 | 0.2796 | 0.0817 | 0.2s |

> **Best Model — Gradient Boosting** with CV R² of 0.4858 ± 0.0516, indicating stable generalisation across folds. R² closer to 1.0 means better variance explanation; lower RMSE and MAE mean tighter predictions.

---

## Feature Importance (Top 6)

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | Average cost for two | 0.201 |
| 2 | Votes | 0.178 |
| 3 | Price range | 0.154 |
| 4 | Online delivery availability | 0.121 |
| 5 | Cuisine diversity count | 0.094 |
| 6 | Table booking availability | 0.073 |

---

## Key Platform Outcomes

- **Rating Prediction** — Estimate a restaurant's aggregate rating based on its business profile before it goes live
- **Success Score Engine** — Composite scoring using votes, cost, delivery, and booking features to rank business viability
- **Customer Preference Map** — Identify top-performing cuisines and price segments by city and country
- **Geospatial Intelligence** — Pinpoint high-density restaurant markets and underserved geographic zones
- **Business Decision Support** — Help investors, franchisees, and operators make data-backed expansion decisions

---

## Business Impact

- **Market entry intelligence:** Cuisine and location analysis helps investors identify underserved markets before committing capital
- **Operational benchmarking:** Success scores allow restaurant owners to benchmark against top performers in their city
- **Customer alignment:** Preference analysis surfaces the most-demanded cuisine types and price points per region
- **Expansion readiness:** Geospatial hotspot mapping identifies ideal new locations based on density and competition data

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd RestaurantIQ

# Install dependencies
pip install -r requirements.txt

# Launch the platform
streamlit run app.py
```

---

## Project Structure

```
RestaurantIQ/
├── app.py                        # Main Streamlit entry point
├── pages/                        # Multi-page Streamlit modules
│   ├── dashboard.py
│   ├── exploration.py
│   ├── descriptive.py
│   ├── geospatial.py
│   ├── feature_engineering.py
│   ├── customer_preference.py
│   ├── predictive_modeling.py
│   ├── success_score.py
│   └── ai_predictor.py
├── src/                          # Core processing logic
├── data/                         # Raw and processed datasets
├── models/                       # Trained ML model artifacts
├── style.css                     # Custom UI theming
├── requirements.txt
└── README.md
```

---

## Requirements

```
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
plotly==5.17.0
folium==0.14.0
streamlit-folium==0.15.0
matplotlib==3.7.2
seaborn==0.12.2
```

---

## Author

**Elansurya Karthikeyan** — Aspiring Data Scientist | ML · Python · SQL · Streamlit

*Cognifyz Data Analytics Internship Project — 2026*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/elansurya-karthikeyan-3b6636380)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat-square&logo=github)](https://github.com/Elansurya)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow?style=flat-square)](https://huggingface.co/spaces/Elansurya)
