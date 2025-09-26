
# Superstore Sales Analysis

## Overview
This project performs Exploratory Data Analysis (EDA), trains a Machine Learning model, and creates an interactive Streamlit dashboard for the Superstore sales dataset.

## Features
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA) with plots
- Random Forest Regression Model to predict Profit
- Streamlit Dashboard with interactive filters and visualizations



## Project Files
- `eda_model.py` – Enhanced EDA and Model Training
- `app.py` – Streamlit Dashboard
- `outputs/` – Saved plots and snapshots (ignored in Git)
- `models/` – Saved ML model (ignored in Git)

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/superstore-analysis.git
   cd superstore-analysis
    ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
            or
   source venv/bin/activate  # Linux/Mac
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run EDA & Model Training:

   ```bash
   python eda_model.py
   ```
5. Run the Streamlit Dashboard:

   ```bash
   streamlit run app.py
   ```

## Dataset

Download the Superstore dataset from Kaggle: [Superstore Sales Dataset](https://www.kaggle.com/datasets/juhi1994/superstore)

## Notes

* The dashboard allows filtering by Year, Region, and Category.
* KPIs include Total Sales, Total Profit, Average Discount, and Total Orders.
* All plots are interactive using Plotly.

