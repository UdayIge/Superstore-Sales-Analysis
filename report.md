# Superstore Sales Analysis and Profit Prediction

## 1. Project Title

**Superstore Sales Analysis and Profit Prediction Using Python**

## 2. Project Duration

**Timeline:** 1st September 2025 - 25th September 2025

## 3. Introduction

The retail industry requires insights into sales, customer behavior, and profit trends to make informed decisions. The Superstore dataset provides transactional sales data including customer, product, sales, profit, and regional information. This project aims to analyze this data, predict profit, and provide an interactive dashboard for visualization and decision-making.

## 4. Objectives

* Clean and preprocess raw sales data.
* Perform Exploratory Data Analysis (EDA) to identify patterns.
* Train a Random Forest Regressor to predict profit.
* Develop an interactive Streamlit dashboard with KPIs and visualizations.

## 5. Dataset

* **Source:** Kaggle – [Superstore Sales Dataset](https://www.kaggle.com/datasets/juhi1994/superstore)
* **Size:** ~10,000 records
* **Columns:** Order ID, Customer ID, Product ID, Product Name, Category, Sub-Category, Sales, Quantity, Discount, Profit, Order Date, Ship Date, Region, Segment
* **Data Types:** Numerical, Categorical, Date/Time

## 6. Methodology

### Step 1: Data Preprocessing

* Dropped irrelevant columns: Row ID, Country, Postal Code
* Handled missing values and duplicates
* Converted Order Date and Ship Date to datetime objects
* Engineered new features:

  * Order Year, Order Month, Order Weekday
  * Order to Ship Time
* Encoded categorical variables using One-Hot and Frequency Encoding

### Step 2: Exploratory Data Analysis (EDA)

* Distribution plots for Sales, Profit, Discount
* Time-series analysis of Sales & Profit over months and years
* Regional and Category-wise sales analysis
* Correlation heatmap for numerical variables
* Top 10 profitable products identified

### Step 3: Model Training

* Task: Predict Profit (Regression problem)
* Model: Random Forest Regressor
* Split: 80% training, 20% testing
* Features: Numerical + Encoded categorical columns
* Evaluation Metrics: MAE, RMSE, R² score (~0.87)
* Feature importance extracted to identify key profit drivers

### Step 4: Dashboard Development

* Built using Streamlit
* Interactive filters: Year, Region, Category
* KPIs: Total Sales, Total Profit, Average Discount, Total Orders
* Visualizations:

  * Line chart for Sales & Profit trends
  * Bar chart for Sales by Region & Category
  * Top 10 products
* Expandable data table for filtered exploration

## 7. Results & Key Findings

* Total Sales: [Insert Value]
* Total Profit: [Insert Value]
* Average Discount: [Insert Value]
* R² Score of Random Forest Model: ~0.87
* Consumer segment most profitable
* Positive correlation observed between Sales and Profit
* Few high-value transactions contribute significantly to profit
* Top products and categories for focus identified

## 8. Conclusion

The project provides comprehensive insights into Superstore sales data through EDA, predictive modeling, and dashboard visualization. The Random Forest model accurately predicts profit, and the interactive dashboard facilitates data-driven decisions. Insights can guide inventory management, marketing, and sales strategies.

## 9. Future Work

* Explore advanced machine learning models (XGBoost, Gradient Boosting) for better prediction.
* Include promotional, seasonal, and external factors for improved forecasting.
* Deploy the dashboard online for real-time analytics.
* Automate report generation from dashboard data for business stakeholders.

## 10. Tools & Technologies

* Python, pandas, numpy, matplotlib, seaborn, scikit-learn, plotly, streamlit, joblib
* VS Code / Jupyter Notebook
* Git & GitHub for version control

## 11. References

* Kaggle Superstore Sales Dataset: [Link](https://www.kaggle.com/datasets/juhi1994/superstore)
* pandas, numpy, matplotlib, seaborn, scikit-learn, plotly, streamlit documentation
* Aurélien Géron, Hands-On Machine Learning with Scikit-Learn
