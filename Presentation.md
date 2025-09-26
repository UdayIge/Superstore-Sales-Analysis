# Superstore Sales Analysis Project Presentation

## Slide 1: Title

**Superstore Sales Analysis and Profit Prediction**

* Datathon Project
* [Uday Ige], [Final Year ECM]

---

## Slide 2: Introduction & Objectives

* Analyze sales data and predict profit.
* Build an interactive dashboard.
* Objectives:

  * Data Cleaning & Preprocessing
  * EDA to uncover patterns
  * Train Random Forest Regressor
  * Develop interactive Streamlit dashboard

---

## Slide 3: Dataset

* Source: Kaggle Superstore Sales Dataset
* ~10,000 records with sales, profit, product, category, region, and date info
* Mixed data types: numerical, categorical, datetime

---

## Slide 4: Data Preprocessing & EDA

* Dropped irrelevant columns, handled missing values
* Converted date columns, engineered features (Order Year, Order to Ship Time)
* Encoded categorical variables
* EDA:

  * Sales & Profit distributions
  * Time-series trends
  * Regional & Category sales
  * Correlation analysis

---

## Slide 5: Model Training & Performance

* Task: Predict Profit (Regression)
* Model: Random Forest Regressor
* Split: 80% train, 20% test
* Metrics: MAE, RMSE, R² (~0.87)
* Feature importance highlights top profit drivers

---

## Slide 6: Streamlit Dashboard

* Interactive filters: Year, Region, Category
* KPIs: Total Sales, Profit, Avg. Discount, Orders
* Visualizations:

  * Line chart: Sales & Profit trends
  * Bar chart: Sales by Region & Category
  * Top 10 products

---

## Slide 7: Key Findings & Conclusion

* Consumer segment most profitable
* Positive correlation: Sales ↔ Profit
* Few high-value transactions dominate profit
* Dashboard provides actionable insights for decision-making

---

## Slide 8: Future Work & References

* Explore advanced ML models (XGBoost, Gradient Boosting)
* Include promotional/seasonal data for better prediction
* Deploy dashboard online
* References:

  * Kaggle Dataset: [Link](https://www.kaggle.com/datasets/juhi1994/superstore)
  * Python libraries & ML documentation
  * Aurélien Géron, Hands-On Machine Learning with Scikit-Learn
