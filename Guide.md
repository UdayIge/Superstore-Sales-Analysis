## Entire project workflow

Entire project workflow step by step so you fully understand how the **EDA + Model pipeline** and the **Streamlit dashboard** work together. I’ll explain it in a structured way, covering data, preprocessing, analysis, model, and visualization.



## **1️⃣ Dataset**

* **Source:** Kaggle **Superstore Sales Dataset**
  [Link to Kaggle dataset](https://www.kaggle.com/datasets/juhi1994/superstore)

* **Content:** Sales transactions with columns like:
  `Order ID, Customer ID, Product ID, Product Name, Category, Sub-Category, Sales, Quantity, Discount, Profit, Order Date, Ship Date, Region, Segment`

* **Goal:** Analyze sales patterns, extract insights, and predict **Profit**.

---

## **2️⃣ Preprocessing (`eda_model.py`)**

The first step is to **clean and prepare the data** for EDA and modeling.

### Steps:

1. **Drop unnecessary columns**
   Columns like `Row ID`, `Country`, `Postal Code` are removed because they are irrelevant for analysis.

2. **Parse dates**

   * `Order Date` and `Ship Date` are converted to `datetime` objects.
   * `errors="coerce"` ensures invalid dates become `NaT`.
   * New features:

     * `Order Year`
     * `Order Month`
     * `Order Day`
     * `Order Weekday`
     * `Order to Ship Time` (difference in days)

3. **Handle categorical variables**

   * Low-cardinality columns (few unique values, e.g., `Region`, `Category`) → **One-Hot Encoding**
   * High-cardinality columns (many unique values, e.g., `Product Name`) → **Frequency Encoding**

4. **Remove duplicates** and handle missing values

   * Numerical columns → median imputation
   * Categorical columns → mode imputation

✅ Output: Cleaned DataFrame ready for EDA and modeling.

---

## **3️⃣ Exploratory Data Analysis (EDA)**

EDA helps uncover insights and patterns.

### Key analyses performed:

1. **Distribution plots**

   * `Sales`, `Profit`, `Discount` → histogram with KDE
   * Identify skewed distributions or outliers

2. **Time-series analysis**

   * Sales and profit trends over **time** (monthly/annual)
   * Line plots reveal seasonal patterns or growth trends

3. **Top products / regions / categories**

   * Top 10 products by **profit**
   * Sales by **Region** and **Category**

4. **Correlation analysis**

   * Heatmap of numerical features
   * Shows relationships like **Sales vs Profit** (strong positive correlation)

5. **EDA outputs**

   * Plots saved to `outputs/` folder for reference and reporting.

---

## **4️⃣ Model Training (`eda_model.py`)**

After EDA, we build a **machine learning model** to predict `Profit`.

### Steps:

1. **Split features and target**

   ```python
   X = df.drop(columns=['Profit', 'Order ID', 'Customer ID', 'Product ID', 'Order Date', 'Ship Date'])
   y = df['Profit']
   ```

2. **Preprocessing pipelines**

   * **Numeric features:** median imputation + standard scaling
   * **Categorical features:** One-Hot Encoding for low-cardinality columns
   * Frequency-encoded columns are left as-is

3. **Pipeline & Random Forest**

   * Use `Pipeline` with preprocessing + `RandomForestRegressor`
   * Optional: `RandomizedSearchCV` for hyperparameter tuning

     * `n_estimators`, `max_depth`, `min_samples_split`, etc.

4. **Train/test split**

   * 80% train, 20% test
   * Model trained on training set, evaluated on test set

5. **Evaluation metrics**

   * **MAE** (Mean Absolute Error)
   * **RMSE** (Root Mean Squared Error)
   * **R²** (coefficient of determination)

6. **Feature importance**

   * Identify top 30 features contributing to profit
   * Saved as `feature_importances.png`

7. **Save trained model**

   * `rf_superstore.joblib` in `models/` for future use

✅ Output: Trained Random Forest model + evaluation + plots

---

## **5️⃣ Streamlit Dashboard (`app.py`)**

The dashboard visualizes data interactively and lets users explore trends.

### Components:

1. **Load data**

   * CSV loaded and preprocessed (same as `eda_model.py`)
   * `@st.cache_data` used for faster reloads

2. **Sidebar filters**

   * Filter data by `Year`, `Region`, `Category`
   * Filters dynamically update plots and metrics

3. **KPIs**

   * Total Sales, Total Profit, Average Discount, Total Orders

4. **Visualizations**

   * **Line chart:** Sales & Profit over time (monthly)
   * **Bar chart:** Sales by Region & Category
   * **Top products:** Horizontal bar chart of top 10 profitable products

5. **Interactive data table**

   * Expandable section showing filtered data

6. **Deployment-ready**

   * Works locally with `streamlit run app.py`
   * Can be deployed to Streamlit Cloud or any web server

---

## **6️⃣ Complete Workflow**

1. **Download dataset** → `Superstore.csv`
2. **Run `eda_model.py`**

   * Cleans data
   * Performs enhanced EDA
   * Trains Random Forest model
   * Saves plots and model
3. **Run `app.py`**

   * Interactive dashboard with filters, KPIs, and plots
   * Explore insights from EDA & model

---

### **7️⃣ Project Highlights**

* **Data cleaning:** Robust handling of dates, missing values, duplicates
* **Feature engineering:** Time features, order-to-ship days, frequency encoding
* **EDA:** Rich visuals for patterns and correlations
* **Predictive modeling:** Random Forest regression with pipeline and hyperparameter tuning
* **Interactive dashboard:** Streamlit + Plotly for KPIs, charts, filters

---

Perfect! Here’s a **step-by-step guide** to upload your entire project (EDA + model + app) to GitHub, including best practices for a clean repo.

---

## **1️⃣ Initialize Git in Your Project Folder**

Open **Command Prompt / Terminal** in your project directory:

```bash
cd "C:\Users\HP\Desktop\Programming\python Project\superstore_analysis"
git init
```

This creates a `.git` folder in your project.

---

## **2️⃣ Create a `.gitignore` File**

Make sure you **don’t upload unnecessary or large files** (like datasets or model files you don’t want in GitHub).

Create a file named `.gitignore` in the project folder:

```
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.pkl
*.joblib

# Environment / IDE
.env
.venv/
*.vscode/
*.idea/

# Data / Output
Superstore.csv
outputs/
models/
```

> You can upload a **[sample dataset](https://drive.google.com/file/d/1D2sHgekeMBBLLeByy2EcBcetvEqL4O4m/view?usp=sharing)** instead of the full CSV if it’s large.

---

## **3️⃣ Add Files to Git**

Add all your project files **except ignored files**:

```bash
git add .
```

Check what’s staged:

```bash
git status
```

---

## **4️⃣ Commit Changes**

```bash
git commit -m "Initial commit: Add EDA, model, and Streamlit app"
```

---

## **5️⃣ Create a GitHub Repository**

1. Go to [GitHub](https://github.com) → Click **New repository**
2. Name it, e.g., `superstore-analysis`
3. Don’t initialize with README (we’ll push our local repo)

---

## **6️⃣ Link Local Repo to GitHub**

```bash
git remote add origin https://github.com/<your-username>/superstore-analysis.git
```

> Replace `<your-username>` with your GitHub username.

---

## **7️⃣ Push Your Code**

```bash
git branch -M main
git push -u origin main
```

Your project should now be live on GitHub.

---

## **8️⃣ Optional: Add `requirements.txt`**

Generate it from your environment:

```bash
pip freeze > requirements.txt
```

This ensures anyone cloning your repo can install the same packages.

---

