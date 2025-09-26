"""
Enhanced EDA and Model Training for Superstore Sales Dataset

Save this file as `superstore_eda_model.py` and run it in your project folder containing `Superstore.csv`.

What it does:
- Loads and preprocesses data (safe date parsing, missing values, feature engineering).
- Performs enhanced EDA and saves key plots to `outputs/`.
- Encodes categorical variables using a hybrid strategy (one-hot for low-cardinality, frequency for high-cardinality).
- Builds a scikit-learn pipeline with imputation and scaling.
- Tunes a RandomForestRegressor via RandomizedSearchCV (lightweight defaults; adjust for final runs).
- Evaluates model (MAE, RMSE, R2) and saves trained model to `models/`.

Requirements:
- pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
- Install with: pip install pandas numpy matplotlib seaborn scikit-learn joblib

"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from joblib import dump

# ------------------------- Helper utilities -------------------------
OUTPUT_DIR = Path('outputs')
MODEL_DIR = Path('models')
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def load_data(path='Superstore.csv'):
    """Load CSV with robust options."""
    df = pd.read_csv(path, encoding='ISO-8859-1')
    # strip column names
    df.columns = df.columns.str.strip()
    return df


def preprocess(df, drop_high_card_cols=True, high_card_thresh=50):
    """Preprocess dataframe.

    Steps:
    - drop obviously useless columns if present
    - robust date parsing
    - drop rows with invalid dates
    - create order-to-ship time, month, day, weekday features
    - clean column names
    - frequency-encode high-cardinality text columns
    - keep track of columns for downstream encoding
    """
    df = df.copy()

    # drop if present
    for col in ['Row ID', 'Country', 'Postal Code']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Parse dates robustly
    df['Order Date'] = pd.to_datetime(df['Order Date'], infer_datetime_format=True, errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], infer_datetime_format=True, errors='coerce')

    # drop rows with invalid critical fields
    df.dropna(subset=['Order Date', 'Ship Date', 'Sales', 'Profit'], inplace=True)

    # Feature engineering
    df['Order to Ship Time'] = (df['Ship Date'] - df['Order Date']).dt.days.fillna(0).astype(int)
    df['Order Year'] = df['Order Date'].dt.year
    df['Order Month'] = df['Order Date'].dt.month
    df['Order Day'] = df['Order Date'].dt.day
    df['Order Weekday'] = df['Order Date'].dt.dayofweek

    # Strip and lowercase object columns
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # Identify high-cardinality object columns
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    high_card_cols = [c for c in obj_cols if df[c].nunique() > high_card_thresh]
    low_card_cols = [c for c in obj_cols if c not in high_card_cols]

    # Frequency-encode high-cardinality columns (helps RandomForest)
    for c in high_card_cols:
        freq_name = f"{c}_freq"
        freq = df[c].value_counts(normalize=True)
        df[freq_name] = df[c].map(freq).fillna(0)
        # optionally drop the original high-card column
        if drop_high_card_cols:
            df.drop(columns=c, inplace=True)

    return df, low_card_cols


def enhanced_eda(df):
    """Produce and save EDA plots to outputs/"""
    sns.set(style='whitegrid')

    # Distribution plots for sales, profit, discount
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df['Sales'], bins=50, kde=True, ax=axes[0])
    axes[0].set_title('Sales Distribution')
    sns.histplot(df['Profit'], bins=50, kde=True, ax=axes[1])
    axes[1].set_title('Profit Distribution')
    sns.histplot(df['Discount'], bins=30, kde=True, ax=axes[2])
    axes[2].set_title('Discount Distribution')
    plt.tight_layout()
    plt.show()
    plt.savefig(OUTPUT_DIR / 'distributions.png')
    plt.close()

    # Sales over time (monthly)
    monthly = df.set_index('Order Date').resample('M')['Sales'].sum()
    plt.figure(figsize=(12, 5))
    monthly.plot(marker='o')
    plt.title('Monthly Sales')
    plt.ylabel('Sales')
    plt.show()
    plt.savefig(OUTPUT_DIR / 'monthly_sales.png')
    plt.close()

    # Top 15 products by sales (if Product Name present)
    if 'Product Name' in df.columns:
        top_products = df.groupby('Product Name')['Sales'].sum().nlargest(15)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_products.values, y=top_products.index)
        plt.title('Top 15 Products by Sales')
        plt.xlabel('Sales')
        plt.tight_layout()
        plt.show()
        plt.savefig(OUTPUT_DIR / 'top_products.png')
        plt.close()

    # Correlation heatmap for numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=False)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    plt.savefig(OUTPUT_DIR / 'correlation_matrix.png')
    plt.close()

    # Sales by Region & Category
    if {'Region', 'Category'}.issubset(df.columns):
        region_cat = df.groupby(['Region', 'Category'])['Sales'].sum().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(data=region_cat, x='Region', y='Sales', hue='Category')
        plt.title('Sales by Region and Category')
        plt.tight_layout()
        plt.show()
        plt.savefig(OUTPUT_DIR / 'sales_region_category.png')
        plt.close()

    print(f"EDA plots saved to {OUTPUT_DIR.resolve()}")


def build_and_train_model(df, low_card_cols, target='Profit'):
    """Build ML pipeline, tune and train RandomForest, evaluate and save model."""
    # Drop columns that are identifiers or leak info
    drop_columns = ['Order Date', 'Ship Date', 'Order ID', 'Customer ID', 'Product ID', target]
    X = df.drop(columns=[c for c in drop_columns if c in df.columns])
    y = df[target].astype(float)

    # Identify numeric and categorical columns for ColumnTransformer
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # From low_card_cols, keep only those still in X
    cat_cols = [c for c in low_card_cols if c in X.columns]

    print(f"Training data shape: {X.shape}. Numeric cols: {len(numeric_cols)}. Cat cols: {len(cat_cols)}")

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, cat_cols)
    ], remainder='passthrough')  # passthrough will include any freq-encoded cols

    # Full pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parameter distribution for RandomizedSearchCV
    param_dist = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['auto', 'sqrt', 0.5]
    }

    rnd_search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='neg_mean_absolute_error',
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    print("Starting model tuning (this may take a while depending on resources)...")
    rnd_search.fit(X_train, y_train)

    print("Best params:", rnd_search.best_params_)

    # Best estimator evaluation
    best = rnd_search.best_estimator_
    y_pred = best.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")

    # Save the model
    model_path = MODEL_DIR / 'rf_superstore.joblib'
    dump(best, model_path)
    print(f"Model saved to {model_path}")

    # Feature importance (works after preprocessing -> get feature names)
    try:
        # Get feature names from preprocessor
        num_feats = numeric_cols
        cat_feats = []
        if len(cat_cols) > 0:
            ohe = best.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe']
            cat_ohe_names = list(ohe.get_feature_names_out(cat_cols))
            cat_feats = cat_ohe_names
        # remainder (passthrough) features (e.g., freq encoded) appear at end
        remainder_feats = [c for c in X.columns if c not in numeric_cols + cat_cols]

        all_feats = num_feats + cat_feats + remainder_feats
        importances = best.named_steps['model'].feature_importances_

        # reduce to top 30 for plotting
        fi = pd.Series(importances, index=all_feats).nlargest(30)
        plt.figure(figsize=(8, 10))
        sns.barplot(x=fi.values, y=fi.index)
        plt.title('Top 30 Feature Importances')
        plt.tight_layout()
        plt.show()
        plt.savefig(OUTPUT_DIR / 'feature_importances.png')
        plt.close()
        print(f"Feature importances saved to {OUTPUT_DIR / 'feature_importances.png'}")
    except Exception as e:
        print("Could not compute feature importances:", e)

    # Return trained estimator and metrics
    return best, {'mae': mae, 'rmse': rmse, 'r2': r2}


def main():
    df = load_data('Superstore.csv')
    df, low_card_cols = preprocess(df, drop_high_card_cols=True, high_card_thresh=50)

    print('Data shape after preprocessing:', df.shape)

    # Save a small cleaned snapshot
    df.head().to_csv(OUTPUT_DIR / 'clean_head_snapshot.csv', index=False)
    df.to_csv(OUTPUT_DIR / 'clean_snapshot.csv', index=False)

    # EDA
    enhanced_eda(df)

    # Model
    best_model, metrics = build_and_train_model(df, low_card_cols, target='Profit')

    print('Training complete. Metrics:', metrics)


if __name__ == '__main__':
    main()
