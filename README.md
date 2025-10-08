# Datathon Project Submission

## Superstore Sales Analysis Dashboard

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive data science project that performs Exploratory Data Analysis (EDA), builds a machine learning model, and creates an interactive dashboard for the Superstore sales dataset.

## 🎯 Overview

This project provides end-to-end analysis of retail sales data, featuring:
- **Data preprocessing** and feature engineering
- **Exploratory Data Analysis** with comprehensive visualizations
- **Machine Learning model** for profit prediction using Random Forest
- **Interactive Streamlit dashboard** with real-time filtering and analytics

## 🧭 Project Stages

| Stage | Description |
| --- | --- |
| Data Collection | Downloaded dataset from Kaggle and loaded using Pandas. |
| Data Preprocessing | Handled missing values, converted dates, removed duplicates, and engineered new features |
| Exploratory Data Analysis | Visualized data distributions, correlations, and trends using Matplotlib and Seaborn |
| Model Training | Built a Random Forest Regressor to predict profit based on sales features. |
| Model Evaluation | Evaluated model using MAE and R² metrics with 80-20 train-test split. |
| Dashboard Development | Developed an interactive Streamlit dashboard using Plotly for data visualization. |

## ✨ Features

### 📈 Data Analysis
- Data cleaning and preprocessing with robust error handling
- Comprehensive EDA with distribution plots, correlation analysis, and trend analysis
- Feature engineering including time-based features and frequency encoding
- Automated visualization generation saved to `outputs/` directory

### 🤖 Machine Learning
- Random Forest Regression model for profit prediction
- Hyperparameter tuning using RandomizedSearchCV
- Model evaluation with MAE, RMSE, and R² metrics
- Feature importance analysis and visualization
- Model persistence using joblib

### 🎨 Interactive Dashboard
- Real-time filtering by Year, Region, and Category
- Key Performance Indicators (KPIs) display
- Interactive charts using Plotly
- Sales and profit trend analysis
- Top products and regional performance insights

## 📁 Project Structure

```
superstore_analysis/
├── 📄 app.py                    # Streamlit dashboard application
├── 📄 eda_model.py             # EDA and ML model training
├── 📄 test.py                  # Testing utilities
├── 📄 requirements.txt         # Python dependencies
├── 📄 Superstore.csv           # Dataset (not included in repo)
├── 📂 flowcharts/              # Project flowcharts
│   ├── EDA flow.png
│   ├── model training flow.png
│   └── sales data analysis.png
├── 📂 outputs/                 # Generated plots and data snapshots
├── 📂 models/                  # Trained ML models
└── 📂 venv/                   # Virtual environment (not tracked)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/UdayIge/Superstore-Sales-Analysis.git
   cd Superstore-Sales-Analysis
   ```

2. **Set up virtual environment** (recommended)
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Download the Superstore dataset from [Kaggle](https://www.kaggle.com/datasets/juhi1994/superstore)
   - Place `Superstore.csv` in the project root directory

### Usage

#### 🔍 Run EDA and Model Training
```bash
python eda_model.py
```
This will:
- Load and preprocess the data
- Generate EDA visualizations (saved to `outputs/`)
- Train and tune the Random Forest model
- Save the trained model to `models/`
- Display evaluation metrics

#### 📊 Launch Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501` to access the dashboard.

## 📊 EDA Workflow

<img src="./flowcharts/sales data analysis.png" alt="EDA Workflow" width="100%"/>

## 🎛️ Dashboard Features

### Key Performance Indicators
- **Total Sales**: Sum of all sales in filtered data
- **Total Profit**: Sum of all profits in filtered data  
- **Average Discount**: Mean discount percentage
- **Total Orders**: Count of orders

<img src="./dashboard/Dashboard (1).png" alt="Dashboard Features" width="100%"/>

### Interactive Visualizations
- **Sales & Profit Trends**: Time series analysis over months
- **Regional & Category Analysis**: Bar charts showing sales distribution
- **Top Products**: Horizontal bar chart of most profitable products
- **Data Preview**: Expandable table of filtered data

<img src="./dashboard/Dashboard (2).png" alt="Dashboard Features" width="100%"/>

### Filtering Options
- **Year**: Multi-select year filter (default: all years)
- **Region**: Multi-select region filter (default: all regions)
- **Category**: Multi-select category filter (default: all categories)

<img src="./dashboard/Dashboard (3).png" alt="Dashboard Features" width="100%"/>

## 🔬 Machine Learning Model

### Model Architecture
- **Algorithm**: Random Forest Regressor
- **Target Variable**: Profit
- **Features**: Sales, Discount, Region, Category, Product Name, Customer Segment, Ship Mode, and engineered features

### Feature Engineering
- **Time Features**: Order Year, Month, Day, Weekday
- **Shipping Features**: Order-to-Ship time
- **Frequency Encoding**: High-cardinality categorical variables
- **One-Hot Encoding**: Low-cardinality categorical variables

### Model Performance
The model is evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)  
- **R² Score** (Coefficient of Determination)

## 📦 Dependencies

Key packages used in this project:
- `streamlit` - Interactive web dashboard
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `plotly` - Interactive visualizations
- `scikit-learn` - Machine learning algorithms
- `matplotlib` - Static plotting
- `seaborn` - Statistical data visualization
- `joblib` - Model persistence

## 📈 Generated Outputs

The EDA process generates several visualizations in the `outputs/` directory:
- `distributions.png` - Sales, Profit, and Discount distributions
- `monthly_sales.png` - Monthly sales trends
- `top_products.png` - Top 15 products by sales
- `correlation_matrix.png` - Feature correlation heatmap
- `sales_region_category.png` - Sales by region and category
- `feature_importances.png` - Top 30 feature importances

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: [Kaggle Superstore Dataset](https://www.kaggle.com/datasets/juhi1994/superstore)
- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Plotly](https://plotly.com/)
- Machine learning with [scikit-learn](https://scikit-learn.org/)

<!-- ## 📞 Contact

**Uday Ige** - [GitHub](https://github.com/UdayIge)

Project Link: [https://github.com/UdayIge/Superstore-Sales-Analysis](https://github.com/UdayIge/Superstore-Sales-Analysis) -->

---

⭐ If you found this project helpful, please give it a star!

