# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Assume the file is downloaded as 'Superstore.csv'
# file_path = './Superstore.csv'

# # Load the data into a DataFrame
# try:
#     df = pd.read_csv(file_path, encoding='ISO-8859-1')
#     print("Data loaded successfully!")
# except FileNotFoundError:
#     print(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")

# # Display the first 5 rows to get a feel for the data
# print(df.head())
# print("\nDataFrame Info:")
# print(df.info())

# # Check for missing values in each column
# print("Missing values before preprocessing:\n", df.isnull().sum())

# # Drop irrelevant columns to simplify the dataset
# columns_to_drop = ['Row ID', 'Country', 'Postal Code']
# df.drop(columns=columns_to_drop, inplace=True)

# def _parse_mixed_dates(series: pd.Series) -> pd.Series:
#     """Parse date strings that may contain mixed formats.

#     First attempt uses format='mixed' with dayfirst=True; remaining NaT values
#     are re-parsed with dayfirst=False. Returns a pandas Series of datetimes.
#     """
#     cleaned = series.astype(str).str.strip()
#     parsed = pd.to_datetime(cleaned, format='mixed', dayfirst=True, errors='coerce')
#     missing_mask = parsed.isna()
#     if missing_mask.any():
#         reparsed = pd.to_datetime(cleaned[missing_mask], format='mixed', dayfirst=False, errors='coerce')
#         parsed.loc[missing_mask] = reparsed
#     return parsed

# # Convert 'Order Date' and 'Ship Date' to datetime objects for proper analysis
# df['Order Date'] = _parse_mixed_dates(df['Order Date'])
# df['Ship Date'] = _parse_mixed_dates(df['Ship Date'])

# # Drop rows where dates could not be parsed
# rows_before = len(df)
# df = df.dropna(subset=['Order Date', 'Ship Date'])
# rows_after = len(df)
# if rows_before != rows_after:
#     print(f"Dropped {rows_before - rows_after} rows with invalid dates after parsing.")

# # Check for duplicates and remove them
# print(f"\nNumber of duplicate rows before removal: {df.duplicated().sum()}")
# df.drop_duplicates(inplace=True)
# print(f"Number of duplicate rows after removal: {df.duplicated().sum()}")

# # Create a new feature: 'Order to Ship Time' (in days)
# df['Order to Ship Time'] = (df['Ship Date'] - df['Order Date']).dt.days

# # Display the data types and info after cleaning
# print("\nDataFrame Info after preprocessing:")
# print(df.info())


# sns.set_style("whitegrid")

# # Plot the distribution of 'Sales' and 'Profit'
# plt.figure(figsize=(15, 6))
# plt.subplot(1, 2, 1)
# sns.histplot(df['Sales'], bins=50, kde=True)
# plt.title('Distribution of Sales')

# plt.subplot(1, 2, 2)
# sns.histplot(df['Profit'], bins=50, kde=True)
# plt.title('Distribution of Profit')
# plt.tight_layout()
# plt.show()

# # Analyze profit by 'Segment'
# plt.figure(figsize=(8, 5))
# sns.barplot(x='Segment', y='Profit', data=df, estimator=sum)
# plt.title('Total Profit by Customer Segment')
# plt.show()

# # Analyze sales over time
# df['Order Year'] = df['Order Date'].dt.year
# sales_over_time = df.groupby('Order Year')['Sales'].sum()
# plt.figure(figsize=(10, 6))
# sales_over_time.plot(kind='line', marker='o')
# plt.title('Total Sales Over the Years')
# plt.ylabel('Total Sales')
# plt.xlabel('Year')
# plt.show()

# # Correlation matrix to find relationships between numerical variables
# numerical_cols = ['Sales', 'Quantity', 'Discount', 'Profit', 'Order to Ship Time']
# plt.figure(figsize=(10, 8))
# sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix of Numerical Features')
# plt.show()

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score

# # Feature Engineering: One-hot encode ALL remaining categorical (object) variables
# categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
# if len(categorical_columns) > 0:
#     df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# # Define features (X) and target (y)
# X = df.drop(columns=['Order Date', 'Ship Date', 'Order ID', 'Customer ID', 'Product ID', 'Order Year', 'Profit'], errors='ignore')
# y = df['Profit']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the Random Forest model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model's performance
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Absolute Error (MAE): {mae:.2f}")
# print(f"R-squared (R²): {r2:.2f}")



# ----------------------------------------------------------------------------------------------
# import streamlit as st
# import pandas as pd
# import plotly.express as px

# st.set_page_config(layout="wide")

# st.title("Superstore Sales Analysis Dashboard")
# st.write("Explore key metrics and insights from the Superstore Sales data.")

# # ---------- Load Data ----------
# # Make sure the CSV is in the same folder as this app or update the path.
# df = pd.read_csv("Superstore.csv", encoding="ISO-8859-1")

# # ---------- Pre-processing ----------
# # Drop unused columns if they exist
# for col in ["Row ID", "Country", "Postal Code"]:
#     if col in df.columns:
#         df.drop(columns=col, inplace=True)

# # Parse dates safely (handles mm/dd/yyyy automatically)
# df["Order Date"] = pd.to_datetime(
#     df["Order Date"], infer_datetime_format=True, errors="coerce"
# )
# df["Ship Date"] = pd.to_datetime(
#     df["Ship Date"], infer_datetime_format=True, errors="coerce"
# )

# # Drop rows with invalid dates if any
# df.dropna(subset=["Order Date", "Ship Date"], inplace=True)

# # Create Order Year column
# df["Order Year"] = df["Order Date"].dt.year

# # ---------- Metrics ----------
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.metric("Total Sales", f"${df['Sales'].sum():,.2f}")
# with col2:
#     st.metric("Total Profit", f"${df['Profit'].sum():,.2f}")
# with col3:
#     st.metric("Average Discount", f"{df['Discount'].mean():.2%}")

# st.markdown("---")

# # ---------- Charts ----------
# st.subheader("Sales and Profit Over Time")
# time_data = (
#     df.groupby("Order Year")[["Sales", "Profit"]].sum().reset_index()
# )
# fig = px.line(
#     time_data,
#     x="Order Year",
#     y=["Sales", "Profit"],
#     title="Total Sales and Profit by Year",
# )
# st.plotly_chart(fig, use_container_width=True)

# st.subheader("Sales by Region and Category")
# region_cat_sales = (
#     df.groupby(["Region", "Category"])["Sales"].sum().reset_index()
# )
# fig2 = px.bar(
#     region_cat_sales,
#     x="Region",
#     y="Sales",
#     color="Category",
#     barmode="group",
#     title="Sales by Region and Category",
# )
# st.plotly_chart(fig2, use_container_width=True)