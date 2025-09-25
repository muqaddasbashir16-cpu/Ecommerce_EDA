
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title("E-Commerce Dataset EDA")

# --- Load Data ---
uploaded_file = st.file_uploader("Upload your ecommerce_dataset.csv", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['order_date'] = pd.to_datetime(df['order_date'])

    # --- Basic Info ---
    st.subheader("Basic Info")
    st.write("Shape:", df.shape)
    st.write("Missing values:", df.isnull().sum())
    st.write("Duplicate rows:", df.duplicated().sum())
    st.write("Info:")
    buffer = []
    df.info(buf=buffer)
    info_str = "\n".join(buffer)
    st.text(info_str)
    st.write("Describe (all):")
    st.write(df.describe(include='all'))

    # Create revenue column
    df['revenue'] = df['price'] * df['quantity'] * (1 - df['discount'])

    # --- 1. Numeric Distributions ---
    st.subheader("Numeric Distributions")
    numeric_cols = ['quantity', 'price', 'discount']
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)

    # --- 2. Categorical Counts ---
    st.subheader("Categorical Counts")
    categorical_cols = ['category','region','payment_method']
    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(6,4))
        df[col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f'Count of {col}')
        st.pyplot(fig)

    # --- 3. Correlation Heatmap ---
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation between Numeric Columns')
    st.pyplot(fig)

    # --- 4. Monthly Revenue Trend ---
    st.subheader("Monthly Revenue Trend")
    monthly_revenue = df.set_index('order_date').groupby(pd.Grouper(freq='M'))['revenue'].sum()
    fig, ax = plt.subplots(figsize=(10,5))
    monthly_revenue.plot(marker='o', ax=ax)
    ax.set_title('Monthly Revenue Trend')
    ax.set_xlabel('Month')
    ax.set_ylabel('Revenue')
    st.pyplot(fig)

    # --- EXTRA GRAPHS ---
    st.subheader("Extra Graphs")

    # 5. Top 10 Products by Revenue
    top_products = df.groupby('product_id')['revenue'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=top_products.index.astype(str), y=top_products.values, ax=ax)
    ax.set_title('Top 10 Products by Revenue')
    st.pyplot(fig)

    # 6. Average Discount by Category
    avg_discount = df.groupby('category')['discount'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=avg_discount.index, y=avg_discount.values, ax=ax)
    ax.set_title('Average Discount by Category')
    st.pyplot(fig)

    # 7. Revenue by Region
    revenue_by_region = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=revenue_by_region.index, y=revenue_by_region.values, ax=ax)
    ax.set_title('Total Revenue by Region')
    st.pyplot(fig)

    # 8. Monthly Orders by Payment Method
    orders_time_payment = df.groupby(
        [pd.Grouper(key='order_date', freq='M'), 'payment_method']
    ).size().reset_index(name='orders')
    fig, ax = plt.subplots(figsize=(12,6))
    sns.lineplot(data=orders_time_payment, x='order_date', y='orders',
                 hue='payment_method', marker='o', ax=ax)
    ax.set_title('Monthly Orders by Payment Method')
    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to begin.")
