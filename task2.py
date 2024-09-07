import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Import your custom LinearRegression model
from multi_linear_reg_module import LinearRegression

# Define a function for data preprocessing
def preprocess_data(df):
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Convert categorical columns with numerical data
    df['num-of-doors'].replace({'two': 2, 'four': 4}, inplace=True)
    df['num-of-cylinders'].replace({'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12}, inplace=True)

    # Identify numerical and categorical columns
    numerical_cols = ['normalized-losses', 'num-of-doors', 'bore', 'stroke', 'horsepower', 'peak-rpm']
    categorical_cols = ['make', 'fuel-type', 'aspiration', 'body-style', 
                         'drive-wheels', 'engine-location', 'engine-type', 
                         'fuel-system']

    # Impute missing values for numerical columns with the mean
    imputer_num = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

    # Impute missing values for categorical columns with the mode
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    df_encoded = df_encoded.astype(float)

    # Handle the target variable 'price'
    df_encoded['price'] = pd.to_numeric(df_encoded['price'], errors='coerce')
    df_encoded.dropna(subset=['price'], inplace=True)

    return df_encoded, numerical_cols

# Define a function for EDA and visualizations
def perform_eda(df_encoded, numerical_cols):
    st.subheader('Distribution of Car Prices')
    st.write("### Histogram of Prices")
    st.write(sns.histplot(df_encoded['price'], kde=True, bins=30))
    st.pyplot()

    st.subheader('Histograms of Numerical Columns')
    for column in numerical_cols:
        st.write(f"### Distribution of {column}")
        st.write(sns.histplot(df_encoded[column], bins=30, kde=True))
        st.pyplot()

    st.subheader('Pairplot of Numerical Features and Price')
    st.write(sns.pairplot(df_encoded[['price'] + numerical_cols]))
    st.pyplot()

    st.subheader('Correlation Heatmap for Numerical Features')
    correlation_matrix = df_encoded[numerical_cols + ['price']].corr()
    st.write(sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f'))
    st.pyplot()

# Define a function for PCA and model training
def train_and_evaluate_model(df_encoded):
    y = df_encoded['price']
    X = df_encoded.drop('price', axis=1)
    
    # Apply MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    
    st.subheader('Cumulative Explained Variance Ratio (PCA)')
    st.write(plt.figure(figsize=(12, 6)))
    st.write(plt.plot(cumulative_explained_variance, marker='o', linestyle='--'))
    st.title('PCA Components')
    st.pyplot()

    num_components = np.argmax(cumulative_explained_variance >= 0.95) + 1
    st.write(f'Number of components explaining 95% of the variance: {num_components}')

    # Apply PCA with chosen number of components
    pca = PCA(n_components=num_components)
    X_pca_reduced = pca.fit_transform(X_scaled)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_pca_reduced, y, test_size=0.2, random_state=42)

    # Train Model
    model = LinearRegression(alpha=0.01, epochs=1000)
    model.fit(X_train, y_train)

    # Predict and Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    st.subheader('Model Performance')
    st.write(f'Mean Squared Error (MSE): {mse:.2f}')
    st.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    st.write(f'R-squared (R²): {r2:.2f}')
    
    # Plot Actual vs Predicted prices
    st.subheader('Actual vs Predicted Prices')
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.grid(True)
    st.pyplot()

# Streamlit app
def main():
    st.title('Automobile Data Analysis and Regression Model')

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read and process the CSV file
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())
        
        # Preprocess the data
        df_encoded, numerical_cols = preprocess_data(df)
        
        # EDA and Visualization
        perform_eda(df_encoded, numerical_cols)
        
        # Train and Evaluate Model
        train_and_evaluate_model(df_encoded)

if __name__ == "__main__":
    main()
