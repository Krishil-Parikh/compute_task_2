import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class LinearRegression:
    def __init__(self, alpha=0.01, epochs=1000):
        self.alpha = alpha  # Learning rate
        self.epochs = epochs  # Number of iterations for gradient descent
        self.m = None  # Coefficients
        self.means = None  # Means for normalization
        self.stds = None  # Standard deviations for normalization
        self.cost_history = []

    def fit(self, X, y):
        # Convert y to a NumPy array to avoid indexing issues
        y = np.array(y)
        
        # Normalize the features
        self.means = X.mean(axis=0)
        self.stds = X.std(axis=0)
        X = (X - self.means) / self.stds

        # Add a column of ones to X for the intercept (bias) term
        X = np.c_[np.ones(X.shape[0]), X]

        # Initialize parameters
        self.m = np.zeros(X.shape[1])

        # Gradient descent
        for epoch in range(self.epochs):
            self.m = self.gradient_descent_multi(X, y)
            cost = np.mean((np.dot(X, self.m) - y) ** 2)
            self.cost_history.append(cost)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Cost = {cost}, Coefficients = {self.m}")

    def plot(self):
        # Plot cost history
        plt.figure(figsize=(14, 6))
        plt.plot(range(self.epochs), self.cost_history, label='Cost Function Convergence')
        plt.xlabel('Epochs')
        plt.ylabel('Cost (MSE)')
        plt.title('Convergence of Gradient Descent for Multiple Linear Regression')
        plt.legend()
        plt.show()

    def gradient_descent_multi(self, X, y):
        m_gradient = np.zeros_like(self.m)
        n = len(y)

        for i in range(n):
            y_pred = np.dot(X[i], self.m)
            error = y_pred - y[i]
            m_gradient += (2/n) * error * X[i]

        return self.m - self.alpha * m_gradient

    def predict(self, new_data):
        new_data = np.array(new_data)
        new_data_normalized = (new_data - self.means) / self.stds
        new_data_normalized = np.c_[np.ones(new_data_normalized.shape[0]), new_data_normalized]
        predictions = np.dot(new_data_normalized, self.m)
        return predictions

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("Student_Performance.csv")  # Replace with your file path
    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

    # Automatically select features (all columns except the last one) and target (last column)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Initialize and fit the model
    model = LinearRegression(alpha=0.01, epochs=1000)
    model.fit(X, y)

    # Predict with new data
    new_data = np.array([[8, 91, 0, 4, 5]])  # Example new data
    predictions = model.predict(new_data)
    print("Predictions for new data:", predictions)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)
    model = LinearRegression(alpha=0.01, epochs=1000)
    model.fit(X_train, y_train)
    y_predictions = model.predict(X_test)
    mse = model.mean_squared_error(y_test, y_predictions)
    print(f"Mean Squared Error: {mse}")
