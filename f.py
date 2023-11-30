import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tkinter import Tk, filedialog

def load_data_gui():
    """Load data interactively using Tkinter GUI."""
    root = Tk()
    root.title("Data Loading GUI")

    file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

    if file_path:
        df = pd.read_csv(file_path)
        root.destroy()
        return df
    else:
        root.destroy()
        return None

def preprocess_data(df):
    """Preprocess data by handling missing values."""
    # Drop rows with missing values
    df.dropna(subset=['solar_mass', 'solar_radius', 'solar_gravity', 'solar_distance'], inplace=True)
    return df

def visualize_bar_plot(feature, title):
    """Visualize bar plot for a specific feature."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='name', y=feature, data=df)
    plt.title(f'{title} vs. Name')
    plt.xlabel('Names')
    plt.ylabel(title.capitalize())
    plt.show()

def perform_regression(X, y):
    """Perform linear regression using TensorFlow and Keras."""
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build a simple neural network for regression using Keras
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1)
    ])


    model.compile(optimizer='adam', loss='mean_squared_error')


    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    mse = model.evaluate(X_test_scaled, y_test)
    print(f'Mean Squared Error: {mse}')


    predictions = model.predict(X_test_scaled)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=predictions.flatten())
    plt.title('Actual vs. Predicted Solar Gravity')
    plt.xlabel('Actual Solar Gravity')
    plt.ylabel('Predicted Solar Gravity')
    plt.show()


df = load_data_gui()

if df is not None:
    # Preprocess data
    df = preprocess_data(df)


    visualize_bar_plot('solar_mass', 'mass')
    visualize_bar_plot('solar_radius', 'radius')
    visualize_bar_plot('solar_gravity', 'gravity')
    visualize_bar_plot('solar_distance', 'distance')

    X = df[['solar_distance']]
    y = df['solar_gravity']
    perform_regression(X, y)
