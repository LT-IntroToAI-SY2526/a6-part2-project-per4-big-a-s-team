"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
- 
- 
- 
- 

Dataset: [Diamond Price Dataset]
Predicting: [Diamond Price]
Features: [Carat, cut, color]
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# TODO: Update this with your actual filename
DATA_FILE = 'diamond_features.csv'

def load_and_explore_data(filename):
    """
    Load your dataset and print basic information
    
    TODO:
    - Load the CSV file
    - Print the shape (rows, columns)
    - Print the first few rows
    - Print summary statistics
    - Check for missing values
    """
    print("=" * 70)
    print("LOADING AND EXPLORING DATA")
    print("=" * 70)
    
    # Your code here
    data = pd.read_csv(filename)
    print(f"\n Dataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(data.head())
    print(f"\nBasic statistics:")
    print(data.describe())
    return data

mapping = {
    'Fair': 1.0,
    'Good': 2.0,
    'Very Good': 3.0,
    'Premium': 4.0,
    'Ideal': 5.0
}

cuts = ['Fair', 'Good']

def visualize_data(data):
    """
    Create visualizations to understand your data
    
    TODO:
    - Create scatter plots for each feature vs target
    - Save the figure
    - Identify which features look most important - carat, cut, color 
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
    """
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)
    
    # Your code here
    # Hint: Use subplots like in Part 2!
    plt.figure(figsize=(10,6))
    fig, axes = plt.subplots(2,2,figsize=(12,10))
    fig.suptitle('Diamond Features vs Price', fontsize = 16, fontweight='bold')
    
    axes[0,0].scatter(data['carat'], data['price'], color = 'blue', alpha = 0.6)
    axes[0,0].set_xlabel('Carat')
    axes[0,0].set_ylabel('Price($)')
    axes[0,0].set_title('Carat vs Price')
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].scatter(data['cut'], data['price'], color = 'red', alpha = 0.6)
    axes[0,1].set_xlabel('Cut')
    axes[0,1].set_ylabel('Price($)')
    axes[0,1].set_title('Cut vs Price')
    axes[0,1].grid(True, alpha=0.3)

    axes[1,0].scatter(data['color'], data['price'], color = 'yellow', alpha = 0.6)
    axes[1,0].set_xlabel('Color')
    axes[1,0].set_ylabel('Price($)')
    axes[1,0].set_title('Color vs Price')
    axes[1,0].grid(True, alpha=0.3)

    axes[1, 1].text(0.5, 0.5, 'Space for additional features', 
                    ha='center', va='center', fontsize=12)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('diamond_features.png', dpi=300, bbox_inches='tight')
    plt.show()


def prepare_and_split_data(data):
    """
    Prepare X and y, then split into train/test
    
    TODO:
    - Separate features (X) and target (y)
    - Split into train/test (80/20)
    - Print the sizes
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)
    
    features_columns = ['carat', 'cut', 'color']
    X = data[features_columns]
    y = data['price']

    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"\nFeatures Columns: {list(X.columns)}")

    X_train = X.iloc[:100] # .iloc[row_index, col_indecx] exclusive slicing
    X_test = X.iloc[100:125]
    y_train = y.iloc[:100]
    y_test = y.iloc[100:125]

    print(f"Training Set: {len(X_train)} samples (first 100 diamonds)")
    print(f"Testing Set: {len(X_train)} samples (25 diamons)")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, feature_names):
    """
    Train the linear regression model
    
    TODO:
    - Create and train a LinearRegression model
    - Print the equation with all coefficients
    - Print feature importance (rank features by coefficient magnitude)
    
    Args:
        X_train: training features
        y_train: training target
        feature_names: list of feature names
        
    Returns:
        trained model
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    # model = LinearRegression()
    # model.fit(X_train, y_train)
    
    # print(f"\n=== Model Training Complete ===")
    # print(f"Intercept: ${model.intercept_:.2f}")
    # print(f"\nCoefficients:")
    # for name, coef in zip(feature_names, model.coef_):
    #     print(f"  {name}: {coef:.2f}")
    
    # print(f"\nEquation:")
    # equation = f"Price = "
    # for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
    #     if i == 0:
    #         equation += f"{coef:.2f} × {name}"
    #     else:
    #         equation += f" + ({coef:.2f}) × {name}"
    # equation += f" + {model.intercept_:.2f}"
    # print(equation)
    
    # return model


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate model performance
    
    TODO:
    - Make predictions on test set
    - Calculate R² score
    - Calculate RMSE
    - Print results clearly
    - Create a comparison table (first 10 examples)
    
    Args:
        model: trained model
        X_test: test features
        y_test: test target
        
    Returns:
        predictions
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)

    # predictions = model.predict(X_test)

    # r2 = r2_score(y_test, predictions)
    # mse = mean_squared_error(y_test, predictions)
    # rmse = np.sqrt(mse)

    # print(f"\n=== Model Performance ===")
    # print(f"R² Score: {r2:.4f}")
    # print(f" -> Model explains {r2*100:.2f}% of price variation")
    
    # print(f"\nRoot Mean Squared Error: ${rmse:.2f}")
    # print(f" -> On average, perdictions are off by ${rmse:.2f}")
    
    # print(f"\n=== Feature Importance ===")
    # feature_importance = list(zip(feature_names, np.abs(model.coef_)))
    # feature_importance.sort(key=lambda x: x[1], reverse=True)

    # for i, (name, importance) in enumerate(feature_importance, 1):
    #     print(f"{i}. {name}: {importance:.2f}")

    # return predictions    

# def compare_predictions(y_test, predictions, num_examples=5):
#     """
#     Show side-by-side comparison of actual vs predicted prices
    
#     Args:
#         y_test: actual prices
#         predictions: predicted prices
#         num_examples: number of examples to show
#     """
#     print(f"\n=== Prediction Comparison ===")
#     print(f"{'Actual Price':<15} {'Predicted Price':<18} {''}")
#     print("-" * 60)

#     for i in range(min(num_examples, len(y_test))):
#         actual = y_test.iloc[i]
#         predicted = predictions[i]
#         error = actual - predicted
#         pct_error = (abs(error) / actual) * 100

#         print(f"${actual:>13.2f} ${predicted:>13.2f} ${error:>10.2f} {pct_error:>6.2f}%")


def make_prediction(model, carat, cut, color):
    """
    Make a prediction for a new example
    
    TODO:
    - Create a sample input (you choose the values!)
    - Make a prediction
    - Print the input values and predicted output
    
    Args:
        model: trained model
        feature_names: list of feature names
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    # diamond_features = pd.DataFrame([[carat, cut, color]],
    #                               columns=['Carat', 'Cut', 'Color'])
    # predicted_price = model.predict(diamond_features)[0]

    # print(f"\n=== New Prediction ===")
    # print(f"House features: {carat:.0f} carats, {cut} cut, {color}  color")
    # print(f"Predicted price: ${predicted_price:,.2f}")

    # return predicted_price
    
    # Your code here
    # Example: If predicting house price with [sqft, bedrooms, bathrooms]
    # sample = pd.DataFrame([[2000, 3, 2]], columns=feature_names)
    
    pass


if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data('diamond_features.csv')
    
    # Step 2: Visualize
    visualize_data(data)
    
    # Step 3: Prepare and split
    X_train, X_test, y_train, y_test = prepare_and_split_data(data)
    
    # # Step 4: Train
    # model = train_model(X_train, y_train, ['carat', 'cut', 'color'])
    
    # # # Step 5: Evaluate
    # predictions = evaluate_model(model, X_test, y_test, ['carat', 'cut', 'color'])
    
    # # # Step 6: Make a prediction, add features as an argument
    # make_prediction(model, 1.2, "Good", "I")
    
    # print("\n" + "=" * 70)
    # print("PROJECT COMPLETE!")
    # # print("=" * 70)
    # print("\nNext steps:")
    # print("1. Analyze your results")
    # print("2. Try improving your model (add/remove features)")
    # print("3. Create your presentation")
    # print("4. Practice presenting with your group!")
