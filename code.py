# Simple House Price Prediction Model for Beginners
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a sample dataset
print("Creating sample house data...")
np.random.seed(0)  # For reproducible results

# Generate sample house features
data = {
    'size_sqft': np.random.normal(2000, 500, 100),  # House size
    'bedrooms': np.random.randint(1, 5, 100),       # Number of bedrooms
    'bathrooms': np.random.randint(1, 3, 100),      # Number of bathrooms
    'age_years': np.random.randint(0, 30, 100),     # House age
    'location': np.random.choice(['Downtown', 'Suburb', 'Rural'], 100),  # Location
    'garage': np.random.choice(['Yes', 'No'], 100),     # Has garage
    'garden': np.random.choice(['Yes', 'No'], 100),     # Has garden
    'pool': np.random.choice(['Yes', 'No'], 100)        # Has pool
}

# Create DataFrame
df = pd.DataFrame(data)
print(f"Dataset created with {len(df)} houses")
print(df.head())

# Step 2: Calculate house prices based on features
print("\nCalculating house prices...")

# Base price calculation
base_price = (df['size_sqft'] * 150 + 
              df['bedrooms'] * 10000 + 
              df['bathrooms'] * 15000 - 
              df['age_years'] * 2000)

# Location premiums
location_premium = df['location'].map({
    'Downtown': 50000, 
    'Suburb': 20000, 
    'Rural': 0
})

# Amenity premiums
garage_premium = df['garage'].map({'Yes': 15000, 'No': 0})
garden_premium = df['garden'].map({'Yes': 10000, 'No': 0})
pool_premium = df['pool'].map({'Yes': 25000, 'No': 0})

# Add some random noise to make it realistic
noise = np.random.normal(0, 20000, 100)

# Final price calculation
df['price'] = (base_price + location_premium + garage_premium + 
               garden_premium + pool_premium + noise)

print(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")

# Step 3: Prepare data for machine learning
print("\nPreparing data for machine learning...")

# Separate features (X) and target (y)
X = df.drop('price', axis=1)  # Features
y = df['price']               # Target (price)

# Step 4: One-hot encoding for categorical variables
print("Applying one-hot encoding...")
X_encoded = pd.get_dummies(X, drop_first=True)
print(f"Features after encoding: {list(X_encoded.columns)}")

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=0
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Step 6: Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train the linear regression model
print("Training the model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 8: Make predictions
print("Making predictions...")
y_pred = model.predict(X_test_scaled)

# Step 9: Evaluate the model
print("\nModel Performance:")
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print(f"R-squared Score: {r2:.4f} ({r2*100:.2f}%)")
print(f"RMSE: ${rmse:,.2f}")

# Step 10: Show feature importance
print("\nFeature Importance (Coefficients):")
feature_importance = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"{row['Feature']}: {row['Coefficient']:,.0f}")

# Step 11: Show some prediction examples
print("\nPrediction Examples:")
for i in range(5):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    error = abs(actual - predicted)
    print(f"House {i+1}: Actual ${actual:,.0f}, Predicted ${predicted:,.0f}, Error ${error:,.0f}")

# Step 12: Simple prediction function
def predict_new_house(size_sqft, bedrooms, bathrooms, age_years, 
                     location, garage, garden, pool):
    """
    Predict price for a new house
    """
    # Create input data
    new_house = pd.DataFrame({
        'size_sqft': [size_sqft],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'age_years': [age_years],
        'location': [location],
        'garage': [garage],
        'garden': [garden],
        'pool': [pool]
    })
    
    # Apply same encoding and scaling
    new_house_encoded = pd.get_dummies(new_house, drop_first=True)
    
    # Make sure all columns are present
    for col in X_encoded.columns:
        if col not in new_house_encoded.columns:
            new_house_encoded[col] = 0
    
    # Reorder columns to match training data
    new_house_encoded = new_house_encoded[X_encoded.columns]
    
    # Scale and predict
    new_house_scaled = scaler.transform(new_house_encoded)
    prediction = model.predict(new_house_scaled)[0]
    
    return prediction

# Example prediction
print("\nExample: Predicting price for a new house...")
example_price = predict_new_house(
    size_sqft=2200,
    bedrooms=3,
    bathrooms=2,
    age_years=5,
    location='Downtown',
    garage='Yes',
    garden='Yes',
    pool='No'
)

print(f"Predicted price for the example house: ${example_price:,.0f}")

print(f"\n✅ Model successfully trained!")
print(f"✅ Achieved R² score of {r2:.4f} (target was ~0.85)")
print(f"✅ RMSE: ${rmse:,.0f}")
