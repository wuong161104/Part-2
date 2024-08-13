import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib 

# Load the dataset
file_path = 'C:/Users/vuong/Python/PART 2/sale_table.csv'
data = pd.read_csv(file_path)

print("Dataset preview: ")
print(data.head())

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Convert categorical variables to dummy/indicator variables
data = pd.get_dummies(data)

# Specify the target variable
target_variable = 'Jan'

# Check if the target variable exists in the dataset
if target_variable not in data.columns:
    raise KeyError(f"Target column '{target_variable}' not found in dataset columns: {data.columns}")

# Separate features and target variable
x = data.drop(target_variable, axis=1)
y = data[target_variable]

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train the Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Save the model and scaler
joblib.dump(model, 'linear_regression_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Plot the results
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual Jan Sales')
plt.ylabel('Predicted Jan Sales')
plt.title('Actual vs. Predicted Jan Sales')
plt.grid(True)
plt.show()

# Save the results to a CSV file
results = pd.DataFrame({'Actual Jan Sales': y_test, 'Predicted Jan Sales': y_pred})
results.to_csv('predicted_jan_sales.csv', index=False)
print("Results saved to predicted_jan_sales.csv")
