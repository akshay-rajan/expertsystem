import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
data = pd.read_csv('./main/static/main/files/customers.csv')

print(data.head())
# Step 2: Encode the 'Gender' column
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Display the first few rows of the dataset to verify encoding
print(data.head())

# Optional: Save the encoded dataset to a new CSV file
data.to_csv('customers.csv', index=False)