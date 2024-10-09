import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../main/static/main/files/iris.csv')

categorical_columns = ['Species']

label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

df.to_csv('../main/static/main/files/iris.csv', index=False)
print(df.head())

