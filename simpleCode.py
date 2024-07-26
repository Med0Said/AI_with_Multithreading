import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

# Measure the start time
start_time = time.time()

# Define column names based on the dataset documentation
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
    'hours_per_week', 'native_country', 'income'
]
def perform_eda():
    sns.countplot(x='income', data=df1)
    plt.show()

    sns.countplot(x='income', hue='sex', data=df1)
    plt.show()

    sns.pairplot(df1[['age', 'education_num', 'hours_per_week', 'income']], hue='income', markers=["o", "s"])
    plt.show()

# Load the dataset from local file
file_path = "C:/Users/said0/Downloads/adult.data"
df = pd.read_csv(file_path, names=column_names, sep=r'\s*,\s*', engine='python')

# Basic info
print(df.info())

# Handle missing values
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Convert categorical variables to numerical
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop('income_>50K', axis=1)
y = df['income_>50K']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build and evaluate the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Measure the end time
end_time = time.time()
execution_time = end_time - start_time

# Display the execution time
print(f"Total execution time: {execution_time:.2f} seconds")

# Perform EDA (after preprocessing to avoid data corruption)
df1 = pd.read_csv(file_path, names=column_names, sep=r'\s*,\s*', engine='python')
# Perform EDA
perform_eda()
