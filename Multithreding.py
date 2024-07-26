import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import threading
import time

# Define column names based on the dataset documentation
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
    'hours_per_week', 'native_country', 'income'
]

# Load the dataset from local file
file_path = "adult.data"
df = pd.read_csv(file_path, names=column_names, sep=r'\s*,\s*', engine='python')

# Initialize a mutex lock
lock = threading.Lock()

# Data Preprocessing Thread
def preprocess_data(df):
    with lock:
        # Handle missing values
        df.replace('?', pd.NA, inplace=True)
        df.dropna(inplace=True)

        # Convert categorical variables to numerical
        df = pd.get_dummies(df, drop_first=True)
        
        # Features and target
        global X, y
        X = df.drop('income_>50K', axis=1)
        y = df['income_>50K']

        print("Data preprocessing completed.")

# Model Training and Evaluation Thread
def train_and_evaluate_model():
    with lock:
        # Split the data
        global X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Build and evaluate the model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Measure the start time
start_time = time.time()

# Create threads for preprocessing and model training
preprocess_thread = threading.Thread(target=preprocess_data, args=(df,))
train_thread = threading.Thread(target=train_and_evaluate_model)

# Start the threads
preprocess_thread.start()
preprocess_thread.join()  # Ensure preprocessing is complete before training

train_thread.start()
train_thread.join()  # Wait for the training and evaluation to complete

# Measure the end time
end_time = time.time()
execution_time = end_time - start_time

# Display the execution time
print(f"Total execution time: {execution_time:.2f} seconds")

# Perform EDA (after preprocessing to avoid data corruption)
def perform_eda():
    sns.countplot(x='income', data=df)
    plt.show()

    sns.countplot(x='income', hue='sex', data=df)
    plt.show()

    sns.pairplot(df[['age', 'education_num', 'hours_per_week', 'income']], hue='income', markers=["o", "s"])
    plt.show()

# Perform EDA
perform_eda()
