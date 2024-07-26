# AI_with_Multithreading

This project involves a comprehensive analysis and classification of the Adult Income dataset, employing multithreading to optimize performance. The project is structured to preprocess the data, train and evaluate a RandomForestClassifier, and perform Exploratory Data Analysis (EDA) to visualize key aspects of the data. 

## Project Structure

- **Data Loading**: The dataset is loaded from a local file.
- **Multithreading**: Utilizes threading to perform data preprocessing and model training simultaneously, reducing overall execution time.
- **EDA**: Includes visualizations to understand the distribution and relationships in the data.
- **Model Training and Evaluation**: Trains a RandomForestClassifier and evaluates its performance using metrics like accuracy, confusion matrix, and classification report.
- **Execution Time Measurement**: Measures and displays the total execution time for the operations.

## Key Features

### Data Preprocessing

- Handles missing values by replacing them with NaNs and dropping rows with missing values.
- Converts categorical variables into numerical variables using one-hot encoding.

### Model Training and Evaluation

- Splits the dataset into training and testing sets.
- Trains a RandomForestClassifier on the training set.
- Evaluates the model using metrics such as accuracy, confusion matrix, and classification report.

### Exploratory Data Analysis (EDA)

- **Income Distribution**: Visualizes the distribution of income levels in the dataset.
- **Income Distribution by Sex**: Compares the distribution of income levels across different sexes.
- **Pairplot of Selected Features**: Visualizes relationships between key features like age, education number, hours per week, and income.

### Multithreading

- Implements threading to run data preprocessing and model training in parallel.
- Uses mutex locks to ensure thread-safe operations.

### Execution Time Measurement

- Measures the total execution time for data preprocessing, model training, and EDA steps.
- Displays the execution time to highlight the efficiency gains from using multithreading.

## Instructions

### Prerequisites

- Python 3.8 or higher
- Required Python packages: pandas, seaborn, matplotlib, scikit-learn, threading, time

### Downloading the Dataset

1. Download the Adult Income dataset from the UCI Machine Learning Repository:
   [Adult Income Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)

2. Save the downloaded file (`adult.data`) to a local directory, for example, `C:/Users/said0/Downloads/adult.data`.

### Running the Project

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Install the necessary Python packages using pip:
   ```bash
   pip install pandas seaborn matplotlib scikit-learn
   ```

3. **Ensure the File Path is Correct**:
   Verify that the file path in the script points to the location of `adult.data` on your machine:
   ```python
   file_path = "C:/Users/said0/Downloads/adult.data"
   ```

4. **Run the Script**:
   Execute the script:
   ```bash
   python code_multy.py
   ```

5. **View Results**:
   - The script will output model performance metrics, including confusion matrix, classification report, and accuracy score.
   - The EDA plots will be displayed, providing insights into the data distribution and relationships.

## Conclusion

This project demonstrates an efficient approach to data analysis and classification using the Adult Income dataset. By leveraging multithreading, the project reduces execution time and enhances performance. The comprehensive EDA and model evaluation provide valuable insights and reliable predictions on the dataset.
