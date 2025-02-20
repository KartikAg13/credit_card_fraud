# Credit Card Fraud Detection using Anomaly Detection

Predict fraudulent transactions in credit card data using the Isolation Forest algorithm. This project demonstrates an unsupervised approach to identifying anomalies (fraud) in an extremely imbalanced dataset.

## Overview

Credit card fraud detection is challenging due to the rarity of fraudulent transactions compared to normal ones. In this project, the Isolation Forest algorithm is used to isolate anomalies by randomly partitioning the data. The Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset is used for this purpose.

## Repository Contents

- **main.ipynb**: Jupyter Notebook containing the complete code and analysis.
- **LICENSE**: The license for this project.
- **README.md**: This file.

## Project Workflow

### 1. Import Libraries
The notebook imports the required libraries including:
- `pandas`, `numpy` for data manipulation.
- `matplotlib`, `seaborn` for visualization.
- Modules from `scikit-learn` for preprocessing, model building, and evaluation.

### 2. Load the Dataset
The dataset is downloaded from Kaggle and loaded using `pandas.read_csv`.

### 3. Preprocess the Data
- **Normalization**: The **Amount** column is scaled using `StandardScaler`.
- **Dropping Irrelevant Features**: The **Time** column is dropped.
- **Label Conversion**:  
  - Original labels: Fraud (1) and Normal (0).
  - Converted to match Isolation Forest’s output: Fraud is set to `-1` and Normal to `1`.

### 4. Train-Test Split
The data is split into training (80%) and testing (20%) sets.

### 5. Train the Model
An Isolation Forest model is trained with a contamination parameter (set to 0.35% fraud rate) to indicate the expected fraction of outliers (fraud).

### 6. Make Predictions
Predictions are made on the test set and then mapped back:
- **Isolation Forest outputs**: `-1` (anomaly) and `1` (inlier).
- **Mapped to original labels**: `-1` becomes Fraud (`1`), and `1` becomes Normal (`0`).

### 7. Evaluate the Model
Several evaluation metrics are used to assess performance:
- **Confusion Matrix**: Displays True Negatives, False Positives, False Negatives, and True Positives.
- **Classification Report**: Provides precision, recall, F1-score, and support.
- **Prediction Count Plot**: Visualizes the number of predictions for each class.
- **Anomaly Scores**: Derived from the model’s decision function.
- **Average Precision Score (AUPRC)**: Summarizes the Precision-Recall curve, which is useful for imbalanced data.
- **Precision-Recall Curve**: Illustrates the trade-off between precision and recall.
- **F2-Score**: Emphasizes recall (with beta = 2), important in fraud detection.
- **ROC-AUC Score & ROC Curve**: Evaluate the model’s ability to distinguish between fraud and normal transactions.

## How to Run the Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/KartikAg13/credit_card_fraud.git
   ```
2. **Navigate to the Repository Directory:**
   ```bash
   cd credit_card_fraud
   ```
3. **Install Dependencies:**
   Ensure you have Python installed, then install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
4. **Run the Notebook:**
   Launch Jupyter Notebook and open `main.ipynb`:
   ```bash
   jupyter notebook main.ipynb
   ```
   Run all cells to execute the code and view the results.

## Evaluation Metrics Explained

- **Confusion Matrix:**  
  Provides a summary of prediction results:
  - **True Negatives (TN):** Correctly predicted normal transactions.
  - **False Positives (FP):** Normal transactions incorrectly predicted as fraud.
  - **False Negatives (FN):** Fraud transactions missed by the model.
  - **True Positives (TP):** Correctly predicted fraud transactions.

- **Classification Report:**  
  Contains:
  - **Precision:** The ratio of correctly predicted fraud cases to the total predicted as fraud.
  - **Recall:** The ratio of correctly detected fraud cases to all actual fraud cases.
  - **F1-Score:** The harmonic mean of precision and recall.
  - **Support:** The number of instances for each class.

- **Average Precision Score (AUPRC):**  
  Summarizes the area under the Precision-Recall curve, especially useful for imbalanced datasets.

- **Precision-Recall Curve:**  
  Plots the trade-off between precision and recall at different thresholds.

- **F2-Score:**  
  A variant of the F1-score that emphasizes recall more (with beta = 2), prioritizing the detection of fraudulent transactions.

- **ROC-AUC Score & ROC Curve:**  
  Measure the model's ability to distinguish between classes. The ROC curve plots True Positive Rate against False Positive Rate; a higher area under the curve indicates better performance.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Contact

For questions, suggestions, or contributions, please contact [KartikAg13](https://github.com/KartikAg13).
