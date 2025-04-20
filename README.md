# Credit Card Fraud Detection

## Overview
This repository contains a machine learning project focused on detecting fraudulent credit card transactions. Credit card fraud detection is a critical application of machine learning in the financial sector, helping to identify and prevent unauthorized transactions.

## Dataset
The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle, which contains transactions made by European cardholders in September 2013. The dataset includes:

- Anonymized credit card transactions labeled as fraudulent or genuine
- 284,807 transactions with 31 features
- Features V1 through V28 are principal components obtained with PCA transformation
- Features 'Time' and 'Amount' are not transformed
- Highly imbalanced data (only 0.172% of transactions are fraudulent)

## Project Structure
- `credit_card_fraud_detection.ipynb`: The main Jupyter notebook containing data exploration, preprocessing, model building, and evaluation
- Data visualization and analysis of the imbalanced dataset
- Implementation of various classification algorithms
- Evaluation metrics appropriate for imbalanced datasets

## Implemented Models
The project implements and compares several machine learning algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## Key Features
- **Data Preprocessing**: Handling of imbalanced data using resampling techniques
- **Feature Engineering**: Analysis and selection of important features
- **Model Evaluation**: Using metrics suitable for imbalanced datasets including:
  - Confusion Matrix
  - Precision, Recall, F1-Score
  - ROC Curve and AUC Score
  - Classification Report

## Results
The models are evaluated based on their ability to identify fraudulent transactions while minimizing false positives. Results demonstrate the effectiveness of ensemble methods like Random Forest and XGBoost for this specific problem.

## Usage
To use this project:

1. Clone the repository:
```bash
git clone https://github.com/anshmittal2004/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

4. Open and run the Jupyter notebook:
```bash
jupyter notebook credit_card_fraud_detection.ipynb
```

## Future Improvements
- Implementation of deep learning models (neural networks)
- Hyperparameter tuning for better model performance
- Deployment as a real-time fraud detection system
- Cost-sensitive learning implementation

## License
This project is available under the MIT License.

## Acknowledgments
- The dataset is provided by Kaggle and was originally collected by the [ULB Machine Learning Group](https://www.ulb.ac.be/di/map/adalpozz/data/creditcard.csv)
- Thanks to all contributors who have helped improve this fraud detection system
