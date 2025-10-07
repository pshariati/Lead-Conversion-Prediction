# Lead-Conversion-Prediction

Predict which marketing leads are most likely to convert using supervised learning on behavioral and demographic data.

## Overview
This project compares several supervised ML classifiers to identify high-conversion leads and help sales teams prioritize outreach.  
Includes:
- EDA on web/app interactions and lead demographics  
- Feature engineering (interaction frequencies, encodings, engagement metrics)  
- Model training and tuning (Logistic Regression, Decision Tree, SVM, Random Forest)  
- Threshold optimization based on business objectives  

## Data
The dataset includes lead behavior, demographics, and interaction history with a target label `status` (1 = converted, 0 = not converted).  
> **Note:** Data not included — update notebook paths as needed.

## Methods
- **Models:** Logistic Regression, Decision Tree, SVM, Random Forest  
- **Libraries:** scikit-learn, pandas, numpy  
- **Evaluation Metrics:** Precision, Recall, F1-score, ROC–AUC  
- **Thresholding:** Adjusted classification thresholds for optimal recall on converted leads  

## Reproducing Results
1. Load the dataset and update file paths in the notebook.  
2. Run EDA, preprocessing, and model training cells in sequence.  
3. Compare classifiers and tune hyperparameters via GridSearchCV.  
4. Use the precision–recall curve to select an operating threshold.

## Notes
- Class imbalance handled through weighted models and thresholding.  
- Random Forest achieved the highest performance overall.  

