# Census Income Prediction
A supervised machine learning project that predicts whether an individual's income exceeds $50K/year using the UCI Adult Census dataset. The project covers data cleaning, preprocessing, exploratory data analysis, feature encoding, outlier handling, model training, hyperparameter tuning, and logical dummy dataset testing.

*Project Overview*
The goal was to create a robust classification model to predict income levels based on demographic and work-related attributes.
 
Steps included:
1. Preprocessing the dataset by handling missing values and encoding features.
2. Performing exploratory data analysis (EDA) to understand feature relationships.
3. Detecting and handling outliers for cleaner training data.
4. Scaling and normalizing features for better model convergence.
5. Testing multiple ML algorithms and fine-tuning the best-performing model.
6. Creating a logical dummy dataset to test model generalization.

*Dataset & Features*
Dataset columns:
Age, Workclass, Fnlwgt, Education, Education_num, Marital_status, Occupation, Relationship, Race, Sex, Capital_gain, Capital_loss, Hours_per_week, Native_country, Income

Features used for model training:
["Workclass", "Education", "Marital_status", "Occupation", "Relationship", "Sex", "Native_country"]

*Steps Performed*
1. Missing Value Handling
Selected key categorical features for imputation:
["Workclass", "Education", "Marital_status", "Occupation", "Relationship", "Sex", "Native_country"]
Replaced missing values with np.nan, then dropped them in-place.

2. Label Encoding
Applied LabelEncoder to:
["Workclass", "Education", "Occupation", "Marital_status", "Relationship", "Race", "Sex", "Native_country", "Income"]

4. Exploratory Data Analysis (EDA)
Count plots for:
["Workclass", "Education","Education_num", "Occupation", "Marital_status", "Relationship", "Race", "Sex", "Native_country", "Capital_gain", "Capital_loss", "Hours_per_week", "Income"]
Used distplot and countplot to explore distributions.
Checked feature correlation matrix and plotted boxplots.

4. Outlier Detection & Handling
Used IQR method to detect outliers.
Special check for Fnlwgt (a continuous feature often mistaken as categorical due to large spread).

5. Feature Scaling
Applied PowerTransformer for normalization.
Used StandardScaler to standardize feature values.

6. Model Training & Evaluation
Baseline model: Logistic Regression.
Handled class imbalance with SMOTE oversampling.

Tested multiple models:
Logistic Regression (LR)
Decision Tree (DT)
Gradient Boosting (GB)
Support Vector Classifier (SVC)
Extreme Gradient Boosting (XGBoost, XGBT)
Performed cross-validation on each model.
Best model: XGBoost →
ROC-AUC: 0.91
Cross-validation score: ~86.6%

7. Hyperparameter Tuning
Used GridSearchCV to optimize XGBoost parameters.

8. Model Saving
Exported trained XGBoost model using joblib for reuse.

9. Dummy Dataset Testing
Created a logical dummy dataset with realistic attribute combinations.
Passed it through the preprocessing + model pipeline.

Verified predictions aligned with expected trends.

📈 Results
Best model: XGBoost
ROC-AUC: 0.91
Cross-validation: 86.6%
Successfully generalized to unseen dummy data without major performance drop with accuracy at: 82% 
on a 50 row dataset.

💻 Technologies Used
Python 3.11
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
XGBoost
Imbalanced-learn (SMOTE)
Joblib

*How to Run*
1. Have Anaconda Navigator with Jupyter Notebook installed on your device
2. Download the census_income.csv dataset, census_dummy_50_logical.csv dataset and the Training Project 1.ipynb and Training Project 1 Testing.ipynb
3. Have the Paths specified to whereever the files have been saved, in the ipynb files and run the cells.
4. After the model (.pkl) file has been created from Training Project 1.ipynb, run the Training Project 1 Testing.ipynb
5. The Results will be visible

*Future Enhancements*
Try ensemble methods with model stacking.

# © Copyright & Usage Disclaimer
© 2025 Keigan Cardoza. All rights reserved.

This project was developed as part of the IntrnForte – Microsoft × Tally Training Internship. It represents my own work, including all code, analysis, and dataset generation, and is protected under applicable copyright laws.
The contents of this repository are not intended for public use, redistribution, or commercial purposes. They are provided solely for testing, demonstration, and educational review purposes.
By accessing or using any part of this repository, you acknowledge that:
1. You will not claim this work as your own or misrepresent its authorship.
2. You will not redistribute or commercially exploit any part of the code, datasets, or documentation without prior written permission from the author.
