# Glaucoma Detection Analysis

## Introduction

Our team embarked on a comprehensive analysis of a glaucoma dataset, progressing from data cleaning and restructuring to an in-depth exploration of potential methods for detecting glaucoma.

## Problem Framing

1. **Pre-processing Steps**: What pre-processing steps significantly impacted the data?
2. **Classifier Models**: Which classifier models were most effective for diagnosing glaucoma and identifying glaucoma type?
3. **Clustering Algorithm Results**: What were the outcomes from the clustering algorithms used (K-Means, DBScan, Agglomerative)?

## Step 1: Preparing Our Data

### Initial Exploration

- **Commands Used**: `df.info()`, `df.describe()`, `df.head()`, `df.columns`
- **Objective**: Understand the type and distribution of data.

### Data Restructuring

- **Column Separation**: Split non-atomic columns into multiple continuous columns.
    - Visual Field Test Results: Separated into Sensitivity and Specificity.
    - OCT Results: Separated into RNF Thickness, GCC Thickness, Retinal Volume, and Macular Thickness.
- **Column Dropping**: Removed Patient ID, Medication Usage, and Visual Symptoms due to lack of relevance.

### Encoding and Visualization

- **Categorical Encoding**: Encoded variables such as Gender, Family History, Medical History, etc.
- **Quantitative Visualization**: Visualized features like Age, IOP, CDR, etc. Observed distribution and potential correlations.
- **Correlation Matrix**: Checked for correlations between quantitative features, found nearly no significant correlation.

## Step 2: Preprocessing the Data

### Missing Values

- **Command Used**: `df.isnull().sum()`
- **Result**: No missing values found.

### Feature Splitting

- **Independent Features**: `X = df.drop(['Diagnosis', 'Glaucoma Type'], axis=1)`
- **Dependent Features**: `y1 = df['Diagnosis']`, `y2 = df['Glaucoma Type']`

### Outlier Detection

- **Univariate Detection**: Used Tukey's Boxplot, found no outliers.
- **Multivariate Detection**: Used Mahalanobis Distance, identified and removed 22 outliers.

### Scaling

- **Scaling Method**: Used robust scaling for its robustness to outliers and preservation of data ordering.

### Feature Extraction

- **Continuous-Categorical**: Used Fisher Score to identify key features.
- **Categorical-Categorical**: Used Chi-Square to further refine key features.
- **Dataframe Update**: Retained only features with high Fisher and Chi-Square scores.

### Dimensionality Reduction

- **Method Used**: PCA (Principal Component Analysis)
- **Objective**: Reduced to 10 components explaining 95% variance.

## Step 3: Classification

### Classification for Diagnosis (Glaucoma vs. No Glaucoma)

- **Selected Classifiers**: 
    - Logistic Regression
    - Decision Trees
    - KNN
    - XGBoost
- **Hyperparameters**: Tuned hyperparameters for each model using Grid Search.
    - Example for Logistic Regression: `{'C': [0.0001, 0.01, 1, 10, 100, 1000]}`
    - Example for Decision Tree: `{'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10]}`
    - Example for KNN: `{'n_neighbors': [3, 5, 7], 'p': [1, 2]}`
    - Example for XGBoost: `{'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}`
