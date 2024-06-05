# Glaucoma Detection Analysis

## Introduction
The team embarked on a comprehensive analysis of a glaucoma dataset, progressing from data cleaning and restructuring to in-depth exploration of potential methods for detecting glaucoma.

## Let's Frame the Problem:
- What pre-processing steps significantly impacted the data?
- Which classifier models were most effective for diagnosing glaucoma and identifying glaucoma type?
- What were the outcomes from the clustering algorithms used (K-Means, DBScan, Agglomerative)?

## Step 1: Preparing Our Data

### Initial Exploration

- **df.info()**: Useful for understanding the type of variables for each feature.
- **df.describe()**: Provides a general description of the dataset.
- **df.head()**: Offers a surface-level preview of the data, especially after implementing the ';' separator.
- **df.columns**: Lists feature names.

### Data Restructuring

For better visualization, we separated non-atomic columns into multiple columns of continuous values:
- **Visual Field Test Results**: Separated into Sensitivity and Specificity.
- **OCT Results**: Separated into RNF Thickness, GCC Thickness, Retinal Volume, and Macular Thickness.

Patient ID, Medication Usage, and Visual Symptoms do not provide useful information for data analysis. A domain expert's input would be valuable in future analyses to avoid dropping these columns unnecessarily.

### Encoding and Visualization

We encoded categorical data such as Gender, Family History, Medical History, Angle Closure Status, Diagnosis, Glaucoma Type, and Visual Acuity Measurements (LogMAR). Quantitative features such as Age, Intraocular Pressure (IOP), Cup-to-Disc Ratio (CDR), Pachymetry, Sensitivity, Specificity, RNFL Thickness, GCC Thickness, Retinal Volume, and Macular Thickness were visualized.

Observations:
- The first three plots indicate nearly equal observations for each category.
- LogMAR had more cases of 0.0 vision compared to 0.1 and 0.3.

A heatmap (correlation matrix) indicated almost no correlation between the quantitative independent features as their values were either 0 or very close to 0.

## Step 2: Preprocessing the Data

### Missing Values

- **df.isnull().sum()**: Resulted in 0 for every category.

### Feature Splitting

- **Independent Features**: `X = df.drop(['Diagnosis', 'Glaucoma Type'], axis=1)`
- **Dependent Features**: `y1 = df['Diagnosis']`, `y2 = df['Glaucoma Type']`

### Outlier Detection

#### Univariate Outlier Detection (Tukey's Boxplot)
Despite the different positions of these boxplots, no outliers were present. Therefore, we utilized a multivariate outlier detection method as a better outlier identifier.

![Tukey's Boxplot](https://github.com/Zeiad246/Glaucoma_Detection/assets/151476551/d994b06a-03d1-4e56-ad2d-ef3103f1109b)

#### Multivariate Outlier Detection (Mahalanobis Distance)
MD distance was selected because it provided a good estimation of outliers. We used the parameter "3" instead of "1.5" when setting the threshold conditions due to the low variance in the data. DBScan was not selected as it would require a computationally expensive grid search for hyperparameters "minPts" and "epsilon".

![Mahalanobis Distance](https://github.com/Zeiad246/Glaucoma_Detection/assets/151476551/d04a3be8-cc01-49d4-8220-9edff0347766)

- Total number of outliers: 22
- Indices of outliers: [1459, 1629, 1756, 2736, 3047, 3117, 3314, 3592, 3701, 4109, 4215, 4383, 6013, 6359, 6711, 7018, 7223, 7553, 8401, 8882, 9317, 9864]

We then performed outlier removal by dropping the outliers from the dataset and resetting the indexes.

```python
df = df.drop(index=outlierPosition)
df = df.reset_index(drop=True)
```
### Scaling

We selected robust scaling based on the following criteria:
- Robust to outliers.
- Preserves the relative ordering of data points.
- Useful for non-normally distributed data.

### Feature Extraction

#### Continuous-Categorical (Fisher Score)
![Fisher Score](https://github.com/Zeiad246/Glaucoma_Detection/assets/151476551/7f4e0b12-ab7f-41ee-9793-ce10261dcd84)

For the Diagnosis Classification, Pachymetry and Retinal Volume contained the highest Fisher Scores. GCC, Macular, RNFL, and Specificity had some impact on the diagnosis classification while the others had little to no effect.

For the Glaucoma Type Classification, Retinal Volume, IOP, and CDR had the highest Fisher Scores. Almost every feature had a fairly high score as well.

#### Categorical-Categorical (Chi Square)
![Chi Square](https://github.com/Zeiad246/Glaucoma_Detection/assets/151476551/e794e381-334b-4281-b9ea-b9dabc2efe02)

For the Diagnosis Classification, Medical History contained the highest Chi Square Score. Visual Acuity Measurement had some impact on the diagnosis classification while the others had minimal effect.

For the Glaucoma Type Classification, Medical History contained the highest Chi Square Score, while Visual Acuity Measurement had some impact on the diagnosis classification. The other features had minimal effect on Glaucoma Type.

We then updated the dataframe to retain only features with the highest Fisher and Chi-Square scores for both dependent features.

### Dimensionality Reduction

We performed dimensionality reduction using PCA on train-test splits. PCA implementation was chosen to address high dimensionality in the data. We opted for PCA over kernelPCA due to its computational efficiency and ease of interpretation. We selected 10 components to explain 95% of the variance based on this dataset.

```python
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
```
### Classification

#### Diagnosis (Glaucoma vs. No Glaucoma)

We chose the following classifiers for Diagnosis:
- Logistic Regression
- Decision Trees
- KNN
- XGBoost

These classifiers are well-suited for binary classification tasks. We defined the hyperparameters for each model as follows:

```python
hyperList = {
    'Logistic Regression': {'C': [0.0001, 0.01, 1, 10, 100, 1000]},
    'Decision Tree': {'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10]},
    'KNN': {'n_neighbors': [3, 5, 7], 'p': [1, 2]},
    'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
}
```
### Classifier Hyperparameter Optimization for Glaucoma Type Prediction

For predicting Glaucoma Type, we utilized three classifiers: Decision Tree, KNN, and XGBoost. Each classifier was optimized using grid search to find the best hyperparameters.

#### Methodology

We defined a dictionary `hyperList` containing the advisable hyperparameters for each classifier. Grid search was then performed to identify the optimal combination of hyperparameters for each classifier.

A dictionary `classifiers` was created to store the classifiers along with their names. For each classifier, we initialized an empty dictionary `bestParameters` to store the best hyperparameters and a variable `bestAccuracies` to track the best accuracy achieved.

We iterated over each classifier and its associated hyperparameters defined in `hyperList`. Using a nested loop, we performed grid search by testing all possible combinations of hyperparameters.

For each combination, we trained the classifier on the training data and evaluated its performance on the test data using accuracy as the evaluation metric. We updated `bestParameters` and `bestAccuracies` if a higher accuracy was achieved.

#### Results

The results of the grid search optimization were printed for each classifier, displaying the current hyperparameters being tested and the corresponding accuracy score. This iterative process allowed us to identify the best hyperparameters for each classifier based on the highest accuracy achieved.

#### Best Hyperparameters

The best hyperparameters identified for each classifier were as follows:

- **Decision Tree**: {'max_depth': None, 'min_samples_split': 5}
- **KNN**: {'n_neighbors': 5, 'p': 1}
- **XGBoost**: {'n_estimators': 200, 'learning_rate': 0.2}

These optimized hyperparameters will be used for predicting Glaucoma Type on new data instances.

#### Classification Reports

Detailed classification reports were generated for each classifier, providing insights into their performance across different classes. Each classification report included the following metrics:

- **Precision**: Precision measures the ratio of correctly predicted instances of a class to the total instances predicted as that class. It indicates the ability of the classifier to avoid false positives.
- **Recall**: Recall, also known as sensitivity, measures the ratio of correctly predicted instances of a class to the total instances of that class in the dataset. It indicates the ability of the classifier to identify all relevant instances.
- **F1-score**: F1-score is the harmonic mean of precision and recall. It provides a balance between precision and recall, making it a useful metric for evaluating classifiers in imbalanced datasets.
- **Support**: Support is the number of actual occurrences of each class in the dataset.

These metrics were calculated for each class in the dataset, providing a comprehensive evaluation of the classifiers' performance across all classes.

#### Best Classifier

The best classifier for this dataset was KNN, achieving an accuracy score of 16.63% on predicting Glaucoma Type.


