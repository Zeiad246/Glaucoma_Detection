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
