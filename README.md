# Glaucoma Detection Analysis
## Introduction
The Team started a complete analysis on a Glaucoma dataset, all the way from cleaning, restructuring, and analysing of the data set to explore new ways of detecting Glaucoma.

## Let's Frame the Problem:
  What are the pre-processing steps used that made the most difference in this data?
  Which classifier models were best for Diagnosis and Glaucoma Type?
  What were the results obtained from the clustering algorithm selected for this problem? (K-Means, DBScan, Agglomerative)

## Step 1, Preparing our data.
  df.info(), Useful information, especially the type of variable of each feature.\
  df.describe(), General description of the dataset.\
  df.head(), A surface-level preview of what the data looks like especially after the ; separator being implemented.\
  df.columns, Feature names.\
  For better visualization, we separated non-atomic columns into multiple columns of continuous values, the following was changed:\
    Visual Field Test Results -> Sensitivity and Specifity.\
    OCT Results -> RNF Thickness, GCC Thickness, Retinal Volume, and Macular Thickness.

    Patient ID, Medication Usage, and Visual Symptoms do not give us any information that we can use for data analysis.
We call for a domain/field expert for the purpose of providing more informative results in the future such that we do not drop the columns.

We then encode the categorical data, such as: Gender, Family History, Medical History, Angle Closure Status, Diagnosis, Glaucoma Type, Visual Acuity Measurements (LogMAR) and visualize them\

We then visualize the quantitative features, such as:'Age', 'Intraocular Pressure (IOP)', 'Cup-to-Disc Ratio (CDR)',
'Pachymetry', 'Sensitivity', 'Specificity', 'RNFL Thickness','GCC Thickness', 'Retinal Volume', 'Macular Thickness'

Results in: The first 3 plots indicate nearly equal observations for each category./
LogMAR however had more cases of 0.0 vision in comparison to 0.1 and 0.3./
We then start visualizing the quantitative multivariate features by doing a heatmap (Correlation Matrix of quantitative features)/

This matrix indicates almost no correlation between the quantitative independent features because they either give the correlation value of 0, or they are very close to 0./

## Step 2, PreProcessing the data.

df.isnull().sum(), resulted in 0 for every category./
Splitting the independent and dependent features, This will be used later when we get to the model training./
X = df.drop(['Diagnosis', 'Glaucoma Type'], axis=1)/
y1 = df['Diagnosis']/
y2 = df['Glaucoma Type']/

Outlier treatment using Univariate Outlier Detection (Tukey's Boxplot)./
![image](https://github.com/Zeiad246/Glaucoma_Detection/assets/151476551/d994b06a-03d1-4e56-ad2d-ef3103f1109b)/
Despite the different positions of these boxplots, there are no outliers present here. We will utilize a multivariate outlier detection method as a better outlier identifier./

Then, we perform Multivariate Outlier Detection (Mahalanobis Distance):/
MD distance was selected because it provided us with a good estimation of outliers. We used the parameter "3" instead of "1.5" when setting the threshold conditions because the data does not contain much variance./
DBScan was not selected in this problem because although it will be able to detect outliers between points, the challenge is to find the best hyperparameters "minPts" and "epsilon". It would require a Grid Search which is computationally expensive given that we are finding outliers between the independent features./








