# Glaucoma Detection Analysis
## Introduction
The Team started a complete analysis on a Glaucoma dataset, all the way from cleaning, restructuring, and analysing of the data set to explore new ways of detecting Glaucoma.

## Let's Frame the Problem:
  What are the pre-processing steps used that made the most difference in this data?
  Which classifier models were best for Diagnosis and Glaucoma Type?
  What were the results obtained from the clustering algorithm selected for this problem? (K-Means, DBScan, Agglomerative)

## Step 1, Preparing our data
  df.info(), Useful information, especially the type of variable of each feature.
  df.describe(), General description of the dataset.
  df.head(), A surface-level preview of what the data looks like especially after the ; separator being implemented.
  df.columns, Feature names.
  For better visualization, we separated non-atomic columns into multiple columns of continuous values, the following was changed:
    Visual Field Test Results -> Sensitivity and Specifity.
    OCT Results -> RNF Thickness, GCC Thickness, Retinal Volume, and Macular Thickness.


