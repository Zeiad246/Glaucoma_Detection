# Glaucoma Detection Analysis
## Introduction
The Team started a complete analysis on a Glaucoma dataset, all the way from cleaning, restructuring, and analysing of the data set to explore new ways of detecting Glaucoma.

## Let's Frame the Problem:
  What are the pre-processing steps used that made the most difference in this data?
  Which classifier models were best for Diagnosis and Glaucoma Type?
  What were the results obtained from the clustering algorithm selected for this problem? (K-Means, DBScan, Agglomerative)

## Step 1, Preparing our data
  df.info(), Useful information, especially the type of variable of each feature.\
  df.describe(), General description of the dataset.\
  df.head(), A surface-level preview of what the data looks like especially after the ; separator being implemented.\
  df.columns, Feature names.\
  For better visualization, we separated non-atomic columns into multiple columns of continuous values, the following was changed:\
    Visual Field Test Results -> Sensitivity and Specifity.\
    OCT Results -> RNF Thickness, GCC Thickness, Retinal Volume, and Macular Thickness.

    Patient ID, Medication Usage, and Visual Symptoms do not give us any information that we can use for data analysis.
We call for a domain/field expert for the purpose of providing more informative results in the future such that we do not drop the columns.

We then encode the categorical data, such as: Gender, Family History, Medical History, Angle Closure Status, Diagnosis, Glaucoma Type, Visual Acuity Measurements (LogMAR)

We then visualize the quantitative features, such as:'Age', 'Intraocular Pressure (IOP)', 'Cup-to-Disc Ratio (CDR)',
'Pachymetry', 'Sensitivity', 'Specificity', 'RNFL Thickness','GCC Thickness', 'Retinal Volume', 'Macular Thickness'

