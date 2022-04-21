# Comprehensive-prediction-analysis-of-alcohol-and-drug-use-disorder-using-machine-deep-learning-algor
This repository represents our work in the paper `Comprehensive prediction analysis of alcohol and drug use disorder using machine/deep learning algorithms` ![flowchart](images/figure1-1.png)

# Dataset
An observational cohort of 6978 adults was admitted in the western region of Alabama at three medical facilities, the screening, brief intervention, and referral to treatment in Alabama (AL-SBIRT) program between February 2020  and December 2020.

## Pre-processing the dataset
The cleaning data methods and advanced pre-processing (e.g., a missing data imputation technique and an augmented sampling data method ) of Electronic Health Records (EHRs) were performed. 
### The first stage of pre-processing the dataset
This is initial-processing step.  Start the notebook server from the command line:

```bash
jupyter notebook
```

See (`initial-preprocessing/Cleaned the data (initial pre-processing)`). 

### The second stage of pre-processing the dataset
 Accordingly, to develop more successful models, the data pre-processing techniques are used (``see preprocessing_Stage_2``) :
 
  - Handling missing data

  - Synthetic Minority Over-sampling Technique for Nominal and Continuous features (SMOTE-NC)

## Our data analysis was dependent on three main experiments 
 - Experiment (1): we applied the machine learning/deep learning models for classification of DAST/AUDIT scores using all the features as predictors.
 - Experiment (2): we  applied the features selection model before training the machine learning/deep learning models.
 - Experiment (3): we applied  mixed-effect for classification of DAST/AUDIT scores.
