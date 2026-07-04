# Heart Disease Data Preprocessing Pipeline

A Python-based data preprocessing pipeline for the UCI Heart Disease dataset. This project demonstrates essential data preparation techniques used in machine learning workflows, including exploratory data analysis (EDA), feature encoding, feature scaling, and automated preprocessing.

## Overview

High-quality data preprocessing is a critical step before training machine learning models. This project prepares raw heart disease data by applying common preprocessing techniques to produce a clean, model-ready dataset.

The workflow was developed as part of a machine learning experimentation project.

## Features

- Exploratory Data Analysis (EDA)
- Missing value inspection
- Categorical feature encoding
- Numerical feature scaling
- Automated preprocessing script
- Export of processed dataset for downstream machine learning tasks

## Project Structure

```
Eksperimen_SML_WincentCP/
│
├── namadataset_raw/
│   └── heart.csv
│
├── preprocessing/
│   ├── Eksperimen_WincentCP.ipynb
│   ├── automate_WincentCP.py
│   └── namadataset_preprocessing/
│       └── heart_preprocessed.csv
│
└── .gitignore
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## Data Preprocessing Workflow

The preprocessing pipeline includes:

1. Loading the raw dataset.
2. Performing exploratory data analysis (EDA).
3. Encoding categorical variables.
4. Scaling numerical features using StandardScaler.
5. Preparing a clean dataset suitable for machine learning.
6. Exporting the processed dataset for future model development.

## Dataset

Dataset: Heart Disease Dataset

This project uses the widely adopted Heart Disease dataset for educational and machine learning experimentation purposes.

## Output

The preprocessing pipeline generates:

```
heart_preprocessed.csv
```

This processed dataset is ready for machine learning model training and evaluation.

## Learning Objectives

This project demonstrates practical experience with:

- Data preprocessing
- Feature engineering
- Data normalization
- Machine learning data preparation
- Python data analysis workflow
- Reusable preprocessing scripts

## Future Improvements

- Automated train-test splitting
- Feature selection
- Model training and evaluation
- Hyperparameter tuning
- MLflow experiment tracking
- Docker containerization
- CI/CD pipeline integration
