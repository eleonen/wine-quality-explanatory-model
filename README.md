# Red Wine Quality Analysis

## **Project Overview**
This repository contains an analysis of the Red Wine Quality dataset, which examines the relationship between physicochemical features and perceived wine quality. The project focuses on explanatory modeling using linear regression and logistic regression to identify factors most strongly associated with wine quality.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Analysis Overview](#analysis-overview)
7. [Key Findings](#key-findings)
8. [Improvements and Future Work](#improvements-and-future-work)
9. [Stakeholders and Goals](#stakeholders-and-goals)
10. [Contributors](#contributors)

## Introduction
The purpose of this analysis is to explore the Red Wine Quality dataset and identify the key physicochemical variables that influence perceived wine quality. By using statistical techniques and explanatory models, we aim to provide actionable insights for stakeholders in the wine industry.

## Setup

### Prerequisites
- Python 3.x
- Poetry for dependency management
- Jupyter Notebook (optional, for viewing the analysis)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Vixamon/wine-quality-explanatory-model/
   cd wine-quality-explanatory-model
   ```

2. Set up a virtual environment and install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

4. (Optional) If using Jupyter Notebook to view or modify the analysis:  
   ```bash
   poetry add notebook
   ```

## Project Structure

- `pyproject.toml`: Poetry configuration file listing dependencies.
- `poetry.lock`: Lock file with exact package versions.
- `winequality-red.csv`: Dataset for the analysis.
- `wine_explanatory_model.ipynb`: Jupyter notebook with the analysis and insights derived from the dataset.
- `utilities.py`: Contains helper functions for data cleaning, outlier detection, and visualizations.

## Usage

### Running the Jupyter Notebook
To interact with the data analysis or run your own queries, use the Jupyter notebook:
1. Start Jupyter Notebook:
   ```bash
   jupyter notebook wine_explanatory_model.ipynb
   ```
2. Follow the cells to explore the analysis, or modify them to perform your own exploration.

## Dataset

### Red Wine Quality Dataset

- **Description:** The dataset contains physicochemical features (e.g., pH, alcohol) and quality ratings of red wine samples.
- **Columns:**
  - `fixed_acidity`: Tartaric acid concentration.
  - `volatile_acidity`: Acetic acid concentration, linked to smell and vinegar taste.
  - `citric_acid`: Citric acid concentration, contributing to the wine's freshness.
  - `residual_sugar`: Sugar left after fermentation.
  - `chlorides`: Salt concentration.
  - `free_sulfur_dioxide`: SO2 not bound and available to act as an antimicrobial.
  - `total_sulfur_dioxide`: Total SO2 in wine (bound + free).
  - `density`: Wine density, affected by sugar and alcohol content.
  - `pH`: Acidity level.
  - `sulphates`: Preservatives to prevent spoiling and oxidation, provides protection from bacteria.
  - `alcohol`: Alcohol content (% by volume).
  - `quality`: A score (0–10) representing the sensory quality of the wine.

## Analysis Overview

The analysis includes:

- **Exploratory Data Analysis (EDA):** Examination of feature distributions, correlations, and potential outliers.
- **Feature Selection:** Identification of the most important physicochemical variables.
- **Modeling:**
  - Linear regression to explain the variance in wine quality scores.
  - Logistic regression to predict wine quality categories (e.g., low vs. high).
- **Model Evaluation:** Assessment of model performance using metrics such as R-squared, residuals, and classification accuracy.

## Key Findings

- **Influential Variables:**
  - Alcohol content has a strong positive effect on wine quality.
  - Volatile acidity is the most negatively correlated variable with wine quality.
  - Other significant variables include sulphates, total sulfur dioxide, pH, and chlorides.

- **Model Performance:**
  - Linear regression explains 35% of the variance in wine quality (adjusted R-squared = 0.350).
  - Logistic regression achieves reasonable accuracy in classifying high and low-quality wines.

- **Insights:**
  - Alcohol and volatile acidity are key determinants of perceived wine quality.
  - Adjusting these variables during production could improve wine quality.

## Improvements and Future Work

1. **Dataset Enhancements:**
   - Address class imbalances in wine quality categories.

2. **Feature engineering:**
   - Experiment with polynomial and interaction terms to capture non-linear relationships.

3. **Simplified Categories:**
   - Use binary classification (e.g., high vs. low quality) to simplify the logistic regression model.

4. **Non-linear models:**
   - Test non-linear models such as random forests or gradient boosting.

## Stakeholders and Goals

### Stakeholders

- **Wine Producers:** To refine production techniques and enhance wine quality.
- **Wine Marketers:** To develop targeted marketing strategies based on quality attributes.

### Goals

- Identify physicochemical features most strongly associated with perceived wine quality.
- Provide actionable recommendations for improving production and marketing strategies.
- Develop a model that explains the key drivers of wine quality.

## Contributors
- [Erikas Leonenka](https://github.com/Vixamon)
