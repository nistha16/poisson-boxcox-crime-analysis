# Crime Analysis in London: Box–Cox and Regression Methods

A Python research project exploring how Box–Cox transformations affect the performance of Poisson and linear regression models when analysing crime data across London.

The project investigates whether transforming numerical features can improve predictive accuracy and whether the benefits differ depending on the type of data being modelled.

## Project Overview

The research focuses on comparing statistical approaches to modelling crime patterns across London boroughs.

The analysis explores three main areas:

- **Crime density:** Investigating how transformations affect predictions of crime density.
- **Crime counts:** Comparing regression methods for predicting the number of reported crimes.
- **Distance to police stations:** Exploring whether distance-related features benefit from Box–Cox transformations.

The aim was to understand when data transformations improve model performance and when they introduce little or no benefit.

## Methodology

### 1. Data Preprocessing

The datasets were prepared using Python, pandas and scikit-learn.

The preprocessing involved:

- Cleaning and organising crime data.
- One-hot encoding categorical variables such as borough names.
- Applying Box–Cox transformations to selected numerical features.
- Using chronological training and testing splits.

The models were trained on data from **2022–2024** and evaluated on **2025** data.

### 2. Regression Models

Two main regression approaches were compared:

**Poisson Regression**
- Used to model non-negative outcomes, particularly crime counts.
- Evaluated using Mean Squared Error and Mean Poisson Deviance.

**Linear Regression**
- Used as a baseline comparison.
- Evaluated using Mean Squared Error and R².

### 3. Box–Cox Transformation

The Box–Cox transformation was applied to selected numerical predictors to investigate whether changing their distribution improved regression performance.

The research also considered two approaches to combining transformations and regression:

- **Two-step transformation:** Estimate the Box–Cox parameter before fitting the regression model.
- **Joint estimation:** Optimise the transformation and regression parameters together.

The Python implementation uses scikit-learn preprocessing and regression pipelines to compare transformed and untransformed features. Full joint optimisation of the transformation and regression parameters is discussed in the research report but is not implemented as a separate custom optimisation method.

### 4. Model Evaluation

Model performance was compared using:

- **Mean Squared Error (MSE):** Measures prediction error.
- **Mean Poisson Deviance:** Evaluates Poisson regression predictions.
- **R² Score:** Measures the proportion of variation explained by linear regression.

Matplotlib was used to visualise transformations and model comparisons.

## Key Findings

### Crime Density

Applying Box–Cox transformations improved the Poisson regression results.

- Baseline Poisson MSE: approximately 65
- Box–Cox + Poisson MSE: approximately 40
- Approximate reduction in MSE: 38%

This suggests that transforming selected numerical predictors can improve model performance for crime density.

### Crime Counts

Box–Cox transformations did not produce a meaningful improvement when predicting crime counts.

One possible explanation is that transforming the month feature does not adequately capture the underlying patterns in discrete crime count data.

This highlighted the importance of choosing appropriate features and transformations rather than assuming that a transformation will always improve results.

### Distance to Police Stations

Applying Box–Cox transformations to distance-related features produced only minimal improvements.

This suggests that the usefulness of a transformation depends on the characteristics of the feature and the relationship being modelled.

### Overall Conclusion

The results demonstrate that **Box–Cox transformations are not universally beneficial**.

Their effectiveness depends on the target variable, the predictors and the regression approach.

The research reinforced the importance of evaluating statistical techniques empirically rather than assuming that additional preprocessing will automatically improve predictive performance.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib

## Repository Structure

```text
poisson-boxcox-crime-analysis/
├── count_BC_poisson_linear_reg.py
├── crime_BC_poisson_linear_reg.py
├── crime_count_with_distance.py
├── dist_BC_poisson_linear_reg.py
├── BoxCox_for_crime_analysis_in_London.pdf
├── RESOURCES.md
└── README.md
```

## Research Report

The full research document explains the motivation behind the project, the Box–Cox transformation approaches and the findings from the regression experiments.

**[Read the Research Report — Crime Analysis Using Box–Cox](./BoxCox_for_crime_analysis_in_London.pdf)**

## Running the Code

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
```

The scripts use CSV datasets containing crime counts, crime density and distance-related information.

These datasets are not currently included in the public repository, so the relevant input files must be provided before running the experiments.

## Project Status

Research project exploring statistical regression, feature transformations and model evaluation using London crime data.

