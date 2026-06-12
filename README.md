# Ames Housing Dataset - House Price Prediction

This repository contains an Exploratory Data Analysis (EDA) and predictive modeling project focused on estimating residential house prices in Ames, Iowa, using advanced regression techniques.

## Project Structure
* `House Prices EDA.ipynb` – Jupyter Notebook containing the full exploratory analysis, visualizations, and insights.
* `train.csv` – The training dataset.
* `test.csv` – The test dataset.
* `data_description.txt` – Full documentation explaining each of the 79 housing features.

---

## Key EDA Insights & Discoveries

Through systematic univariate, bivariate, and multivariate analysis, the following structural patterns and data anomalies were identified:

### 1. Target & Continuous Feature Distributions
* **SalePrice & Spatial Metrics:** The target variable (`SalePrice`), `LotArea`, and `LotFrontage` exhibit heavy right-skewness, indicating a need for mathematical transformations (such as log transformation) to satisfy regression assumptions.
* **Zero-Inflation:** Several numerical features (e.g., `TotalBsmtSF`, `PoolArea`, `WoodDeckSF`) show massive spikes at exactly `0`, representing properties where those specific structural features are absent.

### 2. Data Anomalies & Collection Rules
* **The Remodeling Anchor:** Un-remodeled houses default to their original construction year in the `YearRemodAdd` column. This creates an artificial data peak in the 2000s due to a concurrent new-construction boom.
* **Temporal Truncation:** Market data for the year 2010 abruptly stops in July, artificially lowering the raw volume count for the latter half of that year. June consistently registers as the peak month for sales volume.
* **Leverage Outliers:** Scatter plots identified a few highly dangerous "renegade" data points—specifically, properties with massive ground living areas (`GrLivArea` > 4,000sqft) but anomalously low sale prices (< $300K).

### 3. Quality & Contract Dynamics
* **The Quality-Condition Paradox:** While higher `OverallQual` (materials/finishes) scales predictably with price, `OverallCond` (maintenance/age) behaves non-linearly. Highly maintained older properties often command lower prices than modern, larger homes in average condition.
* **Sale Conditions:** "Normal" sales consistently cluster between $100K and $200K. "Partial" sales (new constructions) shift the price floor upward significantly, while distressed sales (foreclosures, short sales) strictly cap out at lower thresholds.

### 4. Multicollinearity & Redundancy
High-correlation pairings were identified among several predictors, posing a structural risk for standard linear models:
* `TotRmsAbvGrd` and `GrLivArea` ($r = 0.83$)
* `GarageArea` and `GarageCars` ($r = 0.83$)
* `1stFlrSF` and `TotalBsmtSF` ($r = 0.82$)

---

## Next Steps
The insights gathered during this EDA phase serve as the official roadmap for the upcoming **Data Preprocessing and Feature Engineering** pipeline, which will focus on transforming skewed data, isolating leverage outliers, and addressing multicollinearity.
