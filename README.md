# 🏠 House Prices - Advanced Regression Techniques

This project is a machine learning solution for the Kaggle competition **House Prices - Advanced Regression Techniques**.

The goal of the competition is to predict the final sale price of residential homes in Ames, Iowa, using different house-related features such as area, quality, location, garage information, basement information, year built, and sale condition.

This is a **regression problem**, because the target value is a continuous number: `SalePrice`.

---

## 📌 Project Overview

In this project, I built a complete machine learning pipeline for predicting house prices. The project includes:

- Exploratory Data Analysis
- Missing value handling
- Feature engineering
- Categorical feature encoding
- Skewed feature transformation
- Cross validation
- Model comparison
- Final ensemble/blending model
- Kaggle submission file creation

The final prediction file is created as:

```text
submission_final_with_cv_plots.csv
```

---

## 📊 Dataset

The competition provides three main files:

| File | Description |
|---|---|
| `train.csv` | Training data with house features and the target column `SalePrice` |
| `test.csv` | Test data without `SalePrice`; this is used for final prediction |
| `sample_submission.csv` | Sample format required for Kaggle submission |

The training dataset contains 1460 rows and 81 columns.  
The test dataset contains 1459 rows and 80 columns.

The target column is:

```text
SalePrice
```

---

## 🎯 Problem Type

This is a **supervised regression** problem.

The model learns from the training data:

```text
House Features → SalePrice
```

Then it predicts house prices for the test dataset.

---

## 📏 Evaluation Metric

Kaggle evaluates submissions using **Root Mean Squared Logarithmic Error (RMSLE)**.

Because of this, the target variable `SalePrice` is transformed using:

```python
y = np.log1p(train["SalePrice"])
```

After prediction, the values are converted back using:

```python
predictions = np.expm1(predictions)
```

This helps the model handle expensive houses and cheaper houses more fairly.

---

## 🔍 Exploratory Data Analysis

The notebook includes several EDA steps:

- Dataset shape analysis
- Missing value analysis
- SalePrice distribution
- Correlation heatmap
- Important numerical feature analysis
- Neighborhood price analysis
- Quality-based price analysis

Some of the most important features in the dataset are:

- `OverallQual`
- `GrLivArea`
- `TotalBsmtSF`
- `GarageCars`
- `GarageArea`
- `YearBuilt`
- `YearRemodAdd`
- `Neighborhood`
- `KitchenQual`
- `ExterQual`

---

## 🧹 Data Cleaning

Missing values were handled carefully because many missing values in this dataset have meaning.

For example:

- Missing garage values usually mean the house has no garage.
- Missing basement values usually mean the house has no basement.
- Missing pool values usually mean the house has no pool.
- Missing fireplace quality usually means the house has no fireplace.

So these missing values were not filled blindly with mean or mode.

Examples:

```python
GarageType = "None"
BsmtQual = "None"
PoolQC = "None"
FireplaceQu = "None"
```

For numerical columns related to missing facilities, missing values were filled with `0`.

Examples:

```python
GarageArea = 0
GarageCars = 0
TotalBsmtSF = 0
PoolArea = 0
```

---

## 🛠️ Feature Engineering

Several new features were created to improve model performance.

Important engineered features include:

```python
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
```

This represents the total usable area of the house.

```python
TotalBath = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath
```

This represents the total number of bathrooms.

```python
HouseAge = YrSold - YearBuilt
```

This represents the age of the house at the time of sale.

```python
RemodAge = YrSold - YearRemodAdd
```

This represents how many years passed since the last remodel.

Other created features:

- `GarageAge`
- `IsRemodeled`
- `IsNew`
- `HasGarage`
- `HasBasement`
- `HasFireplace`
- `HasPool`
- `Has2ndFloor`
- `HasWoodDeck`
- `HasOpenPorch`
- `TotalPorchSF`
- `TotalQual`
- `QualSF`
- `QualBath`
- `QualGarage`

These features help the model understand the house structure better than using raw columns only.

---

## 🔡 Encoding Categorical Features

The dataset contains many categorical features.

Some features are ordinal, meaning they have a clear order. For example:

```text
Ex > Gd > TA > Fa > Po
```

These features were converted using ordinal mapping.

Example:

```python
quality_map = {
    "None": 0,
    "Po": 1,
    "Fa": 2,
    "TA": 3,
    "Gd": 4,
    "Ex": 5
}
```

Other categorical features without a natural order were converted using one-hot encoding.

---

## 🚧 Outlier Handling

Some outliers were removed from the training data.

The most important outliers were houses with very large living area but unusually low sale price.

These outliers can hurt model performance, so they were removed before training.

Example logic:

```python
train = train.drop(
    train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)].index
)
```

---

## 🤖 Models Used

Several models were trained and compared:

- Lasso Regression
- ElasticNet Regression
- Ridge Regression
- Kernel Ridge Regression
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

The final solution uses a conservative blending approach.

Instead of trusting only one model, predictions from multiple models are combined.

This usually gives more stable results in Kaggle competitions.

---

## 🧠 Final Model Strategy

The final prediction is created using a weighted blend of several models.

The idea is simple:

```text
Final Prediction = weighted average of multiple model predictions
```

Models such as Lasso, ElasticNet, Ridge, Kernel Ridge, Gradient Boosting, XGBoost, and LightGBM make their own predictions.

Then their outputs are combined to create the final submission.

This reduces the risk of relying too much on one model.

---

## ✅ Cross Validation

Cross validation was used to evaluate model performance before creating the final submission.

The notebook includes:

- K-Fold cross validation
- RMSE calculation on log-transformed SalePrice
- Model comparison results
- A plot showing model performance

Cross validation helps estimate how well the model may perform on unseen data.

---

## 🏆 Kaggle Result

The improved model achieved a better Kaggle score compared to the earlier version.

Previous score:

```text
0.13574
```

Improved score:

```text
0.12385
```

This improvement came mainly from:

- Better missing value handling
- Better feature engineering
- Log transformation of SalePrice
- Outlier removal
- Model blending

---

## 📁 Project Files

| File | Description |
|---|---|
| `house_price_final_with_cv_plots.ipynb` | Final notebook with EDA, plots, cross validation, modeling, and submission creation |
| `house_price_final_with_cv_plots.py` | Python script version of the final notebook |
| `submission_final_with_cv_plots.csv` | Final Kaggle submission file |
| `README.md` | Project documentation |

---

## 🚀 How to Run

### Run on Kaggle

1. Open the notebook on Kaggle.
2. Add the competition dataset:

```text
House Prices - Advanced Regression Techniques
```

3. Run all cells.
4. The final submission file will be created automatically.
5. Submit the generated CSV file to Kaggle.

### Run Locally

Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost lightgbm
```

Place these files in the correct folder:

```text
train.csv
test.csv
sample_submission.csv
```

Run the Python file:

```bash
python house_price_final_with_cv_plots.py
```

The output file will be:

```text
submission_final_with_cv_plots.csv
```

---

## 📦 Required Libraries

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
xgboost
lightgbm
```

---

## 💡 Main Lessons Learned

This project shows that improving a machine learning model is not only about choosing a stronger algorithm.

The most important parts were:

- Understanding the dataset
- Handling missing values correctly
- Creating useful new features
- Transforming the target variable
- Removing harmful outliers
- Comparing models with cross validation
- Combining models with blending

In this competition, feature engineering and preprocessing had a major effect on the final score.

---

## ✅ Conclusion

This project successfully builds a complete regression pipeline for predicting house prices.

The final model uses a conservative blend of multiple regression models to produce stable predictions. Compared with the first version, the score improved from `0.13574` to `0.12385`.

The project demonstrates that careful data cleaning, meaningful feature engineering, and model blending can significantly improve prediction quality in a regression competition.

