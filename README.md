# California Housing Price Prediction
![Housing](https://github.com/Onyango-S/California-Housing-Price-Prediction/blob/main/housing.jpg)

## Overview
This project involves building and evaluating a linear regression model to predict the median house price based on the features of the house. The goal was to develop a robust predictive model that can provide accurate and reliable house price predictions, ultimately improving decision-making in the real estate industry. This README file outlines the steps taken in the project, provides key insights, and includes visualizations and code snippets that showcase the work done.


## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Problem Solving Steps](#problem-solving-steps)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Selection](#feature-selection)
6. [Model Building](#model-building)
7. [Model Evaluation](#model-evaluation)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Insights and Conclusions](#insights-and-conclusions)


## Problem Statement
Accurately predicting house prices is a complex challenge due to the multifaceted nature of real estate data and the presence of issues like multicollinearity and noise. This project aims to develop a robust regression model capable of overcoming these challenges to provide more accurate and reliable house price predictions.

## Dataset
- **Source**: Kaggle's [California Housing Prices Data (5 new features!)](https://www.kaggle.com/datasets/fedesoriano/california-housing-prices-data-extra-features)
- **Description**: The dataset contains 20,640 rows and 14 columns, with features including median income, age, total rooms, total bedrooms, and geographical coordinates. The target variable is the median house price.
  
## Problem Solving Steps
The following steps were followed in building this project:
- Load the data into a dataframe
- Perform data preprocessing
- Perform feature selection
- Build models and select the best model that fits the data
- Tune the hyperparameters of the selected model
- Derive insights and conclusions

## Data Preprocessing
Data preprocessing was conducted to check whether assumptions of linearity, normality, and no multicollinearity were met. Here is the correlation matrix for the features in the data.

![Correlation Matrix](https://github.com/Onyango-S/California-Housing-Price-Prediction/blob/main/correlation%20matrix.png)

## Feature Selection
Removed columns showing high collinearity with other features and low correlation with the target variable. This is essential in improving the model's generalizability.

```python
# Drop columns that constitute multicollinearity
df.drop(columns=[col for col in collinear_cols if col not in ['Tot_Rooms', 'Latitude', 'Longitude']], inplace=True)
```
## Model Building
Trained multiple regression models
```python
# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'xgb': XGBRegressor(),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor()
}

# Train and evaluate models
model_performance = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_performance[name] = {'MSE': mse, 'R2': r2}
```

## Model Evaluation
![model performance](https://github.com/Onyango-S/California-Housing-Price-Prediction/blob/main/model%20performance.png)

The Linear and Ridge Regression models had the worst performance while Random Forest and XGBoost were the best performing models, both achieving the lowest MSE of 0.05 and the highest R² scores.
The XGBoost model was selected because it had a slightly better performance than the Random Forest model.
## Hyperparameter Tuning 
```python
#instantiate the regressor
xgb = XGBRegressor()

param_grid = [{'n_estimators':[100,300,400], 'max_depth':[4,5,6],'learning_rate':[0.1,0.3,0.5],
              'colsample_bylevel':[0.7,1]}]
              
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True, 
                                   n_jobs=-1)
grid_search.fit(X_train, y_train)
```

## Perfomance of the Tuned model 
![retrained model performance](https://github.com/Onyango-S/California-Housing-Price-Prediction/blob/main/retrained%20model.png)

The XGBoost model had a Mean Squared Error of 0.049 and R-Squared of 0.85.

## Insights and Conclusions
![feature importance](https://github.com/Onyango-S/California-Housing-Price-Prediction/blob/main/feature%20importance.png)

Out of the initial 13 features, the analysis reveals that only 5 are significant in predicting housing prices in California: total rooms, age of the house, latitude, longitude, proximity to the coast, and median family income. Notably, median income and proximity to the coast emerged as the most influential factors, exerting a greater impact on determining median house prices compared to the other features.

![model vs actual](https://github.com/Onyango-S/California-Housing-Price-Prediction/blob/main/model%20vs%20actual.png)

The XGBoost model, in particular, proved to be the most effective in predicting housing prices, delivering a Mean Squared Error (MSE) of 0.049 and a Root Mean Squared Error (RMSE) of $47K. With an R² score of 0.85, the model accounts for 85% of the variance in housing prices, showcasing its strong predictive power. This high level of accuracy indicates that XGBoost not only reliably estimates housing prices but also provides meaningful insights into the underlying drivers of property values. The model’s ability to minimize prediction errors and closely mirror actual market prices highlights its effectiveness in capturing the complexities of the California housing market.
