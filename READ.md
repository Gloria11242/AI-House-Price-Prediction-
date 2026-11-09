# AI-Powered House Price Prediction System

## 1. Project Overview

The AI-Powered House Price Prediction System is a machine learning project designed to estimate residential property prices based on selected house characteristics.

The project uses the Ames Housing Dataset to train a Random Forest Regression model. The trained model is integrated into an interactive Streamlit application, allowing users to enter property details and receive estimated house prices.

**Project Objective:** To develop an end-to-end machine learning application that uses historical housing data to predict house selling prices.

## 2. Dataset Description

The project uses the Ames Housing Dataset.

- Total records: 1,460
- Original columns: 81
- Target variable: `SalePrice`
- Selected input features: 12

The selected features are:

1. OverallQual
2. YearBuilt
3. GrLivArea
4. 1stFlrSF
5. TotalBsmtSF
6. 2ndFlrSF
7. FullBath
8. TotRmsAbvGrd
9. GarageCars
10. GarageArea
11. LotArea
12. YearRemodAdd

The target variable, `SalePrice`, represents the actual selling price of a house.

## 3. Machine Learning Concepts Explained

### 3.1 What is Regression?

Regression is a supervised machine learning technique used to predict continuous numerical values.

Unlike classification, which predicts categories such as approved or rejected, regression predicts numerical outcomes.

In this project, regression is used to estimate the selling price of a house based on its characteristics.

**Example:**

A house has:
- Living area of 2,000 square feet
- Overall quality of 8
- Four bedrooms

The model uses these characteristics to estimate the house's selling price.

House price prediction is therefore a regression problem because the output is a numerical value.

### 3.2 What is Supervised Machine Learning?

Supervised machine learning is a technique where a model learns from historical data containing both input variables and known outcomes.

The model studies the relationship between the input variables and the correct answers during training.

In this project:

- Input variables (X): House characteristics such as living area, construction year and overall quality.
- Target variable (y): `SalePrice`, the actual selling price.

The model learns from previously sold houses and uses the patterns it discovers to predict prices for new houses.

### 3.3 What is Random Forest?

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to make predictions.

Each decision tree learns patterns from the training data. For regression, the individual tree predictions are combined, typically by averaging them, to produce the final prediction.

**Why Random Forest was selected:**

- It can capture complex relationships between variables.
- It can model nonlinear patterns.
- Combining multiple trees can improve prediction stability compared with relying on a single decision tree.
- It provides feature importance scores that help identify influential variables.

In this project, Random Forest Regression is used to estimate house prices.

### 3.4 What is Feature Selection?

Feature selection is the process of choosing the most relevant input variables for a machine learning model.

A dataset may contain many variables, but not all of them are necessary for a particular prediction task.

For this project, 12 numerical features were selected from the original dataset.

These features describe important property characteristics, including:

- House quality
- Living area
- Basement size
- Construction year
- Garage capacity
- Land area

Feature selection helps simplify the model and ensures that the application only requires the selected variables when making predictions.

### 3.5 What is Data Preprocessing?

Data preprocessing involves preparing raw data before using it to train a machine learning model.

Real-world datasets may contain missing values, inconsistent formats or unsuitable variables.

In this project, preprocessing includes:

- Selecting the required numerical features.
- Separating input variables from the target variable.
- Handling missing numerical values using median imputation.
- Preparing the data for model training.

**Median imputation:** This replaces missing numerical values with the median of the relevant feature.

Preprocessing helps ensure that the model receives data in a suitable format.

### 3.6 What is Train-Test Splitting?

Train-test splitting is the process of dividing a dataset into two portions.

- Training data: Used to teach the machine learning model.
- Testing data: Used to evaluate how well the trained model performs on data it has not seen during training.

This project uses an 80:20 split:

- 80% for training.
- 20% for testing.

A `random_state` of 42 is used to make the split reproducible.

This separation helps us assess how well the model generalizes beyond its training data.

## 4. Model Evaluation Metrics

After training, the model is evaluated using three important regression metrics.

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | $18,670.58 |
| Root Mean Squared Error (RMSE) | $30,292.66 |
| R-squared (R²) | 0.8804 |

### 4.1 Mean Absolute Error (MAE)

MAE measures the average absolute difference between the actual house prices and the prices predicted by the model.

A lower MAE indicates that predictions are, on average, closer to the actual values.

For this project, the MAE is approximately $18,670.58.

This means the model's predictions differ from actual selling prices by an average absolute amount of approximately $18,670.58 on the test dataset.

### 4.2 Root Mean Squared Error (RMSE)

RMSE measures prediction errors by calculating the square root of the average squared differences between actual and predicted values.

It penalizes larger prediction errors more heavily than MAE.

For this project, the RMSE is approximately $30,292.66.

A lower RMSE generally indicates better predictive performance.

### 4.3 R-squared (R²)

R-squared measures the proportion of variation in the target variable that is explained by the model.

Its value is commonly between 0 and 1, although it can be negative on test data.

For this project, the R² score is 0.8804.

This indicates that the model explains approximately 88.04% of the variation in house prices in the test dataset.

**Important:** R² is not the same as prediction accuracy.

## 5. How Does Streamlit Communicate with the Trained Model?

Streamlit is a Python framework used to build interactive web applications.

In this project, Streamlit provides the interface through which users interact with the trained machine learning model.

The process works as follows:

1. The user opens the Streamlit application.
2. The user enters the characteristics of a house.
3. Streamlit collects the input values.
4. The input values are arranged into a Pandas DataFrame using the required feature names and order.
5. The saved machine learning model is loaded using Joblib.
6. The model receives the input through its `predict()` method.
7. The predicted house price is returned and displayed on the application.

The trained model is saved separately from the application so that it does not need to be retrained every time the application runs.

**Application workflow:**

User Input → Streamlit → Pandas DataFrame → Trained Random Forest Model → Price Prediction → Display Result

## 6. Model Development and Application Workflow

The project follows these stages:

1. Data collection and loading.
2. Exploratory data analysis.
3. Data preprocessing.
4. Feature selection.
5. Train-test splitting.
6. Random Forest model development.
7. Model evaluation.
8. Model serialization using Joblib.
9. Streamlit application development.
10. Interactive prediction and visualization.

## 7. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model evaluation |
| Joblib | Saving and loading the trained model |
| Streamlit | Interactive web application |
| Plotly | Interactive charts and visualizations |
| Jupyter Notebook | Data analysis and model development |

## 8. Project Limitations

- The model was trained using US housing data and may not generalize to the Ghanaian housing market.
- Predictions are estimates and should not be treated as guaranteed property valuations.
- Model performance depends on the quality and distribution of the training data.
- The model uses 12 selected numerical features rather than all available dataset variables.

## 9. Future Improvements

Possible improvements include:

- Testing additional regression algorithms.
- Hyperparameter tuning.
- Improving feature engineering.
- Comparing model performance.
- Incorporating additional housing characteristics.
- Exploring suitable Ghanaian housing datasets.
- Deploying the application publicly.

## 10. Conclusion

This project demonstrates an end-to-end machine learning workflow, from data preparation and model development to evaluation and application deployment.

By combining Python, Scikit-learn and Streamlit, the project transforms historical housing data into an interactive house price prediction system.

The project also demonstrates practical applications of supervised machine learning and regression in real estate analytics.