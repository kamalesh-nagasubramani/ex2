OBJECTIVE:

To implement and optimize linear regression and logistic regression models using real-world datasets and analyze their performance.


SCENARIO 1:

Predict ocean water temperature using environmental and depth-related features.

Dataset (Kaggle – Public): https://www.kaggle.com/datasets/sohier/calcofi

Target Variable:

· Water Temperature (T_degC)

Sample Input Features

· Depth (m)

· Salinity

· Oxygen

· Latitude

· Longitude


IN-LAB TASKS

• Import necessary Python libraries (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn).

• Load the CalCOFI dataset into a Pandas DataFrame

• Select relevant numerical features and target variable.

• Handle missing values using mean/median imputation.

• Perform feature scaling using StandardScaler.

• Split the dataset into training and testing sets.

• Train a Linear Regression model using Scikit-learn.

• Predict water temperature for test data.

• Evaluate model performance using: – Mean Squared Error (MSE) – Root Mean Squared Error (RMSE) – R² Score

• Visualize: – Actual vs Predicted temperature – Residual errors

• Optimize model performance using: – Feature selection – Regularization (Ridge / Lasso)


SCENARIO 2: Classify whether LIC stock price will increase (1) or decrease (0) based on historical data.

Dataset (Kaggle – Public): https://www.kaggle.com/datasets/debashis74017/lic-stock-price-data

Target Variable (Derived): • Price Movement – 1 → Closing price > Opening price – 0 → Closing price ≤ Opening price

Input Features: • Open • High • Low • Volume


• Import required Python libraries.

• Load LIC stock dataset into Pandas.

• Create a binary target variable (Price Movement).

• Handle missing values.

• Perform feature scaling.

• Split the dataset into training and testing sets.

• Train a Logistic Regression model.

• Predict stock movement for test data.

• Evaluate classification performance using: – Accuracy – Precision – Recall – F1-Score – Confusion Matrix

• Visualize: – ROC Curve – Feature importance

• Optimize model using: – Hyperparameter tuning (C, penalty) – Regularization
