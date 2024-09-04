import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

raw_data = pd.read_excel('dataset1.xlsx')

print("Summary of the dataset (data types, number of non-null values):\n")
print(raw_data.info(), "\n")

print("Summary of the statistics of the dataset:\n")
print(raw_data.describe(), "\n")

print("Number of missing values per column in the dataset:\n")
print(raw_data.isnull().sum())

data_dropped = raw_data.copy()
data_dropped.dropna()
print("Number of missing values per column after dropping all null values:\n")
print(data_dropped.isnull().sum(), "\n")
print("Summary of the statistics of the dataset after dropping all null values:\n")
print(data_dropped.describe())

data_replace_mean = raw_data.copy()

mean_var4 = raw_data["var4"].mean()
print(f"The mean of the column 'Var 4' is: {mean_var4}\n")

data_replace_mean["var4"] = raw_data["var4"].fillna(mean_var4)

print("Summary of the statistics of the dataset after replacing with the mean:\n")
print(data_replace_mean.describe())

data_knn = raw_data.copy()
columns = ["var1", "var2", "var4"]
df_impute = data_knn[columns]
imputer = KNNImputer(n_neighbors=2)
df_imputed = imputer.fit_transform(df_impute)
data_knn["var4"] = df_imputed[:, 2]

print("Summary of the statistics of the dataset after replacing with the KNN:\n")
print(data_knn.describe())

# Convert the values of var3 to unique integers to represent each country
data_encoded = data_replace_mean
data_encoded["var3"], _ = pd.factorize(data_encoded["var3"])

# Convert the values in var6 to either 1 or 0 representative.
map_dict = {"yes": 1, "no": 0}
data_encoded["var6"] = data_encoded["var6"].map(map_dict)

# Convert any incorrect datetime values to NaT
data_encoded["var7"] = pd.to_datetime(data_encoded["var7"], errors='coerce', dayfirst=True)

# Replace the NaT values with the median datetime value
data_encoded["var7"] = data_encoded["var7"].fillna(data_encoded["var7"].median())

# Convert the values in var7 to unique columns to ensure proper training in the classifier model
data_encoded['year'] = data_encoded['var7'].dt.year
data_encoded['month'] = data_encoded['var7'].dt.month
data_encoded['day'] = data_encoded['var7'].dt.day
data_encoded['hour'] = data_encoded['var7'].dt.hour
data_encoded['minute'] = data_encoded['var7'].dt.minute
data_encoded['day_of_week'] = data_encoded['var7'].dt.dayofweek
data_encoded = data_encoded.drop("var7", axis=1)

print("The first 5 rows of the dataset after all preprocessing is complete:\n")
print(data_encoded.head(), "\n")

print("The last 5 rows of the dataset after all preprocessing is complete:\n")
print(data_encoded.tail(), "\n")


############## Task 2 - Creating classifier

# Remove the target column from the data
X = data_encoded.drop("target", axis=1)
y = data_encoded["target"]

# Split the data into the variables required for the classifiers.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.2f}')

cm_log_reg = confusion_matrix(y_test, y_pred)
disp_log_reg = ConfusionMatrixDisplay(confusion_matrix=cm_log_reg)
disp_log_reg.plot()
plt.title('Logistic Regression Confusion Matrix')
plt.show()

from sklearn.tree import DecisionTreeClassifier

# Initialize and train the model
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Predict and evaluate
y_pred_tree = decision_tree.predict(X_test)
print(f'Decision Tree Accuracy: {accuracy_score(y_test, y_pred_tree):.2f}')

# Confusion Matrix
cm_tree = confusion_matrix(y_test, y_pred_tree)
disp_tree = ConfusionMatrixDisplay(confusion_matrix=cm_tree)
disp_tree.plot()
plt.title('Decision Tree Confusion Matrix')
plt.show()

from sklearn.ensemble import RandomForestClassifier

# Initialize and train the model
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = random_forest.predict(X_test)
print(f'Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}')

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp_rf.plot()
plt.title('Random Forest Confusion Matrix')
plt.show()