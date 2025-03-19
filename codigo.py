# codigo waterloo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

hpc = pd.read_csv('house_prices_canada.csv', encoding='iso-8859-1')
hpc.head(5)

hpc.isna().sum()

# data preparation

hpc['Province'].value_counts().sort_index()

hpc = hpc.drop(hpc[hpc['Province'] != 'Ontario'].index)
hpc

hpc['City'].value_counts().sort_index()

hpc = hpc.drop(hpc[hpc['City'] == 'Nanaimo'].index)
hpc = hpc.drop(hpc[hpc['City'] == 'Regina'].index)
hpc = hpc.drop(hpc[hpc['City'] == 'Saskatoon'].index)
hpc = hpc.drop(hpc[hpc['City'] == 'Winnipeg'].index)

indicator_city = pd.get_dummies(hpc['City'], prefix='City')
indicator_city

hpc2 = hpc.drop(['City', 'Address', 'Province', 'Population', 'Latitude', 'Longitude', 'Median_Family_Income'], axis=1)
hpc2 = hpc2.join(indicator_city)
hpc2

# removing outliers: number of beds and price (number of outliers is somewhat irrelevant 
# as it has high correlation with number of beds)

# beds outliers
x = hpc2["Number_Beds"]
mean = x.mean()
std = x.std()
hpc2 = hpc2[(hpc2["Number_Beds"] > mean - 25 * std) & (hpc2["Number_Beds"] < mean + 25 * std)]

# price outliers
x = hpc2["Price"]
mean = x.mean()
std = x.std()
hpc2 = hpc2[(hpc2["Price"] > mean - 40 * std) & (hpc2["Price"] < mean + 40 * std)]
hpc2

min_max = MinMaxScaler()
hpc_scaled = min_max.fit_transform(hpc2)
hpc_scaled = pd.DataFrame(hpc_scaled, index=hpc2.index, columns=hpc2.columns)
hpc_scaled

sns.pairplot(hpc_scaled[['Price', 'Number_Beds', 'Number_Baths']])

sns.pairplot(hpc_scaled[['Price', 'Number_Beds', 'Number_Baths']])

from sklearn.preprocessing import PolynomialFeatures

deg = 2
poly_features = PolynomialFeatures(degree=deg, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

train_predictions = poly_model.predict(X_train_poly)
test_predictions = poly_model.predict(X_test_poly)

train_mae = mean_absolute_error(y_train, train_predictions)
train_mse = mean_squared_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print("Train MAE:", train_mae)
print("Train MSE:", train_mse)
print("Test MAE:", test_mae)
print("Test MSE:", test_mse)

# Linear regresion

X = hpc_scaled.drop(['Price'], axis=1)
y = hpc_scaled['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156)

model = LinearRegression()
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_mae = mean_absolute_error(y_train, train_predictions)
train_mse = mean_squared_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print("Train MAE:", train_mae)
print("Train MSE:", train_mse)
print("Test MAE:", test_mae)
print("Test MSE:", test_mse)

# cross validation

from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, X, y, cv=10, scoring='neg_mean_squared_error')

result = -cv_results["test_score"].mean()
result

plt.hist(-cv_results["test_score"])



# loocv

# It takes 3 minutes to perform this task
cv_results = cross_validate(model, X, y, cv=X.shape[0], scoring="neg_mean_squared_error")
-cv_results["test_score"].mean()



# boostrap

from sklearn.utils import resample

bootstrap_estimates = []
n_iterations = 1_000
n_size = int(len(X) * 0.7)

for i in range(n_iterations):
    # prepare the train and test values
    X_sample, y_sample = resample(X_train, y_train, n_samples=n_size)

    # fit model
    model = LinearRegression()
    model.fit(X_sample, y_sample)

    # We evaluate the model
    predictions = model.predict(X_test)
    score = mean_squared_error(y_test, predictions)
    bootstrap_estimates.append(score)

# Calculate 95% confidence interval
alpha = 0.95
lower_p = ((1.0 - alpha) / 2.0) * 100
upper_p = (alpha + ((1.0 - alpha) / 2.0)) * 100
lower = np.percentile(bootstrap_estimates, lower_p)
upper = np.percentile(bootstrap_estimates, upper_p)

print(f'{alpha * 100} confidence interval {lower * 100} and {upper * 100}')

# codigo para pablo

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Update with your actual file path
df = pd.read_csv('data/housing-affordability-canada/housing-supply-and-rental/housing-supply-and-rental/abbotsford_section1_.csv')

X = df.drop(columns=["Attrition"])  # Todas las variables predictoras
y = df["Attrition"]  # Variable objetivo (1 = abandono, 0 = sigue en la empresa)

# Dividir los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156, stratify=y)

# Modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Modelo Árbol de Decisión
tree = DecisionTreeClassifier(max_depth=5, random_state=156)
tree.fit(X_train, y_train)

# Modelo SVM
svm = SVC()
svm.fit(X_train, y_train)

# Predicciones
y_pred_knn = knn.predict(X_test)
y_pred_tree = tree.predict(X_test)
y_pred_svm = svm.predict(X_test)

# Evaluación
print(f"KNN Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_knn):.4f}")
print(f"Árbol de Decisión Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_tree):.4f}")
print(f"SVM Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_svm):.4f}")

# Matriz de Confusión
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - KNN")
plt.show()

# Cross-validation
cv_knn = cross_val_score(knn, X, y, cv=10, scoring="balanced_accuracy")
cv_tree = cross_val_score(tree, X, y, cv=10, scoring="balanced_accuracy")
cv_svm = cross_val_score(svm, X, y, cv=10, scoring="balanced_accuracy")

print(f"KNN Cross-Validation Score: {cv_knn.mean():.4f}")
print(f"Árbol de Decisión Cross-Validation Score: {cv_tree.mean():.4f}")
print(f"SVM Cross-Validation Score: {cv_svm.mean():.4f}")
