# %% [markdown]
# # Salary Analysis A01423983
# 
# Predict the salary of a person based on their years of experience. The method to be used is K-Nearest Neighbors Regressor.
# 
# The dataset is available at [Salary Data](https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer)
# 

# %%
# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# %%
# Load the data

df = pd.read_csv('Salary Data.csv')
df.head()

# %%
# Check for null values

df.isnull().sum()

# %%
# Remove the null values

df.dropna(inplace=True)


# %%
# Transform gender to binary

def gender_to_numeric(gender):
    if gender == "Male":
        return 0
    if gender == "Female":
        return 1
    return -1

df['Gender'] = df['Gender'].apply(lambda x: gender_to_numeric(x))
df['Gender'] = df['Gender'].astype(bool)
df.head()

# %%
# Check for unique values in education level

df['Education Level'].unique()

# %%
# Check for unique values in job title

df['Job Title'].unique()

# %% [markdown]
# Quitamos la columna de $job$ $title$ porque hacer un analisis de texto es un poco mas complicado y hacer one hot encoding de esta columna nos generaria demasiadas columnas.

# %%
# Drop the job title column

df.drop('Job Title', axis=1, inplace=True)
df.head()

# %%
# One hot encode the education level

df = pd.get_dummies(df, columns=['Education Level'])
df.head()


# %%
# K-Nearest Neighbors

# Split the data into training, testing, and validation sets

X = df.drop('Salary', axis=1)
y = df['Salary']

# %%
# Split the data into training, testing, and validation sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
#X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

# %%
# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#X_val = scaler.transform(X_val)

# %% [markdown]
# ## Maximizaremos el valor de $k$ usando cross validation. Esto lo hacemos para evitar overfitting y underfitting. Conforme avanza el valor de $k$, el modelo se vuelve mas preciso, evitando el underfitting. Sin embargo, si el valor de $k$ es muy grande, el modelo se vuelve muy especifico y puede caer en overfitting. Es por eso que la metrica de cross validation nos ayuda a encontrar el valor de $k$ que nos da el mejor balance entre overfitting y underfitting. En la siguiente grafica se presenta una linea que muestra como la precision sube y despues baja. Finalmente esocgemos el valor de $k$ que mejores resultados da al compararse con un set de prueba.

# %%
# Maximize k value using cross validation

k_values = [i for i in range (1,100)]
scores = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=5)    
    scores.append(np.mean(score))


# %%
best_index = np.argmax(scores)
best_k = k_values[best_index]

knn = KNeighborsRegressor(n_neighbors=best_k)
knn.fit(X_train, y_train)


# %%
# Plot the predictions vs the actual values in validation set

y_pred = knn.predict(X_test)

plt.scatter(range(0, len(y_pred)), y_pred)
plt.scatter(range(0, len(X_test)), y_test)
plt.title('Predictions vs Actual Values')
plt.legend(['Predicted', 'Real'])
plt.show()

# %%

print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred), '\n')

# %%
# Show in console the predictions vs the actual values in validation set

print('Tabla con valores de validation set:')

from prettytable import PrettyTable
t = PrettyTable(['Predicted value', 'Real value', 'Difference'])

for i in range(0, len(y_pred)):
    t.add_row([round(y_pred[i], 2), y_test.values[i], round(y_pred[i] - y_test.values[i], 2)])
    
print(t)