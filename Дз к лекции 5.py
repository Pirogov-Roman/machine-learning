import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.svm import SVC

iris = sns.load_dataset('iris')

data = iris[['sepal_length', 'petal_length', 'species']]
data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]

train_data = data_df.iloc[::2]  # берем каждую вторую точку
test_data = data_df.iloc[1::2]   # оставшиеся точки для проверки

X_train = train_data[['sepal_length', 'petal_length']]
y_train = train_data['species']
X_test = test_data[['sepal_length', 'petal_length']]
y_test = test_data['species']


model = SVC(kernel='linear', C=10000)
model.fit(X_train, y_train)

support_vectors = model.support_vectors_

plt.figure(figsize=(10, 6))

train_setosa = train_data[train_data['species'] == 'setosa']
train_versicolor = train_data[train_data['species'] == 'versicolor']
plt.scatter(train_setosa['sepal_length'], train_setosa['petal_length'], label='setosa (train)')
plt.scatter(train_versicolor['sepal_length'], train_versicolor['petal_length'], label='versicolor (train)')

test_setosa = test_data[test_data['species'] == 'setosa']
test_versicolor = test_data[test_data['species'] == 'versicolor']
plt.scatter(test_setosa['sepal_length'], test_setosa['petal_length'], marker='x', label='setosa (test)')
plt.scatter(test_versicolor['sepal_length'], test_versicolor['petal_length'], marker='x', label='versicolor (test)')

plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
            s=200, facecolor="none", edgecolors='black', label='Опорные вектора')

x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)
X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel()]).T, 
                  columns=['sepal_length', 'petal_length'])
y_p = model.predict(X_p)
X_p["species"] = y_p

X_p_setosa = X_p[X_p['species'] == 'setosa']
X_p_versicolor = X_p[X_p['species'] == 'versicolor']

plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha=0.1)
plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.1)

plt.legend()
plt.title("SVM с выделением опорных векторов")
plt.show()

print("\nПроверка влияния опорных векторов:")
print("Предсказание для первой тестовой точки:", model.predict([X_test.iloc[0]]))
print("Реальный класс:", y_test.iloc[0])
