# Метод опорных векторов (SCM - support vector machine) - классификация и регрессия
# выбирается линия по принципу с максимальным отступом
# Этот метод, в отличие от прошлой пары - разделяющая классификация (рисует линию/кривую, которая разделяет классы данных)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.svm import SVC

iris = sns.load_dataset('iris')
'''print(iris.head())

data = iris[['sepal_length', 'petal_length', 'species']]
data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'petal_length']]
y = data_df['species']

data_df_seposa = data_df[data_df['species'] == 'setosa']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']


plt.scatter(data_df_seposa['sepal_length'], data_df_seposa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

model = SVC(kernel='linear', C=10000)
model.fit(X,y)

print(model.support_vectors_)

plt.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=400, facecolor="none", edgecolors='black')
# точки - опорные векторы, система их выбрала, они помогают строить линию(решение)

x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
)
y_p = model.predict(X_p)
X_p["spices"] = y_p

X_p_setosa = X_p[X_p['species'] == 'setosa']
X_p_versicolor = X_p[X_p['species'] == 'versicolor']

plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha=0.2)
plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.2)
'''
# ДЗ. убрать из данных iris часть точек (на которых мы обучаемся) и убедиться, что на предсказание влияют только опорные вектора.
# В случае, если данные перекываются, то идеальной границы не существует. У модели существует гиперпараметр, который определяет "размытие" отступа, он разрешает некоторым точкам заходить в область отступа



data = iris[['sepal_length', 'petal_length', 'species']]
data_df = data[(data['species'] == 'virginica') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'petal_length']]
y = data_df['species']

data_df_virginica = data_df[data_df['species'] == 'virginica']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']


c_value = [[10000,1000,100,10], [1, 0.1, 0.01, 0.001]]
fig, ax = plt.subplots(2,4, sharex='col', sharey='row')
for i in range(2):
    for j in range(4):
        

        ax[i,j].scatter(data_df_virginica['sepal_length'], data_df_virginica['petal_length'])
        ax[i,j].scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])





        # Если С большое, то отступ задается "жестко", нет размытия. Чем меньше С, тем отступ становится более размытым.
 
        model = SVC(kernel='linear', C=c_value[i][j])
        model.fit(X,y)

        #print(model.support_vectors_)

        ax[i,j].scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=400, facecolor="none", edgecolors='black')


        x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
        x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

        X1_p, X2_p = np.meshgrid(x1_p, x2_p)

        X_p = pd.DataFrame(
            np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
        )


        y_p = model.predict(X_p)

        X_p["species"] = y_p

        X_p_virginica = X_p[X_p['species'] == 'virginica']
        X_p_versicolor = X_p[X_p['species'] == 'versicolor']

        ax[i,j].scatter(X_p_virginica['sepal_length'], X_p_virginica['petal_length'], alpha=0.1)
        ax[i,j].scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.1)


plt.show()




# Достоинства
# - Зависимость от небольшого числа опорных векторов => компактность модели
# - После обучения предсказания проходит очень быстро
# - На работу методов влияют только точки, находящиеся возле отступов, поэтому метдоы подходят для многомерных данных

# Недостатки
# - При большом количестве обучающих образцов могут быть значительные вычислительные затраты
# - Большая зависимость от С - размытости. Поиск параметра может привести к большим вычислительным затратам.
# - У результатов отсутствует вероятностная интерпретация (точки могут не участвовать в принятии решения)

