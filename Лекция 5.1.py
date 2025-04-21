# Деревья решений и случайны леса
# Случ леса - непараметрический алгоритм (тк в формуле модели-обучающей функции -непосредственно данные не заложены. В параметрических фиксируем формулу заранее, как лин регрессия, потом подбир параметры, а тут мы не знаем, какие параметры возьмем, все зависит от данных)
# СЛ - пример ансамблевого метода, основанного на агрегации результатов множества простых моделей.
# основан на дереве принятия решений
# В реализациях дерева принятия решений в машинном обучении, вопросы обычно ведут к разделению данных по осям, т.е. каждый узел разбивает данные на 2 группы по одному из признаков.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

iris = sns.load_dataset('iris')

species_int = []
for r in iris.values:
    match r[4]:
        case "setosa":
            species_int.append(1)
        case "versicolor":
            species_int.append(2)
        case "virginica":
            species_int.append(3)
            
species_int_df = pd.DataFrame(species_int)
print(species_int_df.head())

data = iris[['sepal_length', 'petal_length']]
data["species"] = species_int_df

print(data)

data_df = data[(data['species'] == 3) | (data['species'] == 2)]

X = data_df[['sepal_length', 'petal_length']]
y = data_df['species']

data_df_virginica = data_df[data_df['species'] == 3]
data_df_versicolor = data_df[data_df['species'] == 2]


max_depth = [[1,2,3,4], [5,6,7,8]]
fig, ax = plt.subplots(2,4, sharex='col', sharey='row')

for i in range(2):
    for j in range(4):
        ax[i,j].scatter(data_df_virginica['sepal_length'], data_df_virginica['petal_length'])
        ax[i,j].scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])
        model = DecisionTreeClassifier(max_depth=max_depth[i][j])
        model.fit(X,y)

        
        x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
        x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

        X1_p, X2_p = np.meshgrid(x1_p, x2_p)

        X_p = pd.DataFrame(
            np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length']
        )
        y_p = model.predict(X_p)

        ax[i,j].contourf(X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.4, levels=2, cmap='rainbow', zorder = 1)
plt.show()

'''plt.scatter(data_df_seposa['sepal_length'], data_df_seposa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])


model = DecisionTreeClassifier(max_depth=5)
model.fit(X,y)
'''


#plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha=0.2)
#plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha=0.2)










'''
plt.scatter(data_df_seposa['sepal_length'], data_df_seposa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

'''





plt.show()
