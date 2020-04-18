"""
Data Set: Iris
Classification Algorithm: Neural Network
Structure: Input Layer (4 Neurons)
           Hidden Layer (5 Neurons)
           Output Layer (3 Neurons)
Auth: RedDragon
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\Akshat\\Desktop\\Akshat\\Python Projects\\Data\\Iris.csv')
print(data.describe())
sns.pairplot(data=data, vars=('sepal.length', 'sepal.width', 'petal.length', 'petal.width'), hue='variety')

df_norm = data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()))
df_norm.sample(n=5)

target = data[['variety']].replace(['Setosa', 'Versicolor', 'Virginica'], [0, 1, 2])

df = pd.concat([df_norm, target], axis=1)

train_test_per = 90 / 100.0
df['train'] = np.random.rand(len(df)) < train_test_per

train = df[df.train == 1]
train = train.drop('train', axis=1).sample(frac=1)

test = df[df.train == 0]
test = test.drop('train', axis=1)

X = train.values[:, :4]

targets = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
y = np.array([targets[int(x)] for x in train.values[:, 4:5]])

num_inputs = len(X[0])
hidden_layer_neurons = 5
np.random.seed(4)
w1 = 2 * np.random.random((num_inputs, hidden_layer_neurons)) - 1

num_outputs = len(y[0])
w2 = 2 * np.random.random((hidden_layer_neurons, num_outputs)) - 1

_x = np.linspace(-5, 5, 50)
_y = 1 / (1 + np.exp(-_x))
plt.plot(_x, _y)
plt.show()

learning_rate = 0.03
for epoch in range(50000):
    l1 = 1 / (1 + np.exp(-(np.dot(X, w1))))
    l2 = 1 / (1 + np.exp(-(np.dot(l1, w2))))
    er = (abs(y - l2)).mean()
    l2_delta = (y - l2) * (l2 * (1 - l2))
    l1_delta = l2_delta.dot(w2.T) * (l1 * (1 - l1))
    w2 += l1.T.dot(l2_delta) * learning_rate
    w1 += X.T.dot(l1_delta) * learning_rate
print('Error:', er)

X = test.values[:, :4]
y = np.array([targets[int(x)] for x in test.values[:, 4:5]])

l1 = 1 / (1 + np.exp(-(np.dot(X, w1))))
l2 = 1 / (1 + np.exp(-(np.dot(l1, w2))))

print(np.round(l2, 3))

yp = np.argmax(l2, axis=1)
res = yp == np.argmax(y, axis=1)
correct = np.sum(res) / len(res)

testres = test[['variety']].replace([0, 1, 2], ['Setosa', 'Versicolor', 'Virginica'])

testres['Prediction'] = yp
testres['Prediction'] = testres['Prediction'].replace([0, 1, 2], ['Setosa', 'Versicolor', 'Virginica'])

print(testres)
print('Correct:', sum(res), '/', len(res), ':', (correct * 100), '%')
