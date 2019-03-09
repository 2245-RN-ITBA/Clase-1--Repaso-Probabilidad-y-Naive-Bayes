from sklearn import datasets
iris = datasets.load_iris()


import numpy as np
train_data = np.zeros([120,4])
train_data[0:40] = iris.data[0:40]
train_data[40:80] = iris.data[50:90]
train_data[80:120]=iris.data[100:140]

test_data = np.zeros([30,4])
test_data[0:10] = iris.data[40:50]
test_data[10:20] = iris.data[90:100]
test_data[20:30] = iris.data[140:150]

train_target = np.zeros(120)
train_target[0:40] = iris.target[0:40]
train_target[40:80] = iris.target[50:90]
train_target[80:120]=iris.target[100:140]

test_target = np.zeros(30)
test_target[0:10] = iris.target[40:50]
test_target[10:20] = iris.target[90:100]
test_target[20:30] = iris.target[140:150]

np.save('train_data.npy',train_data)
np.save('test_data.npy',test_data)
np.save('train_target.npy',train_target)
np.save('test_target.npy',test_target)


