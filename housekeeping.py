from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np 
X, y= load_diabetes(return_X_y=True)

print(X.shape, y.shape)

# df = pd.read_csv('diabetes.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

np.savetxt('train.txt', X_train)

np.savetxt('test.txt', X_test)
