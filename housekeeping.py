############################################################################################################
# SAVE TRAIN AND TEST DATA INTO .npy FILES
#############################################################################################################

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np 
X, y= load_diabetes(return_X_y=True)

print(X.shape, y.shape)

# df = pd.read_csv('diabetes.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

np.save('train', X_train)

np.save('test', X_test)
