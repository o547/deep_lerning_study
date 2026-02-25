import numpy as np
import Activation
X=np.array([1,2])
W=np.array([[3,4,5],[6,7,8]])
B=np.array([0.1,0.2,0.3])
A=np.dot(X,W)+B
act=Activation.Activation()
Z=act.sigmoid(A)
print(Z)