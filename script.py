import libprova as lb
import numpy as np

x = np.array([-3,10,-2])
w = np.array([1,2,4])

print(lb.relu(lb.wsum(w,x)))