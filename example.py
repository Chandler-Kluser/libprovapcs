import libprova as lb
import numpy as np

x = np.array([-3,10,-2])
w = np.array([1,2,4])

print(lb.relu(lb.wsum(w,x)))

point1 = np.array([0,0])
point2 = np.array([1,1])

print(lb.kmeans.calc_distance(point1,point2))