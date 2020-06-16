import libprova as lb
import numpy as np

# cálculo de saída de perceptron com relu

x = np.array([-3,10,-2])
w = np.array([1,2,4])

print(lb.relu(lb.wsum(w,x)))

# cálculo de distância euclidiana

a = np.array([0,0])
b = np.array([0,2.4])
print(lb.kmeans.calc_distance(a,b))

# cálculo de entropia

samples = np.array([7,2,4])

print(lb.overall_entropy(samples))