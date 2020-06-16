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

print(lb.dectrees.entropy(2,3))

# cálculo de soma de linha de entropia

print(lb.dectrees.parameter_entropy_sum(np.array([2,3,4])))

# cálculo de entropia geral para um parâmetro

list = [[3,0,0],[3,2,0],[1,0,4]]

print(lb.dectrees.overall_entropy(list))