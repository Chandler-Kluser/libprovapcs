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

entropy_S = lb.dectrees.parameter_entropy_sum(np.array([7,4,2]))

print(entropy_S)

# cálculo de entropia geral para um parâmetro

list = [[3,0,0],[3,2,0],[1,0,4]]

entropy_Si = lb.dectrees.overall_parameter_entropy_sum(list)

print(entropy_Si)

# cálculo do ganho de informação

gain = lb.dectrees.gain(entropy_S,entropy_Si)

print(gain)

# exemplo de movimento de robô em grid

gamma = 0.8

r = lb.VIP.move(0,0,gamma,1)
r = lb.VIP.move(r,10,gamma,2)
r = lb.VIP.move(r,0,gamma,3)
r = lb.VIP.move(r,0,gamma,4)
r = lb.VIP.move(r,10,gamma,5)

print(r)

# exemplo de regressão linear e cálculo de RSS

x = [43,21,25,42,57,59]
y = [99,65,79,75,87,81]

reg = lb.linearregression.linreg(x,y)

yhat = lb.linearregression.estimate(y,reg)

lb.linearregression.RSS(y,yhat)

# métricas

TP = 150
TN = 100
FP = 10
FN = 5

lb.confusionmatrix.showmetrics(TP,TN,FP,FN)

# comandos de avaliação estatística

a = lb.prob_2_z(0.79)

print(a)

a = [0.1,0.12,0.11]
b = [0.15,0.10,0.13]

a = lb.get_accuracy(a)
b = lb.get_accuracy(b)

## rejeição de hipótese

lb.wald_test(a,b)