import numpy as np 
import random
import math

def eval(x, n_padres, n_hijos, a, b, c):
    x_sum = np.empty(n_hijos)
    x_sum_cos = np.empty(n_hijos)

    
    for i in range(n_hijos):
        j=0
        while j<n_padres:
            x_sum[i] += x[i,j]
            x_sum_cos[i] += math.cos(c*x[i,j])
            j+=1
    
    y = np.empty_like(x_sum)

    i=0
    while i<n_hijos:
       
        y[i] = -a*math.exp(b*math.sqrt(abs(x_sum[i]/n_padres)))-math.exp(x_sum_cos[i]/n_padres)+a+math.exp(1)

        i+=1

    return y
    

def mutar(father, child_n):
    children = np.empty((child_n,father.size))
    i=0
    while i<father.size:
        j=0
        while j<child_n:
            children[j,i] = father[i] + (-1)**np.random.randint(0,2)
            j+=1
        i+=1

    return children

def max(x, padre):

    i = 0
    best = x[i] + padre[i]
    while i<x.size:
        j=0
        while j<padre.size:
            if x[i] + padre[j] > best:
                best = x[i] + padre[j]
            j+=1
        i+=1

    return best

n_padres = 2
x_min = -32.768
x_max = 32.768
padre = np.random.randint(x_min, x_max)*np.random.random_sample(size=n_padres)
a = 20
b = 0.2
c = 2*math.pi
n_hijos = 3
iterations = 1



cont = 0
while cont<iterations:

    children = mutar(padre, n_hijos)
    
    child_eval = eval(children, n_padres, n_hijos, a, b, c)

    padre = max(child_eval, padre)

    cont+=1


print(f'La mejor evaluacion es = {padre}')
