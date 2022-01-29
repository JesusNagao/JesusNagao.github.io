import numpy as np 
import random
import math




#Un padre varios hijos ###################################################

def eval(x):

    y = np.empty_like(x)
    
    i=0
    while i<x.size:
        y[i] = math.exp(-(x[i]-10)**2)
        i+=1


    return y

def mutar(father, child_n):
    children = father + (-1)**np.random.randint(0,2)*np.random.random_sample(size=child_n)
    return children

def max(x):
    i = 0
    best = x[i]
    while i<x.size:
        if x[i] > best:
            best = x[i]
        i+=1

    return best



padre = np.random.randint(0,6)
n_hijos = 3
iterations = 10

cont = 0
while cont<iterations:

    children = mutar(padre, n_hijos)
    
    child_eval = eval(children)

    padre = max(child_eval)

    cont+=1


print(f'La mejor evaluacion es = {padre}')