import random as rd
import numpy as np
import math

def eq(u,v):
    v.llenar(u.get_profit(), u.get_weight())

def markov_weight(u, v, n_espacios, p, w, T, a):
    
    for i in range(0, n_espacios):
        calcular_vecino(u, v, n_espacios, w, p)
        if (get_weight(v, n_espacios) < get_weight(u, n_espacios)):
            u = v
        else:
            if(rd.uniform(0,1)<math.exp(get_weight(v, n_espacios)-get_weight(u, n_espacios))/T):
                u = v
    
    T = a*T
    return u, T

def markov_profit(u, v, n_espacios, p, w, T, a):
    
    for i in range(0, n_espacios):
        calcular_vecino(u, v, n_espacios, w, p)
        if (get_profit(u, n_espacios) < get_profit(v, n_espacios)):
            u = v
        else:
            if(rd.uniform(0,1)<math.exp(get_profit(u, n_espacios)-get_profit(v, n_espacios))/T):
                u = v
    
    T = a*T
    return u, T

    


'''
def func_eval(u, p):
    fu = np.sum(np.asarray(u) * np.asarray(p))
    return fu

def distance(u, w, w_max):
    wi = np.sum(np.asarray(u) * np.asarray(w))
    return (wi-w_max)**2
'''

def calcular_vecino(u, v, n_espacios, w, p):
    num = rd.randint(0, n_espacios-1)
    for i in range(0, n_espacios):
        v[i].llenar(u[i].get_weight(), u[i].get_profit())
        if(i == num):
            v[i].llenar(w[num], p[num] )

def generar_eval(espacio, n_espacios, pesos, ganancias):

    for i in range(0, n_espacios):
        num = rd.randint(0, n_espacios-1)
        espacio[i].llenar(pesos[num], ganancias[num])


    return espacio

def get_profit(espacio, n_espacios):
    suma = 0
    for i in range(0, n_espacios):
        suma = suma + espacio[i].get_profit()
    return suma

def get_weight(espacio, n_espacios):
    suma = 0
    for i in range(0, n_espacios):
        suma = suma + espacio[i].get_weight()
    return suma