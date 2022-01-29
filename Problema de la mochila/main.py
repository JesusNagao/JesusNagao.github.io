#Este codigo resuelve el problema de la mochila (knapsack problem) utilizando recocido simulado


import random as rd
from operations import calcular_vecino
from operations import generar_eval
from espacio import Espacio
from operations import markov_weight
from operations import markov_profit
from operations import get_weight
from operations import get_profit

n=10 #Numero de diferentes articulos que se pueden incluir en la mochila
p = [None] * n #Inicializar el arreglo de ganancias 
w = [None] * n #Inizializar el arreglo de pesos 

#Dar valores de pesos y ganancias a cada articulo
for i in range(0, n):
    p[i] = rd.randint(0,10) 
    w[i] = rd.randint(0,10)

n_espacios = 10 #Numero de espacios disponibles en la mochila
espacio = [None] * n_espacios 
v = [None] * n_espacios 

for i in range(0, n_espacios):
    espacio[i] = Espacio()
    v[i] = Espacio()


#Inicializar los valores de las constantes y variables del algoritmo
T0 = 0.1
u = generar_eval(espacio, n_espacios, w, p)
iteraciones = 10
cont = 0
a = 0.05

#Iterar el algoritmo para resolver el problema
while(cont<iteraciones):
    
    calcular_vecino(u, v, n_espacios, w, p)
    u, T0 = markov_weight(u,v, n_espacios, p, w, T0, a)
    u, T0 = markov_profit(u,v, n_espacios, p, w, T0, a)
    cont = cont + 1

print('Weight =',get_weight(u, n_espacios))
print('Profit', get_profit(u, n_espacios))