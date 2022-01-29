import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
import random


#Crear un arreglo con los datos obtenidos del archivo genero.txt
x=pd.read_csv('genero.txt', skiprows=1, usecols=[1]).to_numpy()
y=pd.read_csv('genero.txt', skiprows=1, usecols=[2]).to_numpy()

#Dividir los arreglos 'x' y 'y' en sets de entrenamiento y de prediccion utilizando una proporcion
#80% y 20% respectivamente

x_train=x[:round(x.size*0.8)]
x_pred=x[round(x.size*0.2):]

y_train=y[:round(y.size*0.8)]
y_pre=y[round(y.size*0.2):]


#Usar la regresion linear de scikit-learn para crear un modelo predictivo
reg=lr().fit(x_train, y_train)
y_pred=reg.predict(x_pred)

#Graficar todos los puntos y la linea de regresion
plt.scatter(x_train,y_train,color='black')
plt.plot(x_pred,y_pred)


#Calcular el MSE a partir de los datos predecidos por la regresion y los datos reales
mse=0
i=0
n=y_pred.size

#Sumatoria de las diferencias al cuadrado
while i<n:

    mse+=(y_pred[i]-y_pre[i])**2
    i+=1

mse=mse/n

print(f'MSE = {mse}')


#Batch Gradient Descent#########################################################################

#Valor alpha 'Learning Rate'
a=abs(random.gauss(0.5, 0.25))

#Inicializando valores
j=0
i=0
b=np.empty_like(x)

while i<x.size:
    b[i]=random.gauss(0.5, 0.25)
    i+=1

#Iteraciones para obtener el valor del gradiente y los coeficientes de beta
i=0
while i<10:

    grad_f=(np.transpose(x)*(x*b-y))*2/x.size

    b=b-a*grad_f

    i+=1

y_pred_grad=np.dot(np.transpose(b),x)


#Obteniendo el MSE de Batch Gradient Descent
i=0
mse_grad=0
while i<n:

    mse_grad+=(y_pred_grad[i]-y_pre[i])**2
    i+=1

mse_grad=mse_grad/n

print(f'MSE de Batch Gradient Descent = {mse_grad}')

plt.show()
