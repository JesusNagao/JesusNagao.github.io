import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
from mpl_toolkits.mplot3d import Axes3D
import random

#Importar y leer txt
x=pd.read_csv('mtcars.txt',skiprows=1, sep=' ', usecols=[4,7]).to_numpy()
z=pd.read_csv('mtcars.txt',skiprows=1, sep=' ', usecols=[5]).to_numpy()

#Numero de Particiones
k=3

#Dividir el arreglo en particiones
x_train0 = np.array_split(x,k)[0]
z_train0 = np.array_split(z,k)[0]

#Realizar la regresion
reg0=sklearn.linear_model.LinearRegression()
reg0.fit(x_train0,z_train0)

#Dividir el arreglo en particiones
x_train1 = np.array_split(x,k)[1]
z_train1 = np.array_split(z,k)[1]

#Realizar la regresion
reg1=sklearn.linear_model.LinearRegression()
reg1.fit(x_train1,z_train1)

#Dividir el arreglo en particiones
x_train2 = np.array_split(x,k)[2]
z_train2 = np.array_split(z,k)[2]

#Realizar la regresion
reg2=sklearn.linear_model.LinearRegression()
reg2.fit(x_train2,z_train2)

#Predecir Resultados
z_pred0=reg0.predict(x_train0)
z_pred1=reg0.predict(x_train1)
z_pred2=reg0.predict(x_train2)


#Graficar
fig0 = plt.figure()
ax = fig0.add_subplot(111, projection='3d')
ax.scatter(x_train0[:,0], x_train0[:,1], z_train0, marker='.', color='red')
ax.set_xlabel("disp")
ax.set_ylabel("wt")
ax.set_zlabel("hp")

coefs0 = reg0.coef_
intercept0 = reg0.intercept_
xs = np.tile(np.arange(350), (350,1))
ys = np.tile(np.arange(350), (350,1)).T
zs = coefs0[0,0]*np.array(xs)+coefs0[0,1]*np.array(ys)+intercept0
ax.plot_surface(xs,ys,zs, alpha=0.5)


fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(x_train1[:,0], x_train1[:,1], z_train1, marker='.', color='red')
ax.set_xlabel("disp")
ax.set_ylabel("wt")
ax.set_zlabel("hp")

coefs1 = reg1.coef_
intercept1 = reg1.intercept_
xs = np.tile(np.arange(350), (350,1))
ys = np.tile(np.arange(350), (350,1)).T
zs = coefs1[0,0]*np.array(xs)+coefs1[0,1]*np.array(ys)+intercept1
ax.plot_surface(xs,ys,zs, alpha=0.5)

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(x_train2[:,0], x_train2[:,1], z_train2, marker='.', color='red')
ax.set_xlabel("disp")
ax.set_ylabel("wt")
ax.set_zlabel("hp")

coefs2 = reg2.coef_
intercept2 = reg2.intercept_
xs = np.tile(np.arange(350), (350,1))
ys = np.tile(np.arange(350), (350,1)).T
zs = coefs2[0,0]*np.array(xs)+coefs2[0,1]*np.array(ys)+intercept2
ax.plot_surface(xs,ys,zs, alpha=0.5)


#Calculando el MSE
i=0
mse0=0
mse1=0
mse2=0


while i<z_train1.size:

    mse0+=(z_pred0[i]-z_train1[i])**2
    mse1+=(z_pred1[i]-z_train2[i])**2
    mse2+=(z_pred2[i]-z_train0[i])**2

    i+=1 

mse0=mse0/z_train1.size
mse1=mse1/z_train1.size
mse2=mse2/z_train1.size

#Sacando el promedio de los MSEs
mse=(mse0+mse1+mse2)/3
print(f'MSE = {mse}')


#Batch Gradient Descent#########################################################################
a=abs(random.gauss(0.5, 0.25))
j=0
i=0


#print(x_train0[:,0])
b=np.empty_like(x_train0[:,0])



while i<(b.size):
    j=0
    while j<1:
        b[i]=abs(random.gauss(0.5, 0.25))
        j+=1
    i+=1


i=0

#print(np.dot(np.transpose(x_train0),(x_train0*b-z_train0)).shape)


while i<10:

    grad_f=(np.transpose(x_train0[:,0])*((x_train0[:,0]*b)-z_train0))*2/(x_train0.size/2)

    b=b-a*grad_f

    i+=1

y_pred_grad=np.dot(x_train0[:,0],np.transpose(b))

#Obteniendo el MSE de Batch Gradient Descent

i=0
mse_grad=0
while i<z_train0.size:

    mse_grad+=(y_pred_grad[i]-z_train0[i])**2
    i+=1

mse_grad=mse_grad/z_train0.size

print(f'MSE de Batch Gradient Descent = {mse_grad}')


plt.show()

