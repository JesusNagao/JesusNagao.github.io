import random
import pandas as pd
import numpy as np

#Genero.txt
x=pd.read_csv('genero.txt', skiprows=1, usecols=[1]).to_numpy()
y=pd.read_csv('genero.txt', skiprows=1, usecols=[2]).to_numpy()


a=abs(random.gauss(0.5, 0.25))
j=0
i=0

b=np.empty_like(x)

while i<x.size:

    b[i]=random.gauss(0.5, 0.25)
    i+=1

i=0


while i<10:

    grad_f=(np.transpose(x)*(x*b-y))*2/x.size

    b=b-a*grad_f

    i+=1


print(np.dot(np.transpose(b),x).shape)