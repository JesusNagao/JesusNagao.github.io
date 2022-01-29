import math
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size


class Vendedores:

    def __init__(self, n_ciudades, iteraciones, coordenadas):
        self.n_ciudades = n_ciudades
        self.it = iteraciones
        self.coor = coordenadas
        self.main()

        

    def dist(self, ciudad):
        d = math.sqrt(ciudad[0]**2+ciudad[1]**2)
        return d

    def cadenaMarkov(self, du, dv, u, v,T):

        if dv<du:
            u = v
        else:
            if np.random.randint(0,1)<math.exp(-(dv-du)/T):
                u=v

        return u

    def distBest(self, best, coor):

        d = 0

        
        for i in range(size(best)):
            
            d += self.dist(coor[int(best[i]),:])
        
        return d


    def vecino(self, lk):

        v = np.random.randint(0, self.n_ciudades)
        band = True

        while band:
            j=1
            for i in range(size(lk)):
                if v == lk[i]:
                    v = np.random.randint(0, self.n_ciudades)
                    j=0

            if j!=0:
                band = False


        return v

    def main(self):
        a = random.random()
        cont = 0

        while cont<self.it: 

            
            u = np.random.randint(0,self.n_ciudades)
            lk = np.zeros(self.n_ciudades)
            best_lk = np.zeros(self.n_ciudades)


            for i in range(size(lk)):
                lk[i] = -1
                best_lk[i] = -1

            lk[0] = u

            d=0
            T=0.1
            i = 0

            
            for l in range(self.n_ciudades):
                
                v = self.vecino(lk)
            
                du = self.dist(self.coor[u,:])
                dv = self.dist(self.coor[v,:])


                u = self.cadenaMarkov(du, dv, u, v, T)
                d += self.dist(self.coor[u,:])

                T = a*T

                lk[i] = u
                i+=1


            d_best = self.distBest(best_lk, self.coor)

            if d_best>0:
                if d_best>d:
                    best_lk = lk
            else:
                best_lk = lk
            

            cont+=1


        ax = plt.axes()

        for i in range(size(lk)-1):
            ax.arrow(self.coor[int(lk[i]),0], self.coor[int(lk[i]),1], self.coor[int(lk[i+1]),0]-self.coor[int(lk[i]),0], self.coor[int(lk[i+1]),1]-self.coor[int(lk[i]),1], head_width=0.3, head_length=0.2, fc='lightblue', ec='black')


        plt.scatter(self.coor[:,0], self.coor[:,1])
        plt.show()


n_ciudades = 5
coor = np.random.randint(1,2)*np.random.random_sample(size=(n_ciudades, 2))*10
iteraciones = 100
k = 3

vend = [None] * k

for i in range(k):
    vend[i] = Vendedores(n_ciudades, iteraciones, coor)
