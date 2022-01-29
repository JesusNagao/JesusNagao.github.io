import numpy as np
import random
import math
import matplotlib.pyplot as plt

from numpy.core.numeric import empty_like

class Particula():
    
    def __init__(self, xi, vi):
        self.xi = xi
        self.vi = vi
    
        
    def ackley(self):

        a = 20
        b = 0.2
        c = 2*np.pi
        d = 2

        x_sum = 0
        x_sum_cos = 0

        j=0
        while j<d:
            x_sum += self.xi[j]**2
            x_sum_cos += math.cos(c*self.xi[j])
            j+=1


        f = -a*math.exp(-b*math.sqrt(x_sum/d))-math.exp(x_sum_cos/d)+a+math.exp(1)

        return f


    def vel(self, alpha, beta, e1, e2, f_max_loc):
        for i in range(self.vi.size):
            
            #print(i)
            
            self.vi[i] = self.vi[i] + alpha*e1*(g-self.xi[i]) + beta*e2*(f_max_loc-self.xi[i])

    def pos(self):
        for i in range(self.vi.size):
            self.xi[i] = self.xi[i] + self.vi[i]

    def posicion1(self):
        return self.xi[0]

    def posicion2(self):
        return self.xi[1]


    

def max_ack(f, n):
        i = 0
        best = f[i].ackley()
        index = i
        while i<n:
            if f[i].ackley() > best:
                best = f[i].ackley()
                index = i
            i+=1

        return best, index

def max(f, n):
        i = 0
        best = f[i]
        index = i
        while i<n:
            if f[i] > best:
                best = f[i]
                index = i
            i+=1

        return best, index


def evaluar_3D(X,Y):
    a = 20
    b = 0.2
    c = 2*np.pi
    d = 2

    x_sum = 0
    x_sum_cos = 0

    f = np.empty_like(X)
    #print(range(X[0,:].size))
    
    for i in range(X[0,:].size):
        for j in range(Y[:,0].size):
            x_sum = X[i,j]**2 + Y[i,j]**2
            x_sum_cos = math.cos(c*X[i,j]) + math.cos(c*Y[i,j])
            f[i,j] = -a*math.exp(-b*math.sqrt(x_sum/d))-math.exp(x_sum_cos/d)+a+math.exp(1)


    return f
    
    


n_part = 10
part = [None] * n_part
it = 1000


for i in range(n_part):
    xi = random.uniform(-32.768, 32.768)*np.random.random_sample(size = 2)
    vi = empty_like(xi)
    part[i] = Particula(xi,vi)

g, ind_g = max_ack(part, n_part)
best_x0 = part[ind_g].posicion1()
best_x1 = part[ind_g].posicion2()



j = 0
alpha = (2 + random.uniform(-0.25,0.25))**np.random.random_sample(size = n_part)
beta = (2 + random.uniform(-0.25,0.25))**np.random.random_sample(size = n_part)
e1 = random.random()*np.random.random_sample(size = n_part)
e2 = random.random()*np.random.random_sample(size = n_part)
xi_g = 0

while j<it:

    f = empty_like(alpha)

    for i in range(n_part):
        
        part[i].vel(alpha[i], beta[i], e1[i], e2[i], xi_g)
        part[i].pos()
        f[i] = part[i].ackley()
        xi_g, index = max(f, n_part)

    if(g<xi_g):
        g = xi_g
        best_x0 = part[index].posicion1()
        best_x1 = part[index].posicion2()

    j += 1


print(f"El maximo global es {g}")
print(f"La posicion x0 para el maximo global es {best_x0}")
print(f"La posicion x1 para el maximo global es {best_x1}")



x = np.linspace(-33, 33, 66)
y = np.linspace(-33, 33, 66)
X, Y = np.meshgrid(x, y)
Z = evaluar_3D(X,Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
for i in range(n_part):
    ax.scatter(part[i].posicion1(), part[i].posicion2(), part[i].ackley())
ax.scatter(best_x0, best_x1, g, c='r')
ax.legend(['posicion', 'x*', 'g*'])
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x)')
ax.view_init(60, 35)
plt.show()

