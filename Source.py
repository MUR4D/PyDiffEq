import math

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
alpha = 1.0
N = 100
L = 1.0
h = L / (N+1)
tol = 1e-6


def f(x):
    return math.sin(x)


u = [0]*(N+2)
x = [0]*(N+2)

for i in range(len(x)):
    x[i] = i*h

for i in range(len(u)):
    u[i] = x[i]

error = 1.0

while error > tol:
    error = 0.0
    for i in range(1, N+1):
        tmp = u[i]
        u[i] = (u[i-1] + u[i+1] + h*h*f(x[i])) / (2.0 + alpha * h * h * u[i] * u[i])
        error += abs(u[i]-tmp)
    error /= N

for i in range(len(u)):
    print(u[i])



#u=[0,0.100116,0.199491, 0.297438, 0.393378, 0.486883, 0.577715, 0.665853, 0.751521, 0.835203, 0.917669,1]
#x = np.linspace(10, 1.0, 1)

#sol = odeint(vanderpol, X0, t)
#print(sol)

#x = sol[:, 0]
#y = sol[:, 1]

#t = np.linspace(0, 1.0)
plt.plot(np.array(x),np.array(u))
#plt.xlabel('u')
#plt.ylabel('x')
#plt.legend(('x', 'u'))
plt.show()



