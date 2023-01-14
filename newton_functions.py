import numpy as np
from scipy.integrate import solve_ivp

#Solves Lane-Emden Equation
def lane_emden_solve(n, xi_f=20):
    #n is the parameter in Lane-Emden equation
    #xi_f is the final xi value of integration 
    #!!! theta should reach zero before xi_f !!!
    y0 = np.asarray([1, 0]).astype(float) #Initial value

    #RHS of Lane-Emden equation
    def f(t, y):
        f = np.zeros(len(y)) #Initialize to zero vector

        f[0] = y[1]
        if t==0:
            f[1] = 0
        else:
            f[1] = -np.power(y[0], n) - (2/t) * y[1]
        return f

    #Calling solver
    sol = solve_ivp(fun=f, t_span=(0,xi_f), y0=y0, method='RK45', max_step = 1e-4)
    
    #Take values from solution
    xi = sol.t
    theta = sol.y[0,:]
    theta_prime = sol.y[1,:]

    #Return final values
    return xi, theta, theta_prime

