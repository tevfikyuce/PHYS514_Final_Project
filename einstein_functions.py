import numpy as np
from scipy.integrate import solve_ivp
import scipy.constants as constants
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.optimize import curve_fit

#Constants used in calculations
solar_mass = 1.98847e30 #kg
earth_radius = 6.3781e6 #m

#This function solves TOV equation for given central pressure and return R and M
def solve_TOV(p_c, R_f=50):
    #Constant Parameters
    K_NS = 100
    pressure_limit = 1e-8

    #Calculates RHS of TOV equation
    def TOV_rhs(r, y):
        rhs = np.zeros(len(y)) #Initialize to zero

        rho = np.sqrt(y[2]/K_NS)

        #Fill the rhs
        rhs[0] = 4 * np.pi * np.power(r, 2) * rho
        if r == 0:
            rhs[1] = 0
            rhs[2] = 0
        else:
            rhs[1] = 2 * (y[0] + 4 * np.pi * np.power(r, 3) * y[2]) / (r * (r - 2*y[0]))
            rhs[2] = -(1/2) * (y[2] + rho) * rhs[1]
        
        return rhs

    #Define initial vector
    y0 = np.asarray([0, 0, p_c]).astype(float)

    result = solve_ivp(fun=TOV_rhs, t_span=(0,R_f), y0=y0, method='RK45')

    if result.y[2,-1] > pressure_limit:
        return None, None
    else:
        return result.t[-1], result.y[0,-1]

#This function solves TOV equation with baryonic mass for given central pressure and return R and M
def solve_TOV_mp(p_c, R_f=50):
    #Constant Parameters
    K_NS = 100
    pressure_limit = 1e-8

    #Calculates RHS of TOV equation
    def TOV_rhs(r, y):
        rhs = np.zeros(len(y)) #Initialize to zero

        rho = np.sqrt(y[2]/K_NS)

        #Fill the rhs
        rhs[0] = 4 * np.pi * np.power(r, 2) * rho #dm/dr
        if r == 0:
            rhs[1] = 0
            rhs[2] = 0
        else:
            rhs[1] = 2 * (y[0] + 4 * np.pi * np.power(r, 3) * y[2]) / (r * (r - 2*y[0])) #dv/dr
            rhs[2] = -(1/2) * (y[2] + rho) * rhs[1] #dp/dt
            rhs[3] = 4 * np.pi * np.power(1 - (2*y[0]/r), -1/2) * np.power(r, 2) * rho #dm_p/dr

        return rhs

    #Define initial vector
    y0 = np.asarray([0, 0, p_c, 0]).astype(float)

    result = solve_ivp(fun=TOV_rhs, t_span=(0,R_f), y0=y0, method='RK45')

    if result.y[2,-1] > pressure_limit:
        return None, None, None
    else:
        return result.t[-1], result.y[0,-1], result.y[3,-1]

