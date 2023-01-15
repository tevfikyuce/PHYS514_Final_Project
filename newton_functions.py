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

#Function that calculates (R,M) pair for given rho_c and D
def calculate_R_M(rho_c, D, q, K_star):
    #Parameters
    C = (5 * K_star * np.power(D, 5/q)) / 8
    max_step = 1e5

    #Function to be used in IVP solution (RHS of mass-density ODE)
    def f_rhs(t, y):
        rhs = np.zeros(len(y)) #Initialize RHS to zero

        x = np.power(y[1]/D, 1/q) #Calculate x
        df_dx = (8*C*np.power(x, 4))/(np.sqrt(np.power(x,2) + 1)) #Calculate dP/dx


        rhs[0] = 4*np.pi*np.power(t, 2) * y[1]
        if t == 0:
            rhs[1] = 0
        else:
            rhs[1] = (1/df_dx)*q*D*np.power(y[1]/D, -(1/q)+1) * (-constants.gravitational_constant*y[0]*y[1]/np.power(t, 2))
        
        return rhs

    #Initial value vector of IVP
    y0 = np.asarray([0, rho_c]).astype(float)

    #Solution of IVP
    solution = solve_ivp(fun=f_rhs, y0=y0, t_span=(0,3e7), method='RK45')
    
    #Return R-M pair
    return solution.t[-1], solution.y[0,-1]

#This function finds appropriate min and max central density values for given D
def find_central_density_limits(D, rho_c_initial, n_search_samples, min_R, max_R, q, K_star):
    #Broad range of central density values
    rho_c_vals = rho_c_initial * np.logspace(0.15, 10, n_search_samples)

    #Finding R values of rho_c values
    R_vals = []
    for rho_c in rho_c_vals:
        R,M = calculate_R_M(rho_c=rho_c, D=D, q=q, K_star=K_star)
        R_vals.append(R)

    R_vals = np.asarray(R_vals).astype(float) #Convert to numpy array

    rho_c_min_idx = np.abs(R_vals - min_R).argmin() #İndex of closest element to min R
    rho_c_max_idx = np.abs(R_vals - max_R).argmin() #İndex of closest element to max R

    return rho_c_vals[rho_c_min_idx], rho_c_vals[rho_c_max_idx]

#Function that finds D value
def find_D(radius_arr, mass_arr, rho_c_initial, D_initial, N_samples, q, K_star):
    #Function to be used in built-in minimization function
    def fun_minimize_D(ln_D):
        D = np.exp(ln_D)
        rho_c_min, rho_c_max = find_central_density_limits(D=D, rho_c_initial=rho_c_initial, n_search_samples=10, min_R=np.min(radius_arr), max_R=np.max(radius_arr), q=q, K_star=K_star)
        rho_c_vals = np.logspace(np.log10(rho_c_min), np.log10(rho_c_max), N_samples)
        #err = calculate_err_D(spline_func=spline, rho_c_vals=rho_c_vals, D=np.exp(ln_D))
        #Calculate R-M pairs for given central density vals
        R_vals = []
        M_vals = []
        for rho_c in rho_c_vals:
            R,M = calculate_R_M(rho_c=rho_c, D=D, q=q, K_star=K_star)
            R_vals.append(R)
            M_vals.append(M)

        R_vals = np.asarray(R_vals).astype(float)
        M_vals = np.asarray(M_vals).astype(float)
        
        #Fitting Spline to calculated values
        spline = CubicSpline(x=R_vals, y=M_vals)

        #Calculate Error
        err = np.sqrt( np.mean(np.power(mass_arr - spline(radius_arr), 2)) ) / solar_mass

        print('ln(D) = ' + str(ln_D) + '  Err = ' + str(err))

        return err

    result = minimize(fun=fun_minimize_D, x0=[np.log(D_initial)], options={'disp':True, 'ftol':1e-12}, bounds=[(10,np.log(D_initial))])

    return np.exp(result.x[0])