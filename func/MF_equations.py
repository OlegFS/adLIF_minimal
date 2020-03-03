import numpy as np
from scipy.integrate import odeint,solve_ivp,ode
from numpy import arange
from scipy.optimize import fsolve as fsolve
from math import erf as erfp
from math import exp as exp
import seaborn as sns
from scipy.optimize import root as root
import scipy
na = np.array
from math import erf
from scipy.integrate import quad
from scipy.special import erfcx,ndtr,erf
pi_sqrt = np.sqrt(np.pi);


pi_sqrt = np.sqrt(np.pi);

def F(mu,sigma,params):
#     print(mu,sigma)
    theta = params['theta']
    Vr = params['V_res']#/theta
    tau_m = params['tauMem']#/1000
#     mu/=theta
#     sigma/=theta
    
#     r= tau_m*pi_sqrt
    def integrand(x):
        return erfcx(-x)#exp((x**2))*(1+erfp(x))#
    
    upper=(theta-mu)/sigma
    lower=(Vr-mu)/sigma
#     print(upper,lower)
    I= quad(integrand, lower, upper)[0]
    # add tau_rp??
    tau_rp = params['t_ref']#/1000
    nu0 = tau_rp +tau_m*pi_sqrt*I#
    return 1/nu0

def mu_(nu0, params):
#     print(nu0)
    nu_ext=params['p_rate']
    Ke = params['Ke']
    tau_m = params['tauMem']#/1000
    tau_w = params['tau_w']
    g = params['g']
    gamma = params['gamma']
    J = params['J']
    J_ext = params['J_ext']
    b = params['b']
    mu_ext = nu_ext*J_ext*tau_m
    mu_rec =tau_m*Ke*J*nu0*(1-(g*gamma))#tau_m*b
    return  mu_ext+mu_rec-b#*100

def mu_w(nu0,w, params):
#     print(nu0)
    nu_ext=params['p_rate']
    Ke = params['Ke']
    tau_m = params['tauMem']#/1000
    tau_w = params['tau_w']
    g = params['g']
    gamma = params['gamma']
    J = params['J']
    J_ext = params['J_ext']
    mu_ext = nu_ext*J_ext*tau_m
    mu_rec =tau_m*Ke*J*nu0*(1-(g*gamma))
    return  mu_ext+mu_rec-w #*100    

def sigma_sq(nu0,params):
    nu_ext=params['p_rate']
    Ke = params['Ke']
    tau_m = params['tauMem']#/1000
    g = params['g']
    gamma = params['epsilon']
    J = params['J']
    J_ext = params['J_ext']
    sigma_ex =tau_m*nu_ext*(J_ext**2)
    sigma_rec = (tau_m*Ke*(J**2)*nu0)*(1+((g**2)*gamma))
    return np.sqrt((sigma_ex+sigma_rec))
    
def ps_dyn(nu,t,params):
    mu = mu_(nu,params)
    sigma = sigma_sq(nu,params)
    return -nu+F(mu,sigma,params)




def ps_dynW(nu0,t,w,params):
#     print(mu,sigma)
    mu = mu_w(nu0,w,params)
    sigma = sigma_sq(nu0,params)
    print(mu,w,sigma)
    return -nu0+F(mu,sigma,params)
    

def MFI_dyn(x,t,params):
    nu0,w = x
    tau_w = params['tau_w']
    tau_m = params['tauMem']
    b = params['b']
    t_ = np.arange(0,1000,0.1)
    mu = mu_w(nu0,w,params)
    sigma = sigma_sq(nu0,params)
#     print(mu,w,sigma) 
#     sol1 = odeint(ps_dynW, (Nus[-1]), t_, args =(w,params))
#     print(sol[-1])
#     sol2 = odeint(ps_dynW, (10.), t_, args =(w,params))
#     print(sol1[-1][0],sol2[-1][0])
#     nu = np.min([sol1[-1][0],sol2[-1][0]])
#     print(nu)
#     nu = sol1[-1][0]
#     if nu<1.:
#         nu = 0
    Mu.append(mu)
    Si.append(sigma)
    Ts.append(t)
    dnu =-nu0+F(mu,sigma,params)
    dw = -(w/tau_w)+(b*nu0)
#     Ts.append(t)
#     Nus.append(nu)
    
    return (dnu,dw)
