# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:01:42 2017

@author: SHALINI GANGULY
"""

import numpy as np
from matplotlib import pyplot as plt
from lmfit import Model
from scipy import integrate
from scipy import optimize
legend_properties = {'weight':'bold'}

data = np.loadtxt("time_lag_data.txt")
E = data[:,0]
t_obs = data[:,1]
err = data[:,2]

E_0 = 11.34
H_0 = 67.3
O_M = 0.315
O_L = 1 - O_M
z = 1.41

#n=0 
def intrinsic(x,tau,alpha):
    return (tau*(x**alpha - E_0**alpha))

#
#def lnlike(theta, E, err):
#    t, a = theta
#    model = t*(E**a - E_0**a)
#    inv_sigma2 = 1.0/err**2
#    return -0.5*(np.sum((t_obs-model)**2*inv_sigma2 - np.log(inv_sigma2)))
#

def chi2_n0(theta,E,err):
    t,a=theta
    model = t*(E**a - E_0**a) 
    term = (t_obs-model)/err                                                                                              
    return np.sum(term ** 2)   


## maximize likelihood <--> minimize negative likelihood
#def neg_log_likelihood(theta, E, err):
#    return -lnlike(theta, E, err)
#
theta_guess = [10.0, 0.1]
params_n0 = optimize.fmin_powell(chi2_n0, theta_guess, args=(E, err))
print("Maximum likelihood estimate for 37 data points: \ntau= ", params_n0[0], "\nalpha= ", params_n0[1])
print chi2_n0(params_n0,E,err)/35.0
#plt.figure()
#x = np.linspace(1, 1E5, 10000)
#y = intrinsic(x,params[0],params[1])
#plt.plot(x, y, '-')
#plt.xscale("log")
#plt.yscale("log")
#plt.xlim(30,22000)
#plt.ylim(0.1,10)
#plt.xlabel(r'$ E $ (keV)')
#plt.ylabel(r'$ \Delta t_obs $ (s)')
#plt.errorbar(E, t_obs, yerr=err, fmt=' .k')

n=1
def integrand(i):
    return ((1+i)**n/np.sqrt(O_M*(1+i)**3+O_L))
I1, error = integrate.quad(integrand, 0, z)
def linear(x,tau,alpha,E_qg):
    return (tau*(x**alpha - E_0**alpha) + (-(1+n)/(2*H_0*3.24))*(x**n-E_0**n)*I1*(10**14)/(E_qg**n))

#def lnlike(theta, E, err):
#    t, a, E_qg = theta
#    model = t*(E**a - E_0**a) + (-(1+n)/(2*H_0))*(E**n-E_0**n)*I1/(E_qg**n)
#    inv_sigma2 = 1.0/err**2
#    return -0.5*(np.sum((t_obs-model)**2*inv_sigma2 - np.log(inv_sigma2)))
#    
#def neg_log_likelihood(theta, E, err):
#    return -lnlike(theta, E, err)

def chi2_n1(theta, E_qg, E, err):                                                                                               
    t, a  = theta                                                                                                    
    model = t*(E**a - E_0**a) + (-(1+n)/(2*H_0*3.24))*(E**n-E_0**n)*I1*(10**14)/(E_qg**n)                                          
    term = (t_obs-model)/err                                                                                              
    return np.sum(term ** 2)    


theta_guess = [1.1, 0.17]
#E_QG=1.0e+10
x=np.logspace(7,19,24)
print(x)
params = [optimize.fmin(chi2_n1, theta_guess, args=(E_QG,E, err)) for E_QG in x]
print len(params)
for E_QG in x:
    chi2=[chi2_n1(ps,E_QG,E,err) for  ps in params]
chi2min=min(chi2)
#print("n=1:Maximum likelihood estimate for 37 data points: \ntau= ", params[0], "\nalpha= ", params[1])
#print chi2_n1(params,E,err)-chi2min
print (chi2-chi2min)
#plt.figure()
#ax=plt.axes([0.17,0.17,0.8,0.8])
plt.loglog(x,chi2-chi2min,'k-*',lw=2)
plt.xlabel('$E_{QG}$ (GeV)',fontsize=14,fontweight='bold')
plt.ylabel('$\Delta \chi^2$',fontsize=14,fontweight='bold')
plt.legend(['n=1 LIV'],fontsize=14,frameon=False)
#plt.tick_params(labelsize=18)
plt.axhline(y=4.0,color='magenta',lw=2,linestyle='--')
#plt.axhline(x=1.0e+17,color='magenta',lw=2,linestyle='--')
plt.gca().set_ylim(bottom=5.0e-05)
plt.vlines(x=2.55e+16,ymin=5.0e-05,ymax=4.0,linestyles='dotted',color='magenta',lw=2)
plt.savefig('linearLIV.pdf')



n=2
def integrand(i):
    return ((1+i)**n/np.sqrt(O_M*(1+i)**3+O_L))
I2, error = integrate.quad(integrand, 0, z)
def quadratic(x,tau,alpha,E_qg):
    return (tau*(x**alpha - E_0**alpha) + (-(1+n)/(2*H_0*3.24))*(x**n-E_0**n)*I2*(10**8)/(E_qg**n))

#def chi2_n2(theta, E, err):
#    t, a, E_qg = theta
#    model = t*(E**a - E_0**a) + (-(1+n)/(2*H_0*3.24))*(E**n-E_0**n)*I2*(1.0e+08)/(E_qg**n)
#    term = (t_obs-model)/err
#    return np.sum(term ** 2)

#theta_guess = [2, 0.1, 1e7]
#params = optimize.fmin(chi2_n2, theta_guess, args=(E, err))
#print("n=2:Maximum likelihood estimate for 37 data points: \ntau= ", params[0], "\nalpha= ", params[1], "\nEqg= ", params[2])
#print chi2_n2(params,E,err)/34

#print chi2_n1(params,E,err)/34
#print chi2_n0(params_n0,E,err)/35
#x = np.linspace(1, 1E5, 10000)
#y = quadratic(x,params[0],params[1],params[2])
#plt.plot(x, y, ':')
#plt.show()    
