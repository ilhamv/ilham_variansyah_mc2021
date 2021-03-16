import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import h5py
import scipy as sp
import scipy.integrate


# Load solution
K_list   = [6,8,10,20,40,60,80,100,200,400,600,800,1000,2000,4000,6000]
pow_BE   = []
pow_MBTD = []
pow_CN   = []
for K in K_list:
    data_BE   = np.load('BE/K=%i.npz'%K)
    data_MBTD = np.load('MBTD/K=%i.npz'%K)    
    pow_BE.append(data_BE['pow'][-1])
    pow_MBTD.append(data_MBTD['pow'][-1])
    data_CN   = np.load('CN/K=%i.npz'%K)
    pow_CN.append(data_CN['pow'][-1])
pow_BE = np.array(pow_BE)
pow_MBTD = np.array(pow_MBTD)
pow_CN = np.array(pow_CN)
data_ref = np.load('CN/K=10000.npz')
pow_ref  = data_ref['pow'][-1]

K = np.array(K_list)
T = 60.0
dt = T/K

err_BE = abs(pow_BE - pow_ref)/pow_ref*100
err_CN = abs(pow_CN - pow_ref)/pow_ref*100
err_MBTD = abs(pow_MBTD - pow_ref)/pow_ref*100

first = dt
second = dt**2

fig = plt.figure(figsize=(4,4))
plt.plot(dt,err_BE,'-ob',fillstyle='none',label='BE')
plt.plot(dt,err_MBTD,'-sr',fillstyle='none',label='MBTD')
plt.plot(dt,err_CN,'-Dg',fillstyle='none',label='CN')
plt.plot(dt,first/first[-1]*2E-4,'--k',label='1st order')
plt.plot(dt,second/second[-1]*3E-6,':k',label='2nd order')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid()
plt.xlabel(r'$\Delta t$, s')
plt.ylabel(r'Relative error, %')
plt.savefig('numerical_accuracy_inf.svg', bbox_inches = 'tight', pad_inches = 0, dpi=1200)
plt.show()

t = np.linspace(0.0,T,10000+1)
pow_ref = data_ref['pow']
Tf_ref = data_ref['Tf']
fig = plt.figure(figsize=(4,4))
ax1 = plt.subplot()
l1 = plt.plot(t,pow_ref,'-b',label=r'$P$')
plt.ylabel('Power (W)')
plt.xlabel('Time (s)')
plt.grid()
ax2 = ax1.twinx()
l2 = plt.plot(t,Tf_ref,'--r',label=r'$T_f$')
plt.ylabel('Fuel Temperature (K)')
ln = l1+l2
lab = [l.get_label() for l in ln]
plt.legend(ln, lab)
plt.savefig('solution_inf.svg', bbox_inches = 'tight', pad_inches = 0, dpi=1200)
plt.show()