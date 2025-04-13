import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import os
import scipy.integrate as integ
import sys
from numba import njit

D_n = 0.2 + (int(sys.argv[1])+1)*(0.5 - 0.2)/64
R = 1

#dx = 1E-4
N_timestep = 1E3

dt = (1E-3)/N_timestep

tamsad_dir = f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/tamsad"
tamsd_dir = f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/tamsd"

files_tamsad = [file for file in os.listdir(tamsad_dir)]
files_tamsd = [file for file in os.listdir(tamsd_dir)]

N = len(files_tamsad)
l = len(np.loadtxt(os.path.join(tamsad_dir, files_tamsad[0])))
T = np.arange(0, (int(l)-1)*dt, dt)

msad = np.zeros(int(l))
msd = np.zeros(int(l))

for i in range(N):
    tamsad = np.loadtxt(os.path.join(tamsad_dir, files_tamsad[i]))
    tamsd = np.loadtxt(os.path.join(tamsd_dir, files_tamsd[i]))

    if np.any(np.isnan(tamsad)):
        print(os.path.join(tamsad_dir, files_tamsad[i]))
    if np.any(np.isnan(tamsd)):
        print(os.path.join(tamsad_dir, files_tamsd[i]))

    msad += tamsad
    msd += tamsd

msad = msad/N #Ensemble Averaged MSD
msd = msd/N #Ensemble Averaged MSD

np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_Euler/N={N_timestep:.2e}/tB={tB:.6f}/msad.txt", msad)
np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_Euler/N={N_timestep:.2e}/tB={tB:.6f}/msd.txt", msd)
                        
plt.title("MSD Plot")
plt.xlabel("Time")
plt.ylabel("MSD/R")
plt.grid(visible=True)
plt.plot(T, msd)
plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_Euler/N={N_timestep:.2e}/tB={tB:.6f}/msd.png")
plt.close()
                        
plt.title("MSAD Plot")
plt.xlabel("Time")
plt.ylabel("MSAD")
plt.grid(visible=True)
plt.plot(T, msad)
plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_Euler/N={N_timestep:.2e}/tB={tB:.6f}/msad.png")
plt.close()
                        
plt.title("MSD Plot")
plt.xlabel("Time")
plt.ylabel("MSD/R")
plt.grid(visible=True)
plt.plot(T, msd)
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_Euler/N={N_timestep:.2e}/tB={tB:.6f}/ll_msd.png")
plt.close()
                        
plt.title("MSAD Plot")
plt.xlabel("Time")
plt.ylabel("MSAD")
plt.grid(visible=True)
plt.plot(T, msad)
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_Euler/N={N_timestep:.2e}/tB={tB:.6f}/ll_msad.png")
plt.close()

llmsad = np.log10(msad)
llmsd = np.log10(msd)

slope_msad = (np.roll(llmsad, shift = -1) - llmsad)/dt
slope_msd = (np.roll(llmsd, shift = -1) - llmsd)/dt

plt.title("MSD Plot")
plt.xlabel("Time")
plt.ylabel("MSD")
plt.grid(visible=True)
plt.plot(T[0:-1], slope_msd[0:-1])
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_Euler/N={N_timestep:.2e}/tB={tB:.6f}/ll_slope_msd.png")
plt.close()
                        
plt.title("MSAD Plot")
plt.xlabel("Time")
plt.ylabel("MSAD")
plt.grid(visible=True)
plt.plot(T[0:-1], slope_msad[0:-1])
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_Euler/N={N_timestep:.2e}/tB={tB:.6f}/ll_slope_msad.png")
plt.close()

pi = np.pi

def msad_pred(t):
    r1 = 1 - ((3*pi**2)*np.exp(-2*D_d*t/R**2)/(4*pi**2 - 16))

    return r1*(pi**2 - 4)/2

msad_prediction = msad_pred(T)

print(msad_prediction)
                        
plt.title("MSAD Plot")
plt.xlabel("Time")
plt.ylabel("MSAD")
plt.grid(visible=True)
plt.plot(T, msad, label = 'simulation')
plt.plot(T, msad_prediction, label = 'first order prediction')
plt.legend()
plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_Euler/N={N_timestep:.2e}/tB={tB:.6f}/msad_comparison.png")
plt.close()