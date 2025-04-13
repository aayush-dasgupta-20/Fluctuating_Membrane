import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import os
import scipy.integrate as integ
import sys
from numba import njit
from tqdm import tqdm

#tB = (int(os.getenv("SLURM_ARRAY_TASK_ID")) - 1)/10
D_n = 0.2 + (int(sys.argv[1])+1)*(0.5 - 0.2)/64
R = 1

#dx = 1E-4
N_timestep = 1000
dt = 1E-3/N_timestep

@njit(fastmath = True)
def msds(theta, phi):
    geodesic_msd = np.zeros(int(len(theta)/2))
    geometric_msd = np.zeros(int(len(theta)/2))

    for i in range(len(geodesic_msd)):
        geodesic_msd_i = 0
        geometric_msd_i = 0
        count = 0
        for j in range(len(theta) - i):
            e1 = [np.sin(theta[j])*np.cos(phi[j]), np.sin(theta[j])*np.sin(phi[j]), np.cos(theta[j])]
            e2 = [np.sin(theta[j+i])*np.cos(phi[j+i]), np.sin(theta[j+i])*np.sin(phi[j+i]), np.cos(theta[j+i])]

            c = (e1[0]*e2[0] + e1[1]*e2[1] + e1[2]*e2[2])

            if np.abs(c)<1:
                geodesic_msd_i += (R*np.arccos(c))**2
            elif c>=1:
                geodesic_msd_i += 0
            elif c<=-1:
                geodesic_msd_i += (R*np.pi)**2 
            
            geometric_msd_i += (R**2)*((e1[0] - e2[0])**2 + (e1[1] - e2[1])**2 + (e1[2] - e2[2])**2)
            count += 1

        geodesic_msd[i] = geodesic_msd_i/count
        geometric_msd[i] = geometric_msd_i/count

    return geodesic_msd, geometric_msd


os.makedirs(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/tamsad", exist_ok = True)
os.makedirs(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/tamsd", exist_ok = True)
os.makedirs(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/Position_Plot", exist_ok = True)
os.makedirs(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/Position_Images", exist_ok = True)

theta_dir = f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/theta"
phi_dir = f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/phi"

files_theta = [file for file in os.listdir(theta_dir)]
files_phi = [file for file in os.listdir(phi_dir)]

N = len(files_theta)
l = len(np.loadtxt(os.path.join(theta_dir, files_theta[0])))

T = np.arange(0, (int(l/2))*dt, dt)

msad = np.zeros(int(l/2))
msd = np.zeros(int(l/2))

for i in range(N):
    theta = np.loadtxt(os.path.join(theta_dir, files_theta[i]))
    phi = np.loadtxt(os.path.join(phi_dir, files_phi[i]))
    tamsad, tamsd = msds(theta, phi) #time averaged MSD
    tamsad = tamsad
    tamsd = tamsd
        
    plt.title(r"$\theta-\phi$ Plot")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\phi$")
    plt.grid(visible=True)
    plt.plot(theta,phi)
    plt.xlim(0,np.pi)
    plt.ylim(0,2*np.pi)
    plt.savefig(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/Position_Plot/Trial_{i}.png")
    plt.close()

    X = R*np.sin(theta)*np.cos(phi)
    Y = R*np.sin(theta)*np.sin(phi)
    Z = R*np.cos(theta)

    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.plot3D(X,Y,Z)
    ax.set_title(f"Particle Position_D={D_n:.6f}_Trial_{i}")
    plt.savefig(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/Position_Images/Trial_{i}.png")
    plt.close()

    np.savetxt(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/tamsad/Trial_{i}.txt", tamsad)
    np.savetxt(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/tamsd/Trial_{i}.txt", tamsd)
    msad += tamsad
    msd += tamsd

msad = msad/N #Ensemble Averaged MSD
msd = msd/N #Ensemble Averaged MSD

pi = np.pi

def msad_pred(t):
    r1 = 4*D_n*np.multiply(np.square(t), t)*R**2
    r2 = (104*np.square(np.multiply(np.square(t), t))*(D_n**2)*R**2)/15

    return r1 - r2

msad_prediction = msad_pred(T)

np.savetxt(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/msad.txt", msad)
np.savetxt(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/msd.txt", msd)
np.savetxt(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/msad_prediction.txt", msad_prediction)
np.savetxt(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/T.txt", T)
                        
plt.title(f"MSD Plot_D={D_n:.6f}_Trial_{i}")
plt.xlabel("Time")
plt.ylabel("MSD/R")
plt.grid(visible=True)
plt.plot(T, msd)
plt.savefig(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/msd.png")
plt.close()
                        
plt.title(f"MSAD Plot_D={D_n:.6f}_Trial_{i}")
plt.xlabel("Time")
plt.ylabel("MSAD")
plt.grid(visible=True)
plt.plot(T, msad)
plt.savefig(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/msad.png")
plt.close()
                        
plt.title(f"log-log MSD Plot_D={D_n:.6f}_Trial_{i}")
plt.xlabel("Time")
plt.ylabel("MSD/R")
plt.grid(visible=True)
plt.plot(T, msd)
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/ll_msd.png")
plt.close()
                        
plt.title(f"log-log MSAD Plot_D={D_n:.6f}_Trial_{i}")
plt.xlabel("Time")
plt.ylabel("MSAD")
plt.grid(visible=True)
plt.plot(T, msad)
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/ll_msad.png")
plt.close()

llmsad = np.log10(msad)
llmsd = np.log10(msd)

slope_msad = (np.roll(llmsad, shift = -1) - llmsad)/dt
slope_msd = (np.roll(llmsd, shift = -1) - llmsd)/dt

plt.title(f"Slope of Log=log MSD Plot_D={D_n:.6f}_Trial_{i}")
plt.xlabel("Time")
plt.ylabel("MSD")
plt.grid(visible=True)
plt.plot(T[0:-1], slope_msd[0:-1])
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/ll_slope_msd.png")
plt.close()
                        
plt.title(f"Slope of Log-log MSAD Plot_D={D_n:.6f}_Trial_{i}")
plt.xlabel("Time")
plt.ylabel("MSAD")
plt.grid(visible=True)
plt.plot(T[0:-1], slope_msad[0:-1])
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/ll_slope_msad.png")
plt.close()
                        
plt.title(f"MSAD Plot_D={D_n:.6f}_Trial_{i}")
plt.xlabel("Time")
plt.ylabel("MSAD")
plt.grid(visible=True)
plt.plot(T, msad, label = 'simulation')
plt.plot(T, msad_prediction, label = 'first order prediction')
plt.legend()
plt.savefig(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/msad_comparison.png")
plt.close()
