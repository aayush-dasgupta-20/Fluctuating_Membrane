import numpy as np # type: ignore
import numpy.random as rand # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
import sys
from numba import njit # type: ignore

#tB = (int(os.getenv("SLURM_ARRAY_TASK_ID")) - 1)/10
D_n = 0.2 + (int(sys.argv[1])+1)*(0.5 - 0.2)/64
R = 1

#dx = 1E-4
N_timestep = int(1E4)
dt = 2E-2/N_timestep

@njit(fastmath = True)
def msds(x,y,z):
    geodesic_msd = np.zeros(int(len(x)/2),dtype=np.double)
    geometric_msd = np.zeros(int(len(x)/2),dtype=np.double)

    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    x_n = np.divide(x,r)
    y_n = np.divide(y,r)
    z_n = np.divide(z,r)

    for i in range(len(geodesic_msd)):
        geodesic_msd_i = 0
        geometric_msd_i = 0
        count = 0
        for j in range(len(x) - i):
            e1 = [x_n[j], y_n[j], z_n[j]]
            e2 = [x_n[j+i], y_n[j+i], z_n[j+i]]
            geo_msd_inst = (R**2)*((e1[0] - e2[0])**2 + (e1[1] - e2[1])**2 + (e1[2] - e2[2])**2)
            geometric_msd_i += geo_msd_inst
            if geo_msd_inst == 0:
                geodesic_msd_i += 0
            else:
                c = (e1[0]*e2[0] + e1[1]*e2[1] + e1[2]*e2[2])

                if np.abs(c)<1:
                    geodesic_msd_i += (R*np.arccos(c))**2
                elif c>=1:
                    geodesic_msd_i += 0
                elif c<=-1:
                    geodesic_msd_i += (R*np.pi)**2 
            
            count += 1

        geodesic_msd[i] = geodesic_msd_i/count
        geometric_msd[i] = geometric_msd_i/count

    return geodesic_msd, geometric_msd


os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/tamsad", exist_ok = True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/tamsd", exist_ok = True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/Position_Images", exist_ok = True)

x_dir = f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/x"
y_dir = f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/y"
z_dir = f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/z"

files_x = [file for file in os.listdir(x_dir)]
files_y = [file for file in os.listdir(y_dir)]
files_z = [file for file in os.listdir(z_dir)]

N = len(files_x)
print(N)
l = len(np.loadtxt(os.path.join(x_dir, files_x[0]),dtype=np.double))

dt_up = 2E-2/l

T = np.arange(0, (int(l/2))*dt_up, dt_up)

msad = np.zeros(int(l/2))
msd = np.zeros(int(l/2))

for i in range(N):
    x = np.loadtxt(os.path.join(x_dir, files_x[i]),dtype=np.double)
    y = np.loadtxt(os.path.join(y_dir, files_y[i]),dtype=np.double)
    z = np.loadtxt(os.path.join(y_dir, files_z[i]),dtype=np.double)
    tamsad, tamsd = msds(x,y,z) #time averaged MSD
    tamsad = tamsad
    tamsd = tamsd

    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.plot3D(x,y,z)
    ax.set_title(f"Particle Position_D={D_n:.6f}_Trial_{i}")
    plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/Position_Images/Trial_{i}.png")
    plt.close()

    np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/tamsad/Trial_{i}.txt", tamsad)
    np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/tamsd/Trial_{i}.txt", tamsd)
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

np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/msad.txt", msad)
np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/msd.txt", msd)
np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/msad_prediction.txt", msad_prediction)
np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/T.txt", T)
                        
plt.title(f"MSD Plot_D={D_n:.6f}")
plt.xlabel("Time")
plt.ylabel("MSD/R")
plt.grid(visible=True)
plt.plot(T, msd)
plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/msd.png")
plt.close()
                        
plt.title(f"MSAD Plot_D={D_n:.6f}")
plt.xlabel("Time")
plt.ylabel("MSAD")
plt.grid(visible=True)
plt.plot(T, msad)
plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/msad.png")
plt.close()
                        
plt.title(f"log-log MSD Plot_D={D_n:.6f}")
plt.xlabel("Time")
plt.ylabel("MSD/R")
plt.grid(visible=True)
plt.plot(T, msd)
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/ll_msd.png")
plt.close()
                        
plt.title(f"log-log MSAD Plot_D={D_n:.6f}")
plt.xlabel("Time")
plt.ylabel("MSAD")
plt.grid(visible=True)
plt.plot(T, msad)
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/ll_msad.png")
plt.close()

llmsad = np.log10(msad[1:])
llmsd = np.log10(msd[1:])

slope_msad = (np.roll(llmsad, shift = -1) - llmsad)/dt_up
slope_msd = (np.roll(llmsd, shift = -1) - llmsd)/dt_up

plt.title(f"Slope of Log=log MSD Plot_D={D_n:.6f}")
plt.xlabel("Time")
plt.ylabel("MSD")
plt.grid(visible=True)
plt.plot(T[1:-1], slope_msd[0:-1])
plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/ll_slope_msd.png")
plt.close()
                        
plt.title(f"Slope of Log-log MSAD Plot_D={D_n:.6f}")
plt.xlabel("Time")
plt.ylabel("MSAD")
plt.grid(visible=True)
plt.plot(T[1:-1], slope_msad[0:-1])
plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/ll_slope_msad.png")
plt.close()
                        
plt.title(f"MSAD Plot_D={D_n:.6f}")
plt.xlabel("Time")
plt.ylabel("MSAD")
plt.grid(visible=True)
plt.plot(T, msad, label = 'simulation')
plt.plot(T, msad_prediction, label = 'first order prediction')
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.legend()
plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/msad_comparison.png")
plt.close()