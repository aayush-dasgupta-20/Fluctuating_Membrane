import os
import numpy as np
import matplotlib.pyplot as plt
import sys

D_n = 0.2 + (int(sys.argv[1])+1)*(0.5 - 0.2)/64
#R = 1

#dx = 1E-4
N_timestep = 1E3
R = 1

dt = 1/N_timestep

theta_dir = f"./codes/Single_Particle_Trace/less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/theta"
phi_dir = f"./codes/Single_Particle_Trace/less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/phi"

files_theta = [file for file in os.listdir(theta_dir)]
files_phi = [file for file in os.listdir(phi_dir)]

N = len(files_theta)
l = len(np.loadtxt(os.path.join(theta_dir, files_theta[0])))

for i in range(N):
    if i%20==0:
        os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Images_for_Video/N={N_timestep:.2e}/Dn={D_n:.6f}/Trial_{i}", exist_ok=True)
        theta = np.loadtxt(os.path.join(theta_dir, files_theta[i]))
        phi = np.loadtxt(os.path.join(phi_dir, files_phi[i]))

        X = R*np.sin(theta)*np.cos(phi)
        Y = R*np.sin(theta)*np.sin(phi)
        Z = R*np.cos(theta)

        for j in range(l):
            if j%10 == 0:
                plt.figure(figsize=[10,10])
                ax = plt.axes(projection='3d')
                ax.grid()
                ax.set_title(f"Particle Position_D={D_n:.6f}_Trial_{i}_t={j*dt:.2e}")
                if j == 0:
                    plt.plot(X[j],Y[j], Z[j], marker='o',markersize=8,color='red')
                elif j < 6:
                    plt.plot(X[j],Y[j], Z[j], marker='o',markersize=8,color='red')
                    plt.plot(X[0:j],Y[0:j], Z[0:j],linewidth=0.5,color='green',alpha=0.6)
                else:
                    plt.plot(X[j],Y[j], Z[j], marker='o',markersize=8,color='red')
                    plt.plot(X[0:j],Y[0:j], Z[0:j],linewidth=0.5,color='green',alpha=0.6)
                    plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Images_for_Video/N={N_timestep:.2e}/Dn={D_n:.6f}/Trial_{i}/t={j*dt:.2f}.png")
                    plt.close()
