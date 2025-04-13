import numpy as np # type: ignore
import numpy.random as rand # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
import sys
from numba import njit # type: ignore

dt = 1E-2
#D_n = 0.2 + (int(sys.argv[1])+1)*(0.5 - 0.2)/64
D_n = 2

N_theta = 16
theta_init = np.linspace(1,N_theta,N_theta)
theta_f = np.pi/5
theta_i = np.pi/10
theta_init = (theta_f - theta_i)*(theta_init)/N_theta + theta_i
t_col_avg = []

for theta in theta_init:
    t_col = np.loadtxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/dt={dt:.2e}/Dn={D_n:.6e}/init_geo_dist={theta:.2e}/t_col.txt")
    t_col_avg.append(np.average(t_col))

os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/Plots/dt={dt:.2e}/Dn={D_n:.6e}",exist_ok=True)

plt.title(f"Collision Time vs. Theta Plot_D={D_n:.6f}")
plt.xlabel("Initial Geodesic Distance")
plt.ylabel("Average Collision Time")
plt.grid(visible=True)
plt.plot(theta_init, t_col_avg)
plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/Plots/dt={dt:.2e}/Dn={D_n:.6e}/t_col_theta.png")
plt.close()

plt.title(f"Collision Time vs. Theta Plot_D={D_n:.6f}")
plt.xlabel("Initial Geodesic Dsitance")
plt.ylabel("Average Collision Time")
plt.grid(visible=True)
plt.plot(theta_init, t_col_avg)
plt.xscale("log")
plt.yscale("log")
plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/Plots/dt={dt:.2e}/Dn={D_n:.6e}/log_log_t_col_theta.png")
plt.close()

np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/Plots/dt={dt:.2e}/Dn={D_n:.6e}/avg_t_collision.txt", t_col_avg)
np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/Plots/dt={dt:.2e}/Dn={D_n:.6e}/theta.txt", theta_init)