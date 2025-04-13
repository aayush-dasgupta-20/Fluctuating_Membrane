import numpy as np # type: ignore
import numpy.random as rand # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
import sys
from numba import njit # type: ignore

arg = np.linspace(0,63,64)

dt = 1E-3
D_n = 0.2 + (arg+1)*(0.5 - 0.2)/64

t_col1 = np.loadtxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/Plots/dt={dt:.2e}/Dn={D_n[0]:.6e}/avg_t_collision.txt")
theta_init = np.loadtxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/Plots/dt={dt:.2e}/Dn={D_n[0]:.6e}/theta.txt")

l1 = len(t_col1)
l2 = len(D_n)

t_col_arr = np.zeros((l2,l1),dtype=float)

for i in range(l2):
    t_col_avg = np.loadtxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/Plots/dt={dt:.2e}/Dn={D_n[i]:.6e}/avg_t_collision.txt")
    for j in range(l1):
        t_col_arr[i,j] = t_col_avg[j]

D,Theta = np.meshgrid(D_n,theta_init,indexing='ij')

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_surface(Theta, D, t_col_arr, cmap='viridis')
ax.set_title("Collision Time vs. Theta Plot")
ax.set_xlabel("Initial Geodesic Distance")
ax.set_ylabel("Diffusion Constant")
ax.set_zlabel("Average Collision Time")
plt.savefig(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/Plots/dt={dt:.2e}/3D_t_col_theta_D.png")