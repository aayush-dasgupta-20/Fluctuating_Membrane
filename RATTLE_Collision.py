import numpy as np # type: ignore
import numpy.random as rand # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
import scipy.integrate as integ # type: ignore
import sys
import math
from numba import njit

#tB = (int(os.getenv("SLURM_array_TASK_ID")) - 1)/10
#D_n = 0.2 + (int(sys.argv[1])+1)*(0.5 - 0.2)/64
N_theta = 64
D_n = 2
theta_f = np.pi/5
theta_i = np.pi/10
theta = theta_i + (theta_f - theta_i)*(int(sys.argv[1])+1)/(N_theta)

R = 1

I = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.double)

dt = 1E-3

def cross(v_1,v_2):
    c_1 = v_1[1,0]*v_2[2,0] - v_1[2,0]*v_2[1,0]
    c_2 = v_1[2,0]*v_2[0,0] - v_1[0,0]*v_2[2,0]
    c_3 = v_1[0,0]*v_2[1,0] - v_1[1,0]*v_2[0,0]
    
    #c_1 = np.double(v_1[1]*v_2[2] - v_1[2]*v_2[1])
    #c_2 = np.double(v_1[2]*v_2[0] - v_1[0]*v_2[2])
    #c_3 = np.double(v_1[0]*v_2[1] - v_1[1]*v_2[0])

    c = np.array([[c_1],[c_2],[c_3]])
    return c

def Constraint(x,y,z):
    res = x**2 + y**2 + z**2 - 1
    return res

#@njit
def Rattle_update(r,p):
    tolerance=1E-6
    
    noise_amp = np.sqrt(2*D_n)
    noise_x = rand.normal(0,noise_amp)
    noise_y = rand.normal(0,noise_amp)
    noise_z = rand.normal(0,noise_amp)
    noise_x1 = rand.normal(0,noise_amp)
    noise_y1 = rand.normal(0,noise_amp)
    noise_z1 = rand.normal(0,noise_amp)

    lagrange_mult_x = 0
    lagrange_mult_p = 0

    x0 = r[0,0]
    y0 = r[1,0]
    z0 = r[2,0]
    
    p_x0 = p[0,0]
    p_y0 = p[1,0]
    p_z0 = p[2,0]
    
    x1 = x0
    y1 = y0
    z1 = z0

    normal_0 = np.array([[x0],[y0],[z0]],dtype=np.double)
    i = 0

    while True:
        normal_1 = np.array([[x1],[y1],[z1]],dtype=np.double)

        p_x05 = p_x0 + ((noise_x - lagrange_mult_x*normal_0[0,0])*dt)/2
        p_y05 = p_y0 + ((noise_y - lagrange_mult_x*normal_0[1,0])*dt)/2
        p_z05 = p_z0 + ((noise_z - lagrange_mult_x*normal_0[2,0])*dt)/2

        res_x = np.array([[x0 - x1 + p_x05*dt],[y0 - y1 + p_y05*dt],[z0 - z1 + p_z05*dt],[Constraint(x1,y1,z1)]],dtype=np.double)
        J_x = np.array([[-1,0,0,-(dt**2)*normal_0[0,0]/2],[0,-1,0,-(dt**2)*normal_0[1,0]/2],[0,0,-1,-(dt**2)*normal_0[2,0]/2],[normal_1[0,0],normal_1[1,0],normal_1[2,0],0]],dtype=np.double)
        d_x = np.matmul(np.linalg.inv(J_x),res_x)
        x1 = x1 - d_x[0,0]
        y1 = y1 - d_x[1,0]
        z1 = z1 - d_x[2,0]
        lagrange_mult_x = lagrange_mult_x - d_x[3,0]
        i += 1

        if np.linalg.norm(res_x)<tolerance or i>1E4:
            break
    
    p_x1 = p_x05 + noise_x1*dt/2
    p_y1 = p_y05 + noise_y1*dt/2
    p_z1 = p_z05 + noise_z1*dt/2

    j = 0

    while True:
        p_x_dot = noise_x1 - lagrange_mult_p*normal_1[0,0] - p_x1
        p_y_dot = noise_y1 - lagrange_mult_p*normal_1[1,0] - p_y1
        p_z_dot = noise_z1 - lagrange_mult_p*normal_1[2,0] - p_z1

        res_p = np.array([[p_x05 - p_x1 + p_x_dot*dt/2],[p_y05 - p_y1 + p_y_dot*dt/2],[p_z05 - p_z1 + p_z_dot*dt/2],[normal_1[0,0]*p_x1 + normal_1[1,0]*p_y1 + normal_1[2,0]*p_z1]],dtype=np.double)
        J_p = np.array([[-1,0,0,-dt*normal_1[0,0]/2],[0,-1,0,-dt*normal_1[1,0]/2],[0,0,-1,-dt*normal_1[2,0]/2],[normal_1[0,0],normal_1[1,0],normal_1[2,0],0]],dtype=np.double)
        d_p = np.matmul(np.linalg.inv(J_p),res_p)
        p_x1 = p_x1 - d_p[0,0]
        p_y1 = p_y1 - d_p[1,0]
        p_z1 = p_z1 - d_p[2,0]
        lagrange_mult_p = lagrange_mult_p - d_p[3,0]

        j += 1

        if np.linalg.norm(res_p)<tolerance or j>1E4:
            break
    
    r1 = np.array([[x1],[y1],[z1]],dtype=np.double)
    v1 = np.array([[p_x1],[p_y1],[p_z1]],dtype=np.double)

    return r1,v1


def geo_dist(r1,r2,j):
    c = np.linalg.norm(r1-r2)
    if (j == 0 or j==1) and theta == theta_i + (theta_f - theta_i)/N_theta:
        p_stat = (theta,c)
        print(f"Trial {j}: {p_stat}")
    return c

#N_theta = 20
#theta_init = np.linspace(1/N_theta,1,N_theta)

col_dist = 5E-2

os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/dt={dt:.2e}/Dn={D_n:.6e}/init_geo_dist={theta:.2e}/x2", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/dt={dt:.2e}/Dn={D_n:.6e}/init_geo_dist={theta:.2e}/y2", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/dt={dt:.2e}/Dn={D_n:.6e}/init_geo_dist={theta:.2e}/z2", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/dt={dt:.2e}/Dn={D_n:.6e}/init_geo_dist={theta:.2e}/v_x2", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/dt={dt:.2e}/Dn={D_n:.6e}/init_geo_dist={theta:.2e}/v_y2", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/dt={dt:.2e}/Dn={D_n:.6e}/init_geo_dist={theta:.2e}/v_z2", exist_ok=True)
t_col = []
t_inst_col = 0
            
r10=np.array([[1.0],[0.0],[0.0]],dtype=np.double)

r20=np.array([[np.cos(theta)],[np.sin(theta)],[0.0]],dtype=np.double)
v_r20=np.array([[0.0],[0.0],[0.0]],dtype=np.double)

x2 = []
y2 = []
z2 = []
vx2 = []
vy2 = []
vz2 = []

x2.append(r20[0,0])
y2.append(r20[1,0])
z2.append(r10[2,0])
vx2.append(v_r20[0,0])
vy2.append(v_r20[1,0])
vz2.append(v_r20[2,0])

for j in range(100):
    rand.seed(j)
    i=0
    while geo_dist(r10,r20,j)>col_dist:
        r20,v_r20 = Rattle_update(r20,v_r20)
        t_inst_col += dt
        x2.append(r20[0,0])
        y2.append(r20[1,0])
        z2.append(r20[2,0])
        vx2.append(v_r20[0,0])
        vy2.append(v_r20[1,0])
        vz2.append(v_r20[2,0])
        i+=1
    
    check_nan2 = not(np.any(np.isnan(x2)) or np.any(np.isnan(y2)) or np.any(np.isnan(z2)) or np.any(np.isnan(vx2)) or np.any(np.isnan(vy2)) or np.any(np.isnan(vz2)))

    if check_nan2:
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/dt={dt:.2e}/Dn={D_n:.6e}/init_geo_dist={theta:.2e}/x2/Trial_{j}.txt", x2)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/dt={dt:.2e}/Dn={D_n:.6e}/init_geo_dist={theta:.2e}/y2/Trial_{j}.txt", y2)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/dt={dt:.2e}/Dn={D_n:.6e}/init_geo_dist={theta:.2e}/z2/Trial_{j}.txt", z2)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/dt={dt:.2e}/Dn={D_n:.6e}/init_geo_dist={theta:.2e}/v_x2/Trial_{j}.txt", vx2)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/dt={dt:.2e}/Dn={D_n:.6e}/init_geo_dist={theta:.2e}/v_y2/Trial_{j}.txt", vy2)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/dt={dt:.2e}/Dn={D_n:.6e}/init_geo_dist={theta:.2e}/v_z2/Trial_{j}.txt", vz2)
        j+=1
    
    f = open(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE_Collision_/dt={dt:.2e}/Dn={D_n:.6e}/init_geo_dist={theta:.2e}/t_col.txt", "a")
    f.write(f"{t_inst_col:.2f}\n")
    f.close()