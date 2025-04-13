from icosphere_py.shapes import RegIcos
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import os
import scipy.integrate as integ
import sys
import math
import pandas as pd

#tB = (int(os.getenv("SLURM_array_TASK_ID")) - 1)/10
D_n = 0.2 + (int(sys.argv[1])+1)*(0.5 - 0.2)/64
R = 1

I = [[1,0,0],[0,1,0],[0,0,1]]

#dx = 1E-4
N_timestep = int(1E3)

dt = 1/N_timestep

def acceleration(theta, phi, w_theta, w_phi, rp):

    a_theta = np.sin(theta)*np.cos(theta)*w_phi**2 - w_theta
    angvel_theta = w_theta
    angvel_phi = w_phi

    kappa = 1 + (2*w_theta*np.cos(theta)/np.sin(theta))
    a_phi = -kappa*w_phi + rp/np.sin(theta)

    return angvel_theta, a_theta, angvel_phi, a_phi

def RK4_update(theta, phi, w_theta, w_phi):
    if D_n != 0:
            noise_amp = np.sqrt(2*D_n)
            r_t = np.random.randn()*noise_amp
            r_p = np.random.randn()*noise_amp
            
            ang_theta_1, a_theta_1, ang_phi_1, a_phi_1 = acceleration(theta, phi, w_theta, w_phi, r_p)
            theta_1 = theta + ang_theta_1*dt/2
            w_theta_1 = w_theta + a_theta_1*dt/2
            phi_1 = theta + ang_phi_1*dt/2
            w_phi_1 = w_phi + a_phi_1*dt/2

            ang_theta_2, a_theta_2, ang_phi_2, a_phi_2 = acceleration(theta_1, phi_1, w_theta_1, w_phi_1, r_p)
            theta_2 = theta + ang_theta_2*dt/2
            w_theta_2 = w_theta + a_theta_2*dt/2
            phi_2 = theta + ang_phi_2*dt/2
            w_phi_2 = w_phi + a_phi_2*dt/2

            ang_theta_3, a_theta_3, ang_phi_3, a_phi_3 = acceleration(theta_2, phi_2, w_theta_2, w_phi_2, r_p)
            theta_3 = theta + ang_theta_3*dt
            w_theta_3 = w_theta + a_theta_3*dt
            phi_3 = theta + ang_phi_3*dt
            w_phi_3 = w_phi + a_phi_3*dt

            ang_theta_4, a_theta_4, ang_phi_4, a_phi_4 = acceleration(theta_3, phi_3, w_theta_3, w_phi_3, r_p)

            d_theta = (ang_theta_1 + 2*ang_theta_2 + 2*ang_theta_3 + ang_theta_4)*dt/6
            d_w_theta = r_t + (a_theta_1 + 2*a_theta_2 + 2*a_theta_3 + a_theta_4)*dt/6
            d_phi = (ang_phi_1 + 2*ang_phi_2 + 2*ang_phi_3 + ang_phi_4)*dt/6
            d_w_phi = (a_phi_1 + 2*a_phi_2 + 2*a_phi_3 + a_phi_4)*dt/6

            e_theta = [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)]
            e_phi = [-np.sin(phi), np.cos(phi), 0]

            dr = d_theta*e_theta + np.sin(theta)*d_phi*e_phi

            return d_theta, d_w_theta, d_phi, d_w_phi
    
    else:
        raise ValueError('Dn cannot be 0')

def acceleration_Cartesian(r,r_dot,noise):
    a_r = -r_dot + noise
    v_r = r_dot

    return a_r,v_r

def RK4_Cartesian(r,r_dot):
     if D_n!=0:
        noise_amp = np.sqrt(2*D_n)
        noise = [[noise_amp*rand.randn()],[noise_amp*rand.randn()],[noise_amp*rand.randn()]]
        norm = r/np.linalg.norm(r)
        norm_Transpose = np.transpose(norm)
        P = I - np.matmul(norm,norm_Transpose)

        acc1, v1 = acceleration_Cartesian(r,r_dot,noise)

        r_1 = np.add(r,v1*dt/2)
        r_dot_1 = np.add(r_dot,acc1*dt/2)

        acc2, v2 = acceleration_Cartesian(r_1,r_dot_1,noise)

        r_2 = np.add(r,v2*dt/2)
        r_dot_2 = np.add(r_dot,acc2*dt/2)

        acc3, v3 = acceleration_Cartesian(r_2,r_dot_2,noise)

        r_3 = np.add(r,v3*dt)
        r_dot_3 = np.add(r_dot,acc3*dt)
        
        acc4, v4 = acceleration_Cartesian(r_3,r_dot_3,noise)

        v_r = np.matmul(P,np.add(np.add(v1,2*v2),np.add(2*v3,v4))/6)
        a_r = np.matmul(P,np.add(np.add(acc1,2*acc2),np.add(2*acc3,acc4))/6)

        return v_r, a_r

poly = RegIcos(R)
for i in range(6):
    poly=poly.subdivide()

poly.vertices.to_hdf("./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly_6.h5", key='poly', mode='w')

def vert_dist(vert,r):
    r_v = np.array([[poly.vertices.x[vert]],[poly.vertices.y[vert]],[poly.vertices.z[vert]]],dtype=float)
    dr = r_v - r

    dist = np.sqrt(np.sum(np.square(dr)))

    return dist

def near_vertex(vert_init, r):
    vert_near = vert_init
    min_dist = vert_dist(vert_init,r)
    while True:
        no_update = True
        neighbour_list = poly.vertices.neighbours[vert_near]
        for i in range(len(neighbour_list)):
            if vert_dist(neighbour_list[i],r)<min_dist:
                no_update = False
                min_dist = vert_dist(neighbour_list[i],r)
                vert_near = neighbour_list[i]
        
        if no_update:
            return vert_near


os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/large_poly_6/N={N_timestep:.2e}/Dn={D_n:.6e}", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/large_poly_6/N={N_timestep:.2e}/Dn={D_n:.6e}/x", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/large_poly_6/N={N_timestep:.2e}/Dn={D_n:.6e}/y", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/large_poly_6/N={N_timestep:.2e}/Dn={D_n:.6e}/z", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/large_poly_6/N={N_timestep:.2e}/Dn={D_n:.6e}/v_x", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/large_poly_6/N={N_timestep:.2e}/Dn={D_n:.6e}/v_y", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/large_poly_6/N={N_timestep:.2e}/Dn={D_n:.6e}/v_z", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/large_poly_6/N={N_timestep:.2e}/Dn={D_n:.6e}/v_z", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/large_poly_6/N={N_timestep:.2e}/Dn={D_n:.6e}/vertex", exist_ok=True)

j = 0

while j < 100:
    j += 1
    r = np.array([[poly.vertices.x[0]],[poly.vertices.y[0]],[poly.vertices.z[0]]],dtype=float)
    v_r = np.array([[0],[0],[0]],dtype=float)

    vert_index = 0

    x = [r[0,0]]
    y = [r[1,0]]
    z = [r[2,0]]
    vx = [v_r[0,0]]
    vy = [v_r[1,0]]
    vz = [v_r[2,0]]

    vert_list = []
    vert_list.append(vert_index)

    for i in range(N_timestep):
        r_dot, r_ddot = RK4_Cartesian(r,v_r)
        r_f = r + r_dot*dt
        v_r += r_ddot*dt

        vert_index = near_vertex(vert_index, r_f)

        r = np.array([[poly.vertices.x[vert_index]],[poly.vertices.y[vert_index]],[poly.vertices.z[vert_index]]],dtype=float)
        x.append(r[0,0])
        y.append(r[1,0])
        z.append(r[2,0])
        vx.append(v_r[0,0])
        vy.append(v_r[1,0])
        vz.append(v_r[2,0])
        vert_list.append(vert_index)

    np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/large_poly_6/N={N_timestep:.2e}/Dn={D_n:.6e}/x/Trial_{j}.txt",x)
    np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/large_poly_6/N={N_timestep:.2e}/Dn={D_n:.6e}/y/Trial_{j}.txt",y)
    np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/large_poly_6/N={N_timestep:.2e}/Dn={D_n:.6e}/z/Trial_{j}.txt",z)
    np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/large_poly_6/N={N_timestep:.2e}/Dn={D_n:.6e}/v_x/Trial_{j}.txt",vx)
    np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/large_poly_6/N={N_timestep:.2e}/Dn={D_n:.6e}/v_y/Trial_{j}.txt",vy)
    np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/large_poly_6/N={N_timestep:.2e}/Dn={D_n:.6e}/v_z/Trial_{j}.txt",vz)
    np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/large_poly_6/N={N_timestep:.2e}/Dn={D_n:.6e}/vertex/Trial_{j}.txt", vert_list)

