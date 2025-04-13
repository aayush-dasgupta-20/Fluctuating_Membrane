from icosphere_py.shapes import RegIcos
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import os
import scipy.integrate as integ
import sys
import math

#tB = (int(os.getenv("SLURM_array_TASK_ID")) - 1)/10
D_n = 0.2 + (int(sys.argv[1])+1)*(0.5 - 0.2)/64
R = 1

I = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.double)

#dx = 1E-4
N_timestep = int(1E6)

dt = 2E-2/N_timestep

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
        noise_proj = np.matmul(P,noise)
        #U, S, VT = np.linalg.svd(P)
        #S_diag = np.diag(S)
        #S_T = np.diag(S)
        #for i in range(S_diag.shape[0]):
        #    if S_diag[i,i] != 0:
        #        S_T[i,i] = 1/S_diag[i,i]

        #P_inv = np.matmul(np.matmul(np.transpose(VT),S_T),np.transpose(U))

        acc1, v1 = acceleration_Cartesian(r,r_dot,noise)

        r_1 = np.add(r,v1*dt/2)
        r_dot_1 = np.add(r_dot,acc1*dt/2)

        acc2, v2 = acceleration_Cartesian(r_1,r_dot_1,noise_proj)

        r_2 = np.add(r,v2*dt/2)
        r_dot_2 = np.add(r_dot,acc2*dt/2)

        acc3, v3 = acceleration_Cartesian(r_2,r_dot_2,noise_proj)

        r_3 = np.add(r,v3*dt)
        r_dot_3 = np.add(r_dot,acc3*dt)
        
        acc4, v4 = acceleration_Cartesian(r_3,r_dot_3,noise_proj)

        v_r = np.add(np.add(v1,2*v2),np.add(2*v3,v4))/6
        a_r = np.add(np.add(acc1,2*acc2),np.add(2*acc3,acc4))/6

        return v_r, a_r

def Jacobian_Pseudoinv(p1,p2,p3):

    J = np.array([[np.subtract(p2,p1)[0,0],np.subtract(p3,p1)[0,0]],[np.subtract(p2,p1)[1,0],np.subtract(p3,p1)[1,0]],[np.subtract(p2,p1)[2,0],np.subtract(p3,p1)[2,0]]],dtype=np.double)
    J_T = np.transpose(J)

    J_inv = np.matmul(np.linalg.inv(np.matmul(J_T,J)),J_T)

    return J, J_inv

def q_to_r(q,p1,p2,p3):
     J,J_inv = Jacobian_Pseudoinv(p1,p2,p3)

     r = np.matmul(J,q) + p1

     return r

def r_to_q(r,p1,p2,p3):
    J,J_inv = Jacobian_Pseudoinv(p1,p2,p3)
    q = np.matmul(J_inv, np.subtract(r,p1))
    return q

def q_to_r_der(q_dot,p1,p2,p3):
     J,J_inv = Jacobian_Pseudoinv(p1,p2,p3)

     r = np.matmul(J,q_dot)
     return r

def r_to_q_der(r_dot,p1,p2,p3):
     J,J_inv = Jacobian_Pseudoinv(p1,p2,p3)
     q = np.matmul(J_inv, r_dot)
     return q

icos = RegIcos(R)
poly2 = icos.subdivide()
poly3 = poly2.subdivide()
poly4 = poly3.subdivide()

def find_common(arr1,arr2):
    common = []
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            if arr1[i] == arr2[j]:
                common.append(arr1[i])

    return common

def fold(pe1_index,pe2_index,pf_index):
    pe1_neighbour_list = poly4.vertices.neighbours[pe1_index]
    pe2_neighbour_list = poly4.vertices.neighbours[pe2_index]

    common = find_common(pe1_neighbour_list,pe2_neighbour_list)

    for j in common:
        if j != pf_index:
            return int(j)
    
    raise Exception("Failed Folding. Only One Common Neighbour Found")

def cross(v_1,v_2):
    c_1 = v_1[1,0]*v_2[2,0] - v_1[2,0]*v_2[1,0]
    c_2 = v_1[2,0]*v_2[0,0] - v_1[0,0]*v_2[2,0]
    c_3 = v_1[0,0]*v_2[1,0] - v_1[1,0]*v_2[0,0]
    
    #c_1 = np.double(v_1[1]*v_2[2] - v_1[2]*v_2[1])
    #c_2 = np.double(v_1[2]*v_2[0] - v_1[0]*v_2[2])
    #c_3 = np.double(v_1[0]*v_2[1] - v_1[1]*v_2[0])

    c = np.array([[c_1],[c_2],[c_3]])
    return c

def update_q(q,v_q,q_dot,q_ddot,p1,p2,p3,p1_index,p2_index,p3_index):
    t_rem = dt
    while True:
        t1 = -np.double(q[1, 0]) / np.double(q_dot[1, 0]) if q_dot[1, 0] != 0 else np.inf
        t2 = (1 - np.double(q[0, 0]) - np.double(q[1, 0])) / (np.double(q_dot[0, 0]) + np.double(q_dot[1, 0])) if (q_dot[0, 0] + q_dot[1, 0]) != 0 else np.inf
        t3 = -np.double(q[0, 0]) / np.double(q_dot[0, 0]) if q_dot[0, 0] != 0 else np.inf

        pos_arr = [t1>0,t2>0,t3>0]
        t = [t1,t2,t3]
        pos_vals = []

        for i in range(len(pos_arr)):
            if pos_arr[i]:
                pos_vals.append(t[i])
            
        if all(ti<=0 for ti in t) or all(pos_val>t_rem for pos_val in pos_vals):
            q += q_dot*t_rem
            v_q += q_ddot*t_rem
            return q,v_q,p1,p2,p3,p1_index,p2_index,p3_index
        else:
            J0 = np.array([[np.subtract(p2,p1)[0,0],np.subtract(p3,p1)[0,0]],[np.subtract(p2,p1)[1,0],np.subtract(p3,p1)[1,0]],[np.subtract(p2,p1)[2,0],np.subtract(p3,p1)[2,0]]], dtype = np.double)
            J0_T = np.transpose(J0)

            J0_inv = np.matmul(np.linalg.inv(np.matmul(J0_T,J0)),J0_T)
            p1_old = p1

            norm_0 = cross(p3-p1,p2-p1)

            if min(pos_vals) == t1:
                p3_index = fold(p1_index,p2_index,p3_index)
                p3 = np.array([[poly4.vertices.x[p3_index]],[poly4.vertices.y[p3_index]],[poly4.vertices.z[p3_index]]],dtype=np.double)
                t_min = t1
                norm_up = cross(p2-p1,p3-p1)
                norm_up = norm_up/np.linalg.norm(norm_up)
                v = (p2 - p1)/np.linalg.norm(p2-p1)
                v_cross = np.array([[0,-v[2,0], v[1,0]],[v[2,0], 0, -v[0,0]],[-v[1,0], v[0,0], 0]])
                c = np.dot(np.transpose(norm_0),norm_up)
                term_2 = np.matmul(v_cross, v_cross)/(1 + c)
                R = I + v_cross + term_2
            elif min(pos_vals) == t2:
                p1_index = fold(p2_index,p3_index,p1_index)
                p1 = np.array([[poly4.vertices.x[p1_index]],[poly4.vertices.y[p1_index]],[poly4.vertices.z[p1_index]]],dtype=np.double)
                t_min = t2
                norm_up = cross(p2-p1,p3-p1)
                norm_up = norm_up/np.linalg.norm(norm_up)
                v = (p3 - p2)/np.linalg.norm(p3-p2)
                v_cross = np.array([[0,-v[2,0], v[1,0]],[v[2,0], 0, -v[0,0]],[-v[1,0], v[0,0], 0]])
                c = np.dot(np.transpose(norm_0),norm_up)
                term_2 = np.matmul(v_cross, v_cross)/(1 + c)
                R = I + v_cross + term_2
            elif min(pos_vals) == t3:
                p2_index = fold(p3_index,p1_index,p2_index)
                p2 = np.array([[poly4.vertices.x[p2_index]],[poly4.vertices.y[p2_index]],[poly4.vertices.z[p2_index]]],dtype=np.double)
                t_min = t3
                norm_up = cross(p2-p1,p3-p1)
                norm_up = norm_up/np.linalg.norm(norm_up)
                v = (p1 - p3)/np.linalg.norm(p1-p3)
                v_cross = np.array([[0,-v[2,0], v[1,0]],[v[2,0], 0, -v[0,0]],[-v[1,0], v[0,0], 0]])
                c = np.dot(np.transpose(norm_0),norm_up)
                term_2 = np.matmul(v_cross, v_cross)/(1 + c)
                R = I + v_cross + term_2

            J_u = np.array([[np.subtract(p2,p1)[0,0],np.subtract(p3,p1)[0,0]],[np.subtract(p2,p1)[1,0],np.subtract(p3,p1)[1,0]],[np.subtract(p2,p1)[2,0],np.subtract(p3,p1)[2,0]]])
            Ju_T = np.transpose(J_u)

            Ju_inv = np.matmul(np.linalg.inv(np.matmul(Ju_T,J_u)),Ju_T)

            Rot_mx = np.matmul(Ju_inv,J0)

            q_dot = np.matmul(Ju_inv,np.matmul(np.matmul(R,J0),q_dot))
            q_ddot = np.matmul(Ju_inv,np.matmul(np.matmul(R,J0),q_ddot))
            v_q = np.matmul(Ju_inv,np.matmul(np.matmul(R,J0),v_q))

            p1_change = np.subtract(p1,p1_old)
            rot_p1_change = np.matmul(Ju_inv,p1_change)

            q = np.subtract(np.matmul(Rot_mx,q),rot_p1_change)
            q += q_dot*t_min
            v_q += q_ddot*t_min
            t_rem -= t_min

def acceleration_Local(q,q_dot,noise):
    a_q = -q_dot + noise
    v_q = q_dot

    return a_q,v_q

def RK4_Local(q,q_dot,p1,p2,p3):
     if D_n!=0:
        J, J_inv = Jacobian_Pseudoinv(p1,p2,p3)
        noise_amp = np.sqrt(2*D_n)
        noise = [[noise_amp*rand.randn()],[noise_amp*rand.randn()],[noise_amp*rand.randn()]]
        r = q_to_r(q,p1,p2,p3)
        noise_proj = np.matmul(J,noise)

        aq1, vq1 = acceleration_Local(q,q_dot,noise_proj)

        q_1 = np.add(q,vq1*dt/2)
        q_dot_1 = np.add(q_dot,aq1*dt/2)

        aq2, vq2 = acceleration_Local(q_1,q_dot_1,noise_proj)

        q_2 = np.add(q,vq2*dt/2)
        q_dot_2 = np.add(q_dot,aq2*dt/2)

        aq3, vq3 = acceleration_Local(q_2,q_dot_2,noise_proj)

        q_3 = np.add(q,vq3*dt)
        q_dot_3 = np.add(q_dot,aq3*dt)
        
        aq4, vq4 = acceleration_Local(q_3,q_dot_3,noise_proj)

        v_q = np.add(np.add(vq1,2*vq2),np.add(2*vq3,vq4))/6
        a_q = np.add(np.add(aq1,2*aq2),np.add(2*aq3,aq4))/6

        return v_q, a_q

os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/x", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/y", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/z", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/v_x", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/v_y", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/v_z", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/q1", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/q2", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/v_q1", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/v_q2", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/p1", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/p2", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/p3", exist_ok=True)

j = 0

while j < 100:
    rand.seed(rand.randint(1,9)*(j+1))
    p1 = np.array([[poly4.vertices.x[0]],[poly4.vertices.y[0]],[poly4.vertices.z[0]]],dtype=np.double)
    p1_index = 0
    p2_index = poly4.vertices.neighbours[0][0]
    p1_neighbour_list = poly4.vertices.neighbours[p1_index]
    p2_neighbour_list = poly4.vertices.neighbours[p2_index]
    common = find_common(p1_neighbour_list,p2_neighbour_list)
    p3_index = int(common[0])
    p2 = np.array([[poly4.vertices.x[p2_index]],[poly4.vertices.y[p2_index]],[poly4.vertices.z[p2_index]]],dtype=np.double)
    p3 = np.array([[poly4.vertices.x[p3_index]],[poly4.vertices.y[p3_index]],[poly4.vertices.z[p3_index]]],dtype=np.double)

    q0=np.array([[0.25],[0.25]],dtype=np.double)
    v_q0=np.array([[0],[0]],dtype=np.double)

    p1_arr = []
    p2_arr = []
    p3_arr = []

    r0 = q_to_r(q0,p1,p2,p3)
    v_r0 = q_to_r_der(v_q0,p1,p2,p3)

    q1 = []
    q2 = []
    vq1 = []
    vq2 = []

    x = []
    y = []
    z = []
    vx = []
    vy = []
    vz = []

    q1.append(q0[0,0])
    q2.append(q0[1,0])
    vq1.append(v_q0[0,0])
    vq2.append(v_q0[1,0])

    x.append(r0[0,0])
    y.append(r0[1,0])
    z.append(r0[2,0])
    vx.append(v_r0[0,0])
    vy.append(v_r0[1,0])
    vz.append(v_r0[2,0])

    p1_arr.append(p1_index)
    p2_arr.append(p2_index)
    p3_arr.append(p3_index)

    for i in range(N_timestep):
        q_dot,q_ddot = RK4_Local(q0,v_q0,p1,p2,p3)
        q0,v_q0,p1,p2,p3,p1_index,p2_index,p3_index = update_q(q0,v_q0,q_dot,q_ddot,p1,p2,p3,p1_index,p2_index,p3_index)
        r0 = q_to_r(q0,p1,p2,p3)
        v_r0 = q_to_r_der(v_q0,p1,p2,p3)

        if i%1000==0:
            q1.append(q0[0,0])
            q2.append(q0[1,0])
            vq1.append(v_q0[0,0])
            vq2.append(v_q0[1,0])

            x.append(r0[0,0])
            y.append(r0[1,0])
            z.append(r0[2,0])
            vx.append(v_r0[0,0])
            vy.append(v_r0[1,0])
            vz.append(v_r0[2,0])

            p1_arr.append(p1_index)
            p2_arr.append(p2_index)
            p3_arr.append(p3_index)
    
    check_nan = not(np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isnan(z)) or np.any(np.isnan(vx)) or np.any(np.isnan(vy)) or np.any(np.isnan(vz),axis=0))

    if check_nan:
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/x/Trial_{j}.txt", x)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/y/Trial_{j}.txt", y)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/z/Trial_{j}.txt", z)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/v_x/Trial_{j}.txt", vx)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/v_y/Trial_{j}.txt", vy)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/v_z/Trial_{j}.txt", vz)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/q1/Trial_{j}.txt", q1)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/q2/Trial_{j}.txt", q2)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/v_q1/Trial_{j}.txt", vq1)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/v_q2/Trial_{j}.txt", vq2)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/p1/Trial_{j}.txt", p1_arr)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/p2/Trial_{j}.txt", p2_arr)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_Icosahedron/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/p3/Trial_{j}.txt", p3_arr)
        j += 1      