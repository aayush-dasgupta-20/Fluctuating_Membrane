import numpy as np # type: ignore
import numpy.random as rand # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
import scipy.integrate as integ # type: ignore
import sys
import math

#tB = (int(os.getenv("SLURM_array_TASK_ID")) - 1)/10
D_n = 0.2 + (int(sys.argv[1])+1)*(0.5 - 0.2)/64
R = 1

I = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.double)

#dx = 1E-4
N_timestep = int(1E4)

dt = 2E-2/N_timestep

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

def Rattle_update(r,p):
    tolerance=1E-6
    
    noise_amp = np.sqrt(2*D_n)
    noise_x = noise_amp*rand.randn()
    noise_y = noise_amp*rand.randn()
    noise_z = noise_amp*rand.randn()

    noise_x1 = noise_amp*rand.randn()
    noise_y1 = noise_amp*rand.randn()
    noise_z1 = noise_amp*rand.randn()

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

    while True:

        normal_1 = np.array([[x1],[y1],[z1]],dtype=np.double)

        p_x05 = p_x0 + ((noise_x - lagrange_mult_x*normal_0[0,0] - p_x0)*dt)/2
        p_y05 = p_y0 + ((noise_y - lagrange_mult_x*normal_0[1,0] - p_y0)*dt)/2
        p_z05 = p_z0 + ((noise_z - lagrange_mult_x*normal_0[2,0] - p_z0)*dt)/2

        res_x = np.array([[x0 - x1 + p_x05*dt],[y0 - y1 + p_y05*dt],[z0 - z1 + p_z05*dt],[Constraint(x1,y1,z1)]],dtype=np.double)
        J_x = np.array([[-1,0,0,-(dt**2)*normal_0[0,0]/2],[0,-1,0,-(dt**2)*normal_0[1,0]/2],[0,0,-1,-(dt**2)*normal_0[2,0]/2],[normal_1[0,0],normal_1[1,0],normal_1[2,0],0]],dtype=np.double)
        d_x = np.matmul(np.linalg.inv(J_x),res_x)
        x1 = x1 - d_x[0,0]
        y1 = y1 - d_x[1,0]
        z1 = z1 - d_x[2,0]
        lagrange_mult_x = lagrange_mult_x - d_x[3,0]

        if np.linalg.norm(res_x)<tolerance:
            break
    
    p_x1 = p_x05 + noise_x1*dt/2
    p_y1 = p_y05 + noise_y1*dt/2
    p_z1 = p_z05 + noise_z1*dt/2

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

        if np.linalg.norm(res_p)<tolerance:
            break
    
    r1 = np.array([[x1],[y1],[z1]],dtype=np.double)
    v1 = np.array([[p_x1],[p_y1],[p_z1]],dtype=np.double)

    return r1,v1

os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/x", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/y", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/z", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/v_x", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/v_y", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/v_z", exist_ok=True)

j = 0

while j < 100:
    rand.seed(rand.randint(1,9)*(j+1))
    
    r0=np.array([[1.0],[0.0],[0.0]],dtype=np.double)
    v_r0=np.array([[0.0],[0.0],[0.0]],dtype=np.double)

    x = []
    y = []
    z = []
    vx = []
    vy = []
    vz = []

    x.append(r0[0,0])
    y.append(r0[1,0])
    z.append(r0[2,0])
    vx.append(v_r0[0,0])
    vy.append(v_r0[1,0])
    vz.append(v_r0[2,0])

    for i in range(N_timestep):
        r0,v_r0 = Rattle_update(r0,v_r0)

        if i%10==0:
            x.append(r0[0,0])
            y.append(r0[1,0])
            z.append(r0[2,0])
            vx.append(v_r0[0,0])
            vy.append(v_r0[1,0])
            vz.append(v_r0[2,0])
    
    check_nan = not(np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isnan(z)) or np.any(np.isnan(vx)) or np.any(np.isnan(vy)) or np.any(np.isnan(vz),axis=0))

    if check_nan:
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/x/Trial_{j}.txt", x)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/y/Trial_{j}.txt", y)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/z/Trial_{j}.txt", z)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/v_x/Trial_{j}.txt", vx)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/v_y/Trial_{j}.txt", vy)
        np.savetxt(f"./codes/Single_Particle_Trace/less_than_tB_RK4_RATTLE/poly4/N={N_timestep:.2e}/Dn={D_n:.6e}/v_z/Trial_{j}.txt", vz)
        j += 1