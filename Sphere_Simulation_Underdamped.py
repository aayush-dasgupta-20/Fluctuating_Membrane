import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import os
import scipy.integrate as integ
import sys
import math

#tB = (int(os.getenv("SLURM_ARRAY_TASK_ID")) - 1)/10
D_n = 0.2 + (int(sys.argv[1])+1)*(0.5 - 0.2)/64
#R = 1

#dx = 1E-4
N_timestep = 1E3

dt = 1E-3/N_timestep

"""def leapfrog_update(theta, phi, w_theta, w_phi):
    if tB != 0:
        noise_amp = tB*np.sqrt(2*D_d)/R
        r_t = (np.random.randn()*np.sin(theta) + np.random.randn()*np.cos(theta)*np.cos(phi) + np.random.randn()*np.cos(theta)*np.sin(phi))*noise_amp
        r_p = (np.random.randn()*np.sin(phi) + np.random.randn()*np.cos(phi))*noise_amp

        dtp = dt/tB
        if np.sin(theta) != 0:
            noise = [r_t/R, r_p/(R*np.sin(theta))]
            kappa = (1/tB) + (2*w_theta/np.tan(theta))
            a_theta = noise[0] + np.sin(theta)*np.cos(theta)*w_phi**2 - (w_theta/tB)
            a_phi = noise[1] - kappa*w_phi

            w_theta_f = w_theta + a_theta*dt
            theta_f = theta + w_theta_f*dt

            w_phi_f = w_phi + a_phi*dt
            phi_f = phi + w_phi_f*dt
        else:
            noise_0 = r_t/R
            a_theta = noise_0 + np.sin(theta)*np.cos(theta)*w_phi**2 - (w_theta/tB)

            w_theta_f = w_theta + a_theta*dt
            theta_f = theta + w_theta_f*dt

            if np.sin(theta_f) != 0:
                noise_1 = r_p/(R*np.sin(theta_f))
                kappa = (1/tB) + (2*w_theta/np.tan(theta_f))
                a_phi = noise_1 - kappa*w_phi
                
                w_phi_f = w_phi + a_phi*dt
                phi_f = phi + w_phi_f*dt
            else:
                w_phi_f = w_phi
                phi_f = phi
                
        theta_ret_1 = (theta_f%(2*np.pi)) - np.pi

        if theta_ret_1<0:
            phi_f += np.pi
        
        theta_ret = np.abs(theta_ret_1)
        phi_ret = (phi%(2*np.pi)) - np.pi

        return theta_ret, phi_ret, w_theta_f, w_phi_f
    else:
        noise_amp = np.sqrt(2*D_d)
        r_t = np.rand.randn()*noise_amp
        r_p = np.rand.randn()*noise_amp

        if np.sin(theta) != 0:

            noise = [r_t/R, r_p/(R*np.sin(theta))]
            theta_f = theta + noise[0]*dt
            phi_f = phi + noise[1]*dt

            w_theta_f = noise[0]
            w_phi_f = noise[1]

        else:
            noise_0 = r_t/R
            theta_f = theta + noise_0*dt
            w_theta_f = noise_0

            if np.sin(theta_f) != 0:
                noise_1 = r_p/(R*np.sin(theta_f))
                phi_f = phi + noise_1*dt
                w_phi_f = noise_1
            else:
                phi_f = phi
                w_phi_f = 0
        
        theta_ret_1 = (theta_f%(2*np.pi)) - np.pi

        if theta_ret_1<0:
            phi_f += np.pi
        
        theta_ret = np.abs(theta_ret_1)
        phi_ret = (phi%(2*np.pi)) - np.pi

        return theta_ret, phi_ret, w_theta_f, w_phi_f"""
    
"""def Euler_update(theta, phi, w_theta, w_phi):
    if D_n != 0:
        noise_amp = np.sqrt(2*D_n)
        r_t = np.random.randn()*noise_amp
        r_p = np.random.randn()*noise_amp

        if np.sin(theta) != 0:
            noise = [r_t, r_p/np.sin(theta)]
            kappa = 1 + (2*w_theta/np.tan(theta))
            a_theta = noise[0] + np.sin(theta)*np.cos(theta)*w_phi**2 - w_theta
            a_phi = noise[1] - kappa*w_phi

            w_theta_f = w_theta + a_theta*dt
            theta_f = theta + w_theta*dt

            w_phi_f = w_phi + a_phi*dt
            phi_f = phi + w_phi*dt
        else:
            noise_0 = r_t
            a_theta = noise_0 + np.sin(theta)*np.cos(theta)*w_phi**2 - w_theta

            w_theta_f = w_theta + a_theta*dt
            theta_f = theta + w_theta*dt

            if np.sin(theta_f) != 0:
                noise_1 = r_p/np.sin(theta_f)
                kappa = 1 + (2*w_theta/np.tan(theta_f))
                a_phi = noise_1 - kappa*w_phi
                
                w_phi_f = w_phi + a_phi*dt
                phi_f = phi + w_phi*dt
            else:
                w_phi_f = w_phi
                phi_f = phi
                
        theta_ret_1 = (theta_f%(2*np.pi)) - np.pi

        if theta_ret_1<0:
            phi_f += np.pi
        
        theta_ret = np.abs(theta_ret_1)
        phi_ret = (phi%(2*np.pi)) - np.pi

        return theta_ret, phi_ret, w_theta_f, w_phi_f
    else:
        noise_amp = np.sqrt(2*D_n)
        r_t = np.rand.randn()*noise_amp
        r_p = np.rand.randn()*noise_amp

        if np.sin(theta) != 0:

            noise = [r_t, r_p/(np.sin(theta))]
            theta_f = theta + noise[0]*dt
            phi_f = phi + noise[1]*dt

            w_theta_f = noise[0]
            w_phi_f = noise[1]

        else:
            noise_0 = r_t
            theta_f = theta + noise_0*dt
            w_theta_f = noise_0

            if np.sin(theta_f) != 0:
                noise_1 = r_p/(np.sin(theta_f))
                phi_f = phi + noise_1*dt
                w_phi_f = noise_1
            else:
                phi_f = phi
                w_phi_f = 0
        
        theta_ret_1 = (theta_f%(2*np.pi)) - np.pi

        if theta_ret_1<0:
            phi_f += np.pi
        
        theta_ret = np.abs(theta_ret_1)
        phi_ret = (phi%(2*np.pi)) - np.pi

        return theta_ret, phi_ret, w_theta_f, w_phi_f"""

def acceleration(theta, phi, w_theta, w_phi):

    a_theta = np.sin(theta)*np.cos(theta)*w_phi**2 - w_theta
    angvel_theta = w_theta
    angvel_phi = w_phi

    if np.abs(np.sin(theta)) >= 1E-6:
        kappa = 1 + (2*w_theta*np.cos(theta)/np.sin(theta))
        a_phi = -kappa*w_phi
    else:
        theta_f = theta + angvel_theta*dt
        if np.abs(np.sin(theta_f)) >= 1E-6:
            kappa = 1 + (2*w_theta*np.cos(theta_f)/np.sin(theta_f))
            a_phi = -kappa*w_phi
        else:
            a_phi = 0

    return angvel_theta, a_theta, angvel_phi, a_phi

def RK4_update(theta, phi, w_theta, w_phi):
    if D_n != 0:
        
        noise_amp = np.sqrt(2*D_n)
        r_t = np.random.randn()*noise_amp
        r_p = np.random.randn()*noise_amp
        
        ang_theta_1, a_theta_1, ang_phi_1, a_phi_1 = acceleration(theta, phi, w_theta, w_phi)
        theta_1 = theta + ang_theta_1*dt/2
        w_theta_1 = w_theta + a_theta_1*dt/2
        phi_1 = theta + ang_phi_1*dt/2
        w_phi_1 = w_phi + a_phi_1*dt/2

        ang_theta_2, a_theta_2, ang_phi_2, a_phi_2 = acceleration(theta_1, phi_1, w_theta_1, w_phi_1)
        theta_2 = theta + ang_theta_2*dt/2
        w_theta_2 = w_theta + a_theta_2*dt/2
        phi_2 = theta + ang_phi_2*dt/2
        w_phi_2 = w_phi + a_phi_2*dt/2

        ang_theta_3, a_theta_3, ang_phi_3, a_phi_3 = acceleration(theta_2, phi_2, w_theta_2, w_phi_2)
        theta_3 = theta + ang_theta_3*dt
        w_theta_3 = w_theta + a_theta_3*dt
        phi_3 = theta + ang_phi_3*dt
        w_phi_3 = w_phi + a_phi_3*dt

        ang_theta_4, a_theta_4, ang_phi_4, a_phi_4 = acceleration(theta_3, phi_3, w_theta_3, w_phi_3)

        d_theta = (ang_theta_1 + 2*ang_theta_2 + 2*ang_theta_3 + ang_theta_4)*dt/6
        d_w_theta = r_t + (a_theta_1 + 2*a_theta_2 + 2*a_theta_3 + a_theta_4)*dt/6
        d_phi = (ang_phi_1 + 2*ang_phi_2 + 2*ang_phi_3 + ang_phi_4)*dt/6
        if np.abs(np.sin(theta)) >= 1E-6:
            d_w_phi = r_p/(np.sin(theta)) + (a_phi_1 + 2*a_phi_2 + 2*a_phi_3 + a_phi_4)*dt/6
        elif np.abs(np.sin(theta + d_theta)) >= 1E-6:
            d_w_phi = r_p/(np.sin(theta + d_theta)) + (a_phi_1 + 2*a_phi_2 + 2*a_phi_3 + a_phi_4)*dt/6
        else:
            d_w_phi = 0

        theta_f = theta + d_theta
        w_theta_f = w_theta + d_w_theta
        phi_f = phi + d_phi
        w_phi_f = w_phi + d_w_phi

        theta_ret_1 = ((theta_f + np.pi)%(2*np.pi)) - np.pi

        if theta_ret_1<0:
            phi_f += np.pi
        
        theta_ret = np.abs(theta_ret_1)
        phi_ret = ((phi_f)%(2*np.pi))

        return theta_ret, phi_ret, w_theta_f, w_phi_f

    else:
        noise_amp = np.sqrt(2*D_n)
        r_t = np.rand.randn()*noise_amp
        r_p = np.rand.randn()*noise_amp

        if np.sin(theta) != 0:

            noise = [r_t, r_p/(np.sin(theta))]
            theta_f = theta + noise[0]*dt
            phi_f = phi + noise[1]*dt

            w_theta_f = noise[0]
            w_phi_f = noise[1]

        else:
            noise_0 = r_t
            theta_f = theta + noise_0*dt
            w_theta_f = noise_0

            if np.sin(theta_f) != 0:
                noise_1 = r_p/(np.sin(theta_f))
                phi_f = phi + noise_1*dt
                w_phi_f = noise_1
            else:
                phi_f = phi
                w_phi_f = 0
        
        theta_ret_1 = (theta_f%(2*np.pi)) - np.pi

        if theta_ret_1<0:
            phi_f += np.pi
        
        theta_ret = np.abs(theta_ret_1)
        phi_ret = (phi%(2*np.pi)) - np.pi

        return theta_ret, phi_ret, w_theta_f, w_phi_f

"""def update(theta, phi, theta_prev, phi_prev):
    if tB != 0:
        noise_amp = np.sqrt(2*D_d)/tB
        r_t = (-np.random.randn()*np.sin(theta) + np.random.randn()*np.cos(theta)*np.cos(phi) + np.random.randn()*np.cos(theta)*np.sin(phi))*noise_amp
        r_p = (np.random.randn()*np.sin(theta) - np.abs(np.random.randn())*np.cos(theta))*noise_amp

        dtp = dt/tB

        ddt_phi = (phi - phi_prev)/dt
        ddt_theta = (theta - theta_prev)/dt

        if np.sin(theta) != 0:
            noise = [r_t/R, r_p/(R*np.sin(theta))]
            kappa = (1/tB) + (2*ddt_theta/np.tan(theta))
        else:
            noise_0 = r_t/R
            theta_f = (2*theta - (1 - dtp/2)*theta_prev + noise_0*dt**2 + np.sin(theta)*np.cos(theta)*ddt_phi**2*dt**2)/(1 + dtp/2)

            if np.sin(theta_f) != 0:
                noise_1 = r_p/(R*np.sin(theta_f))
                kappa = (1/tB) + (2*ddt_theta/np.tan(theta_f))
                phi_f = (2*phi - (1 - dt*kappa/2)*phi_prev + noise_1*dt**2)/(1 + dt*kappa/2)
            else:
                phi_f = (2*phi - (1 - dtp/2)*phi_prev)/(1 + dtp/2)

        theta_ret_1 = (theta_f%(2*np.pi)) - np.pi

        if theta_ret_1<0:
            phi_f += np.pi
        
        theta_ret = np.abs(theta_ret_1)
        phi_ret = (phi%(2*np.pi)) - np.pi

        return theta_ret, phi_ret
    else:
        noise_amp = np.sqrt(2*D_d)
        r_t = (-np.abs(np.random.randn())*np.sin(theta) + np.abs(np.random.randn())*np.cos(theta)*np.cos(phi) + np.abs(np.random.randn())*np.cos(theta)*np.sin(phi))*noise_amp
        r_p = (np.abs(np.random.randn())*np.sin(theta) - np.abs(np.random.randn())*np.cos(theta))*noise_amp

        if np.sin(theta) != 0:

            noise = [r_t/R, r_p/(R*np.sin(theta))]
            theta_f = theta + noise[0]*dt
            phi_f = phi + noise[1]*dt

        else:
            noise_0 = r_t/R
            theta_f = theta + noise_0*dt

            if np.sin(theta_f) != 0:
                noise_1 = r_p/(R*np.sin(theta_f))
                phi_f = phi + noise_1*dt
            else:
                phi_f = phi
        
        theta_ret_1 = (theta_f%(2*np.pi)) - np.pi

        if theta_ret_1<0:
            phi_f += np.pi
        
        theta_ret = np.abs(theta_ret_1)
        phi_ret = (phi%(2*np.pi)) - np.pi

        return theta_ret, phi_ret"""


os.makedirs(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/theta", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/phi", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/w_theta", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/w_phi", exist_ok=True)

j = 0

while j < 100:
    theta_0 = np.pi/2
    phi_0 = 0
    w_theta_i = 0
    w_phi_i = 0

    #noise_amp = np.sqrt(2*gamma/beta)/M
    #r_t = (np.random.randn()*np.sin(theta_0) + np.random.randn()*np.cos(theta_0)*np.cos(phi_0) + np.random.randn()*np.cos(theta_0)*np.sin(phi_0))*noise_amp
    #r_p = (np.random.randn()*np.sin(phi_0) + np.random.randn()*np.cos(phi_0))*noise_amp

    #w_theta_0 = w_theta_i*(1 - dt/tB) + r_t*dt/R
    #noise_1 = r_p/(R*np.sin(theta_0))
    #kappa = 1/tB
    #w_phi_0 = w_phi_i*(1 - kappa*dt) + noise_1*dt

    theta = []
    phi = []
    w_theta = []
    w_phi = []

    theta.append(theta_0)
    phi.append(phi_0)
    w_theta.append(w_theta_i)
    w_phi.append(w_phi_i)

    for i in range(0, int(N_timestep)):
        theta_up, phi_up, w_theta_up, w_phi_up = RK4_update(theta[i], phi[i], w_theta[i], w_phi[i])
        if math.isnan(w_phi_up):
            print(f"Phi, D={D_n:.5e}, trial={j}, time={i}")
        if math.isnan(w_theta_up):
            print(f"Theta, D={D_n:.5e}, trial={j}, time={i}")
        theta.append(theta_up)
        phi.append(phi_up)
        w_theta.append(w_theta_up)
        w_phi.append(w_phi_up)

    if not(any(np.isnan(w_phi)) or any(np.isnan(w_theta))):
        np.savetxt(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/theta/Trial_{j}.txt", theta)
        np.savetxt(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/phi/Trial_{j}.txt", phi)
        np.savetxt(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/w_theta/Trial_{j}.txt", w_theta)
        np.savetxt(f"./codes/Single_Particle_Trace/much_less_than_tB_RK4/N={N_timestep:.2e}/Dn={D_n:.6e}/w_phi/Trial_{j}.txt", w_phi)
        j += 1