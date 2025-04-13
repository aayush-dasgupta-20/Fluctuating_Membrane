import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import os
import scipy.integrate as integ
import sys

R = 1
beta = 1 #1/kT

tB = 0

M = tB
gamma = 1

tau_B = M/gamma

D_d = 1/(beta*gamma) #Diffusion Coefficient

kappa = 0 #bending modulus

H_tilde = 1 #Preferred Curvature

D = [[D_d,0,0],[0,D_d,0],[0,0,D_d]] #Diffusion Coefficient Matrix

I = [[1,0,0],[0,1,0],[0,0,1]] #Identity

#dx = 1E-4
dt = 1E-1

def update(theta, phi, theta_prev, phi_prev):
    if tau_B != 0:
        noise_amp = np.sqrt(2*gamma/beta)/M
        r = np.abs(np.random.randn())*noise_amp
        ang = np.random.rand()*np.pi*2

        dtp = dt/tau_B
        if np.sin(theta) != 0:
            noise = [r*np.cos(ang)/R, r*np.sin(ang)/(R*np.sin(theta))]

            theta_f = (2*theta - (1 - dtp/2)*theta_prev + noise[0]*dt**2)/(1 + dtp/2)
            phi_f = (2*phi - (1 - dtp/2)*phi_prev + noise[1]*dt**2)/(1 + dtp/2)
        else:
            noise_0 = r*np.cos(ang)/R
            theta_f = (2*theta - (1 - dtp/2)*theta_prev + noise_0*dt**2)/(1 + dtp/2)

            if np.sin(theta_f) != 0:
                noise_1 = r*np.sin(ang)/(R*np.sin(theta_f))
                phi_f = (2*phi - (1 - dtp/2)*phi_prev + noise_1*dt**2)/(1 + dtp/2)
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
        r = np.abs(np.random.randn())*noise_amp
        ang = np.random.rand()*np.pi*2

        if np.sin(theta) != 0:

            noise = [r*np.cos(ang)/R, r*np.cos(ang)/(R*np.sin(theta))]
            theta_f = theta + noise[0]*dt
            phi_f = phi + noise[1]*dt

        else:
            noise_0 = r*np.cos(ang)/R
            theta_f = theta + noise_0*dt

            if np.sin(theta_f) != 0:
                noise_1 = r*np.cos(ang)/(R*np.sin(theta_f))
                phi_f = phi + noise_1*dt
            else:
                phi_f = phi
        
        theta_ret_1 = (theta_f%(2*np.pi)) - np.pi

        if theta_ret_1<0:
            phi_f += np.pi
        
        theta_ret = np.abs(theta_ret_1)
        phi_ret = (phi%(2*np.pi)) - np.pi

        return theta_ret, phi_ret

os.makedirs(f"./codes/Single_Particle_Trace/tB={tB:.6f}", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/tB={tB:.6f}/theta", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/tB={tB:.6f}/phi", exist_ok=True)

for j in range(100):
    theta_0 = np.pi/2
    phi_0 = 0

    theta = []
    phi = []

    theta.append(theta_0)
    phi.append(phi_0)

    theta_1, phi_1 = update(theta_0, phi_0, theta_0, phi_0)

    theta.append(theta_1)
    phi.append(phi_1)

    for i in range(1, 2500):
        theta_1, phi_1 = update(theta[i], phi[i], theta[i-1], phi[i-1])
        theta.append(theta_1)
        phi.append(phi_1)
    
    np.savetxt(f"./codes/Single_Particle_Trace/tB={tB:.6f}/theta/Trial_{j}.txt", theta)
    np.savetxt(f"./codes/Single_Particle_Trace/tB={tB:.6f}/phi/Trial_{j}.txt", phi)

def msds(theta, phi):
    geodesic_msd = np.zeros(int(len(theta)/2))
    geometric_msd = np.zeros(int(len(theta)/2))

    for i in range(len(geodesic_msd)):
        geodesic_msd_i = 0
        geometric_msd_i = 0
        count = 0
        for j in range(len(theta) - i):
            R1 = [R*np.sin(theta[j])*np.cos(phi[j]), R*np.sin(theta[j])*np.sin(phi[j]), R*np.cos(theta[j])]
            R2 = [R*np.sin(theta[j+i])*np.cos(phi[j+i]), R*np.sin(theta[j+i])*np.sin(phi[j+i]), R*np.cos(theta[j+i])]

            c = (R1[0]*R2[0] + R1[1]*R2[1] + R1[2]*R2[2])/(R**2)

            if np.abs(c)<1:
                geodesic_msd_i += R*np.arccos(c)
            elif c>=1:
                geodesic_msd_i += 0
            elif c<=-1:
                geodesic_msd_i += R*np.pi 
            geometric_msd_i += np.sqrt((R1[0] - R2[0])**2 + (R1[1] - R2[1])**2 + (R1[2] - R2[2])**2)
            count += 1

        geodesic_msd[i] = geodesic_msd_i/count
        geometric_msd[i] = geometric_msd_i/count

    return geodesic_msd, geometric_msd


os.makedirs(f"./codes/Single_Particle_Trace/tB={tB:.6f}/tamsad", exist_ok = True)
os.makedirs(f"./codes/Single_Particle_Trace/tB={tB:.6f}/tamsd", exist_ok = True)

theta_dir = f"./codes/Single_Particle_Trace/tB={tB:.6f}/theta"
phi_dir = f"./codes/Single_Particle_Trace/tB={tB:.6f}/phi"

files_theta = [file for file in os.listdir(theta_dir)]
files_phi = [file for file in os.listdir(phi_dir)]

N = len(files_theta)
l = len(np.loadtxt(os.path.join(theta_dir, files_theta[0])))

T = np.arange(0, int(l/2)*dt, dt)

msad = np.zeros(int(l/2))
msd = np.zeros(int(l/2))

for i in range(N):
    tamsad, tamsd = msds(np.loadtxt(os.path.join(theta_dir, files_theta[i])), np.loadtxt(os.path.join(phi_dir, files_phi[i]))) #time averaged MSD
    tamsad = tamsad/R
    tamsd = tamsd/R

    np.savetxt(f"./codes/Single_Particle_Trace/tB={tB:.6f}/tamsad/Trial_{i}.txt", tamsad)
    np.savetxt(f"./codes/Single_Particle_Trace/tB={tB:.6f}/tamsd/Trial_{i}.txt", tamsd)
    msad += tamsad
    msd += tamsd

msad = msad/N #Ensemble Averaged MSD
msd = msd/N #Ensemble Averaged MSD

np.savetxt(f"./codes/Single_Particle_Trace/tB={tB:.6f}/msad.txt", msad)
np.savetxt(f"./codes/Single_Particle_Trace/tB={tB:.6f}/msd.txt", msd)
                        
plt.title("MSD Plot")
plt.xlabel("Time")
plt.ylabel("MSD/R")
plt.grid(visible=True)
plt.plot(T, msd)
plt.savefig(f"./codes/Single_Particle_Trace/tB={tB:.6f}/msd.png")
plt.close()
                        
plt.title("MSAD Plot")
plt.xlabel("Time")
plt.ylabel("MSAD")
plt.grid(visible=True)
plt.plot(T, msad)
plt.savefig(f"./codes/Single_Particle_Trace/tB={tB:.6f}/msad.png")
plt.close()
                        
plt.title("MSD Plot")
plt.xlabel("Time")
plt.ylabel("MSD/R")
plt.grid(visible=True)
plt.plot(T, msd)
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/tB={tB:.6f}/ll_msd.png")
plt.close()
                        
plt.title("MSAD Plot")
plt.xlabel("Time")
plt.ylabel("MSAD")
plt.grid(visible=True)
plt.plot(T, msad)
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/tB={tB:.6f}/ll_msad.png")
plt.close()

llmsad = np.log10(msad)
llmsd = np.log10(msd)

slope_msad = (np.roll(llmsad, shift = -1) - llmsad)/dt
slope_msd = (np.roll(llmsd, shift = -1) - llmsd)/dt

plt.title("MSD Plot")
plt.xlabel("Time")
plt.ylabel("MSD")
plt.grid(visible=True)
plt.plot(T[0:-1], slope_msd[0:-1])
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/tB={tB:.6f}/ll_slope_msd.png")
plt.close()
                        
plt.title("MSAD Plot")
plt.xlabel("Time")
plt.ylabel("MSAD")
plt.grid(visible=True)
plt.plot(T[0:-1], slope_msad[0:-1])
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/tB={tB:.6f}/ll_slope_msad.png")
plt.close()

pi = np.pi

def msad_pred(t):
    r1 = 1 - ((3*pi**2)*np.exp(-2*D_d*t/R**2)/(4*pi**2 - 16))

    return r1*(pi**2 - 4)/2

msad_prediction = msad_pred(T)

print(msad_prediction)
                        
plt.title("MSAD Plot")
plt.xlabel("Time")
plt.ylabel("MSAD")
plt.grid(visible=True)
plt.plot(T, msad, label = 'simulation')
plt.plot(T, msad_prediction, label = 'first order prediction')
plt.legend()
plt.savefig(f"./codes/Single_Particle_Trace/tB={tB:.6f}/msad_comparison.png")
plt.close()