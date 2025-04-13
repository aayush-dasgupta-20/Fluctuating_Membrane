import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import os
import scipy.integrate as integ

def height(x,y): #height field function
    return (x**2 + y**2)/2

def laplacian(x,y):
    return 2

def grad_laplacian(x,y):
    return 0,0

def grad_h(x,y):
    return x,y

beta = 1 #1/kT

gamma = 1 #1/friction coeff

M = 1

tau_B = M/gamma

D_d = 1/(beta*gamma) #Diffusion Coefficient

kappa = 0 #bending modulus

H_tilde = 1 #Preferred Curvature

D = [[D_d,0,0],[0,D_d,0],[0,0,D_d]] #Diffusion Coefficient Matrix

I = [[1,0,0],[0,1,0],[0,0,1]] #Identity

dx = 1E-4
dt = 1E-2

def update(x,y,x_p,y_p): #update_function
    dxh, dyh = grad_h(x,y)

    n = [[-dxh], [-dyh], [0]] #Normal_Vector
    nT = np.transpose(n) 

    P = I - np.matmul(n,nT) #Projection Matrix

    #grad_lap_x, grad_lap_y = grad_laplacian(x,y)
    
    #F_e = [[kappa*H_tilde*grad_lap_x], [kappa*H_tilde*grad_lap_y], [0]] #force_due_to_curvature
    F_e = [[0], [0], [0]] #force_due_to_curvature

    noise_amplitude = np.sqrt(2*gamma/beta)/M

    r = np.abs(np.random.randn())*noise_amplitude
    theta = np.random.rand()*np.pi
    phi = np.random.rand()*2*np.pi

    F_gn = [[r*np.sin(theta)*np.cos(phi)], [r*np.sin(theta)*np.sin(phi)], [r*np.cos(theta)]] #Random Noise

    F_tot = [[F_e[0][0] + F_gn[0][0]], [F_e[1][0] + F_gn[1][0]], [F_e[2][0] + F_gn[2][0]]] #Total Unconstrained Force

    v_uc = gamma*np.matmul(D,F_tot) #Unconstrained Velocity

    v_c = np.matmul(P,v_uc) #Constrained Velocity = Projection onto the Membrane

    dt_p = dt/tau_B

    x_f = v_c[0][0]*dt * dt_p + (2 - dt_p)*x - (1 - dt_p)*x_p
    y_f = v_c[1][0]*dt * dt_p + (2 - dt_p)*y - (1 - dt_p)*y_p

    return x_f,y_f

os.makedirs(f"./codes/Single_Particle_Trace", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/x", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/y", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/x_b", exist_ok=True)
os.makedirs(f"./codes/Single_Particle_Trace/y_b", exist_ok=True)

for j in range(100):
    x0 = np.random.rand() - 0.5 #random_starting_point
    y0 = np.random.rand() - 0.5 #random_starting_point

    x = []
    y = []

    x.append(x0)
    y.append(y0)

    x_b = []
    y_b = []

    x_b.append(x0)
    y_b.append(y0)

    dt = 1E-2
    dx = 1E-4

    x1, y1 = update(x0,y0,x0,y0)

    x.append(x1)
    y.append(y1)
    x_b.append(x1)
    y_b.append(y1)

    for i in range(1000):
        xf,yf = update(x[-1],y[-1], x[-2], y[-2])
        x.append(xf)
        y.append(yf)

        x_b.append(xf%1 - 0.5)
        y_b.append(yf%1 - 0.5)
        
    np.savetxt(f"./codes/Single_Particle_Trace/x/Trial_{j}.txt", x)
    np.savetxt(f"./codes/Single_Particle_Trace/y/Trial_{j}.txt", y)
    np.savetxt(f"./codes/Single_Particle_Trace/x_b/Trial_{j}.txt", x_b)
    np.savetxt(f"./codes/Single_Particle_Trace/y_b/Trial_{j}.txt", y_b)

x_m = np.linspace(-0.5,0.5,256)
X,Y = np.meshgrid(x_m,x_m)

H = height(X,Y)
np.savetxt(f"./codes/Single_Particle_Trace/Height.txt", H)

def mean_sd(x,y):
    geo_msd = np.zeros(int(len(x)/2)) #MSD Is calculated upto t_max/2
    proj_msd = np.zeros(int(len(x)/2))
    
    for i in range(len(geo_msd)): #i is the number of timesteps
        x_shift = np.roll(x, shift = -i) 
        y_shift = np.roll(y, shift = -i)

        proj_dis_x = x_shift - x
        proj_dis_y = y_shift - y
        proj_dis_z = height(x_shift,y_shift) - height(x,y) 

        #N = 64

        #dx = (x_shift - x)/N
        #dy = (y_shift - y)/N
        
        #geo_dis_sq_i = np.zeros(x.shape)

        #for j in range(0,N-1):
        #    dh = height(x + (j+1)*dx, y + (j+1)*dy) - height(x + j*dx, y + j*dy)
        #    geo_dis_sq_i += np.sqrt(dx**2 + dy**2 + dh**2)

        #geo_dis_sq_i = np.square(geo_dis_sq_i)

        if i != 0:
            proj_dis_x = proj_dis_x[:-i]
            proj_dis_y = proj_dis_y[:-i]
            proj_dis_z = proj_dis_z[:-i]
            #geo_dis_sq_i = geo_dis_sq_i[:-i]
        
        proj_sd_i = np.square(proj_dis_x) + np.square(proj_dis_y) + np.square(proj_dis_z) #square displacement
        
        proj_msd[i] = np.average(proj_sd_i) #msd for given particle trajectory (Time Averaged MSD)

        geo_msd[i] = proj_msd[i]
        
    return geo_msd, proj_msd

def model(var, x, p):
    eq1 = var[0]*(x + var[1]*var[0]**3)/(1 + x**2 + 2*var[0]**2*(1 + var[1]**2))
    return np.array(eq1,var[0])

def bc(za,zb,p):
    y0,y1 = p
    return np.array([za[1]-y0, zb[1]-y1])

def gd(x0,x1,y0,y1):
    if x1>x0:
        x = np.linspace(x0,x1,20)
        x_sol = np.linspace(x0,x1,100)
        y_guess = np.zeros((2,x.size))
        z = integ.solve_bvp(model, bc, x, y_guess, p = [y0,y1])
        Y = z.sol(x_sol)[1]
        H = height(x_sol, Y)
        dx = np.roll(x_sol, shift = -1) - x_sol
        dy = np.roll(Y, shift = -1) - Y
        dz = np.roll(H, shift = -1) - H
        ds = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))
        L = np.sum(ds[:-1])
    elif x1<x0:
        x = np.linspace(x1,x0,20)
        x_sol = np.linspace(x1,x0,100)
        y_guess = np.zeros((2,x.size))
        z = integ.solve_bvp(model, bc, x, y_guess, p = [y1,y0])
        Y = z.sol(x_sol)[1]
        H = height(x_sol, Y)
        dx = np.roll(x_sol, shift = -1) - x_sol
        dy = np.roll(Y, shift = -1) - Y
        dz = np.roll(H, shift = -1) - H
        ds = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))
        L = np.sum(ds[:-1])

    return L**2

def geodesic_msd(x,y):
    geodesic_msd = np.zeros(int(len(x)/2))

    for i in range(len(geodesic_msd)): #i is the number of timesteps
        geo_msd_i = 0
        count = 0

        for j in range(len(x) - i):
            geo_msd_i+=gd(x[j],y[j],x[j+i],y[j+i])
            count+=1
        
        geodesic_msd[i] = geo_msd_i/count
    
    return geodesic_msd
  

x_dir = f"./codes/Single_Particle_Trace/x"
y_dir = f"./codes/Single_Particle_Trace/y"

files_x = [file for file in os.listdir(x_dir)]
files_y = [file for file in os.listdir(y_dir)]

os.makedirs(f"./codes/Single_Particle_Trace/tamsd", exist_ok = True)
os.makedirs(f"./codes/Single_Particle_Trace/tarmsd", exist_ok = True)
os.makedirs(f"./codes/Single_Particle_Trace/proj_tamsd", exist_ok = True)
os.makedirs(f"./codes/Single_Particle_Trace/proj_tarmsd", exist_ok = True)

N = len(files_x)
l = len(np.loadtxt(os.path.join(x_dir, files_x[0])))

T = np.arange(0, int(l/2)*dt, dt)

msd = np.zeros(int(l/2))
proj_msd = np.zeros(int(l/2))

for i in range(N):
    tamsd, proj_tamsd = mean_sd(np.loadtxt(os.path.join(x_dir, files_x[i])), np.loadtxt(os.path.join(y_dir, files_y[i]))) #time averaged MSD
    tamsd = geodesic_msd(np.loadtxt(os.path.join(x_dir, files_x[i])), np.loadtxt(os.path.join(y_dir, files_y[i])))
    tarmsd = np.sqrt(tamsd) #Time Averaged RMSD
    proj_tarmsd = np.sqrt(proj_tamsd) 

    np.savetxt(f"./codes/Single_Particle_Trace/tamsd/Trial_{i}.txt", tamsd)
    np.savetxt(f"./codes/Single_Particle_Trace/tarmsd/Trial_{i}.txt", tarmsd)
    np.savetxt(f"./codes/Single_Particle_Trace/proj_tamsd/Trial_{i}.txt", proj_tamsd)
    np.savetxt(f"./codes/Single_Particle_Trace/proj_tarmsd/Trial_{i}.txt", proj_tarmsd)
    msd += tamsd
    proj_msd += proj_tamsd

msd = msd/N #Ensemble Averaged MSD
rmsd = np.sqrt(msd) #Ensemble Averaged RMSD
proj_msd = proj_msd/N #Ensemble Averaged MSD
proj_rmsd = np.sqrt(proj_msd) #Ensemble Averaged RMSD

np.savetxt(f"./codes/Single_Particle_Trace/msd.txt", msd)
np.savetxt(f"./codes/Single_Particle_Trace/rmsd.txt", rmsd)
np.savetxt(f"./codes/Single_Particle_Trace/proj_msd.txt", proj_msd)
np.savetxt(f"./codes/Single_Particle_Trace/proj_rmsd.txt", proj_rmsd)

plt.title("RMSD Plot")
plt.xlabel("Time")
plt.ylabel("RMSD")
plt.grid(visible=True)
plt.plot(T, rmsd)
plt.savefig(f"./codes/Single_Particle_Trace/rmsd.png")
plt.close()
                        
plt.title("MSD Plot")
plt.xlabel("Time")
plt.ylabel("MSD")
plt.grid(visible=True)
plt.plot(T, msd)
plt.savefig(f"./codes/Single_Particle_Trace/msd.png")
plt.close()

plt.title("Projected RMSD Plot")
plt.xlabel("Time")
plt.ylabel("RMSD")
plt.grid(visible=True)
plt.plot(T, proj_rmsd)
plt.savefig(f"./codes/Single_Particle_Trace/proj_rmsd.png")
plt.close()
                        
plt.title("Projected MSD Plot")
plt.xlabel("Time")
plt.ylabel("MSD")
plt.grid(visible=True)
plt.plot(T, proj_msd)
plt.savefig(f"./codes/Single_Particle_Trace/proj_msd.png")
plt.close()

plt.clf()
msd_1 = 0
for i in range(N):
    tamsd_1 = np.loadtxt(f"./codes/Single_Particle_Trace/tamsd/Trial_{i}.txt")
    msd_1 += tamsd_1
    T_1 = np.arange(0, int(l/2)*1e-2, 1e-2)
    plt.plot(T_1, tamsd_1, lw=0.5, c="k", alpha=0.4)
msd_1 = msd_1/N
plt.plot(T_1, msd_1, c='r', lw=2)
plt.title("MSD Plot")
plt.xlabel("Time")
plt.ylabel("MSD")
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.grid(visible=True)
plt.savefig(f"./codes/Single_Particle_Trace/log_log_msd_combined.png")
plt.close()

plt.clf()
proj_msd_1 = 0
for i in range(N):
    proj_tamsd_1 = np.loadtxt(f"./codes/Single_Particle_Trace/proj_tamsd/Trial_{i}.txt")
    proj_msd_1 += proj_tamsd_1
    T_1 = np.arange(0, int(l/2)*dt, dt)
    plt.plot(T_1, proj_tamsd_1, lw=0.5, c="k", alpha=0.4)
proj_msd_1 = proj_msd_1/N
plt.plot(T_1, proj_msd_1, c='r', lw=2)
plt.title("Projected MSD Plot")
plt.xlabel("Time")
plt.ylabel("MSD")
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.grid(visible=True)
plt.savefig(f"./codes/Single_Particle_Trace/log_log_proj_msd_combined.png")
plt.close()

llmsd = np.log10(msd[1:])
llrmsd = np.log10(rmsd[1:])
llproj_msd = np.log10(proj_msd[1:])
llproj_rmsd = np.log10(proj_rmsd[1:])
llT = np.log10(T[1:])

plt.title("log-log RMSD Plot")
plt.xlabel("Time")
plt.ylabel("RMSD")
plt.grid(visible=True)
plt.plot(T, rmsd)
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/log_log_rmsd.png")
plt.close()
                        
plt.title("log-log MSD Plot")
plt.xlabel("log(Time)")
plt.ylabel("log(MSD)")
plt.grid(visible=True)
plt.plot(T, msd)
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/log_log_msd.png")
plt.close()

plt.title("log-log Projected RMSD Plot")
plt.xlabel("Time")
plt.ylabel("RMSD")
plt.grid(visible=True)
plt.plot(T, proj_rmsd)
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/log_log_proj_rmsd.png")
plt.close()
                        
plt.title("log-log Projected MSD Plot")
plt.xlabel("Time")
plt.ylabel("MSD")
plt.grid(visible=True)
plt.plot(T, proj_msd)
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.savefig(f"./codes/Single_Particle_Trace/log_log_proj_msd.png")
plt.close()

del_rmsd = np.roll(llrmsd, shift = -1) - llrmsd
del_msd = np.roll(llmsd, shift = -1) - llmsd
del_proj_rmsd = np.roll(llproj_rmsd, shift = -1) - llproj_rmsd
del_proj_msd = np.roll(llproj_msd, shift = -1) - llproj_msd
del_T = np.roll(llT, shift = -1) - llT

rmsd_slope = np.divide(del_rmsd[:-1],del_T[:-1]) #slope_of_loglog_rmsd_and_msd_plots
msd_slope = np.divide(del_msd[:-1],del_T[:-1])
proj_rmsd_slope = np.divide(del_proj_rmsd[:-1],del_T[:-1]) #slope_of_loglog_rmsd_and_msd_plots
proj_msd_slope = np.divide(del_proj_msd[:-1],del_T[:-1])

plt.title("Slope of log-log RMSD Plot")
plt.xlabel("log(Time)")
plt.ylabel("Derivative of loglog RMSD Plot")
plt.grid(visible=True)
plt.plot(llT[:-1], rmsd_slope)
plt.savefig(f"./codes/Single_Particle_Trace/log_log_slope_rmsd.png")
plt.close()
                        
plt.title("Slope of log-log MSD Plot")
plt.xlabel("log(Time)")
plt.ylabel("Derivative of loglog MSD Plot")
plt.grid(visible=True)
plt.plot(llT[:-1], msd_slope)
plt.savefig(f"./codes/Single_Particle_Trace/log_log_slope_msd.png")
plt.close()

plt.title("Slope of log-log RMSD Plot")
plt.xlabel("log(Time)")
plt.ylabel("Derivative of loglog RMSD Plot")
plt.grid(visible=True)
plt.plot(llT[:-1], proj_rmsd_slope)
plt.savefig(f"./codes/Single_Particle_Trace/log_log_slope_proj_rmsd.png")
plt.close()
                        
plt.title("Slope of log-log MSD Plot")
plt.xlabel("log(Time)")
plt.ylabel("Derivative of loglog MSD Plot")
plt.grid(visible=True)
plt.plot(llT[:-1], proj_msd_slope)
plt.savefig(f"./codes/Single_Particle_Trace/log_log_slope_proj_msd.png")
plt.close()

def J_alt(t):
    x = np.exp(-t/tau_B)
    tn = t/tau_B
    sum1 = x*(8 + x**3 - 8*x**2 + 36*x + 48*tn + (12*tn - 37)/x)
    sum2 = x**4 - 8*x**3 + (15 - 6*tn)*x**2 - 40*x + (32 + 6*tn*(tn - 4))

    return (sum1 + sum2)/6

def msd_prediction(t):
    x = np.exp(-t/tau_B)
    tn = t/tau_B

    s1 = (tn - (x**2 - 1)/2 - 2*(1 - x))*4*D_d*tau_B
    s2 = 2*(tau_B**2*D_d**2)*J_alt(t)/3

    return s1 - s2

msd_pred = msd_prediction(T)
           
plt.title("log-log MSD Plot")
plt.xlabel("Time")
plt.ylabel("MSD")
plt.grid(visible=True)
plt.plot(T, msd)
plt.plot(T, msd_pred)
plt.xscale('log')
plt.yscale('log') #log_log_plot
plt.legend(['msd','msd prediction'])
plt.savefig(f"./codes/Single_Particle_Trace/msd_log_comparison.png")
plt.close()

plt.title("MSD Plot")
plt.xlabel("Time")
plt.ylabel("MSD")
plt.grid(visible=True)
plt.plot(T, msd)
plt.plot(T, msd_pred)
plt.legend(['msd','msd prediction'])
plt.savefig(f"./codes/Single_Particle_Trace/msd_comparison.png")
plt.close()

llmsd_pred = np.log10(msd_pred[1:])
del_msd_pred = np.roll(llmsd_pred, shift = -1) - llmsd_pred
msd_pred_slope = np.divide(del_msd_pred[:-1],del_T[:-1])
                        
plt.title("Slope of log-log MSD Plot")
plt.xlabel("log(Time)")
plt.ylabel("Derivative of loglog MSD Plot")
plt.grid(visible=True)
plt.plot(llT[:-1], msd_slope)
plt.plot(llT[:-1], msd_pred_slope)
plt.legend(['msd','msd prediction'])
plt.savefig(f"./codes/Single_Particle_Trace/msd_slope_comparison.png")
plt.close()