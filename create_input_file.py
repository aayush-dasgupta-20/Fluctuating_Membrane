from create_config_file import config_file
import numpy as np
from subprocess import check_output


R=1
nsamples=65
beta_list= np.arange(0.75,1.25,0.05)
#workdir_main = '/home/sgv/Data/Afrac_%2.1f_R_%d'%(Afrac,R)
workdir_main = '/home/aayushd/codes/Single_Particle_Sphere_Simulation'
command= 'mkdir -p %s'%workdir_main
check_output(command, shell=True)
inFileList = "%s/inFileList.txt"%(workdir_main)
inDirList =  "%s/inDirList.txt"%(workdir_main)
ifl = open(inFileList,'w')
idl = open(inDirList,'w')
for runid in range(1,nsamples):

    seed_value = np.random.randint(10000000)
    command = 'mkdir -p %s/Run%d_Active'%(workdir_main,runid)
    check_output(command, shell=True)        

    for beta in beta_list:
        #print('pbind:', pbind, 'Runid:',runid, "Temperature:", T)
        dir = "%s/Run%d/beta_%d"%(workdir_main,runid, beta)
        command = 'mkdir -p ' + dir
        check_output(command, shell=True)
          
        cfgFile = "%s/tB_%d.in"%(dir, runid)
          
        currline = config_file(R, runid, seed_value, beta)  # Use the config_file function
        f = open(cfgFile, 'w')
        f.writelines(currline)
        f.close()
          
        ifl.writelines(cfgFile+"\n")
        idl.writelines(dir+"\n")

'''
for aster_scale in aster_scale_list:
    command = 'mkdir -p %s/aster_scale_%d'%(workdir_main, aster_scale)
    check_output(command, shell=True)
    workdir = '%s/aster_scale_%d'%(workdir_main, aster_scale)
    for pbind in pbind_list:
        command = 'mkdir -p %s/Data_pbind_%2.1f'%(workdir, pbind)
        check_output(command, shell=True)
        present_dir = '%s/Data_pbind_%2.1f'%(workdir, pbind)
        for runid in range(nsamples):
            
            seed_value = np.random.randint(10000000)
            command = 'mkdir -p %s/Run%d_Active'%(present_dir,runid)
            check_output(command, shell=True)

            

            for T in T_list:
                print('pbind:', pbind, 'Runid:',runid, "Temperature:", T)
                dir = "%s/Run%d_Active/T_%d"%(present_dir,runid, T)
                command = 'mkdir -p ' + dir
                check_output(command, shell=True)
                
                cfgFile = "%s/Run_%d.in"%(dir, runid)
                
                currline = config_file(L, spin_conc, runid, pbind, seed_value, T, aster_scale,R)  # Use the config_file function
                f = open(cfgFile, 'w')
                f.writelines(currline)
                f.close()
                
                ifl.writelines(cfgFile+"\n")
                idl.writelines(dir+"\n")

            
            
            
            spin=np.zeros((L,L),dtype=int)
            np.random.seed(seed_value)
          
            for i in range(L): #initialisation
                for j in range(L):
                    if np.random.random()>=spin_conc:
                        spin[i,j]=0
                    else:
                        spin[i,j]=1
            filepath='%s/Run%d_Active/'%(present_dir ,runid)
            np.savetxt(filepath+'initial_config_L_'+str(L)+'.npy',spin, delimiter=',',fmt='%d')
'''
ifl.close()
idl.close()



