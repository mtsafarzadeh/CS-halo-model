import astroML
from sklearn.neighbors import BallTree
import numpy as np
import random

#reading in the file containing the mass and positions of the halos from a N-body simulation. 
All_data=np.loadtxt('your_halo_catalog.csv',delimiter=',')

redshift=0.5 # assuming the redshift is z=0.5
x=All_data[:,1] #reading the X,Y and Z positions of the halos in Comoving units.
y=All_data[:,2]
z=All_data[:,3]
mass=All_data[:,4]#readin the mass in M_sun units
Halo_ID=All_data[:,6] # reading the ID of the halos

factor=(1/.28 -1)**.33 *(1+redshift)**-1
omega_z=1/(1+factor**3)

r_vir=0.184*(mass/1e12)**.33 * (omega_z/0.28)**0.33 # r_virial of halos from NFW formulation
candidates=((mass>2e11)).nonzero()[0] # cut for the minimum mass of the halos considered in the CS model, here m_{h,0}=2e11 M_sun
rand_ids=random.sample(candidates,100000) # sample a fraction of the candidiates, for example chosing 100,000 halos  

#making data tree
data=np.array([x,y,z]).T
data_tree=BallTree(data)

#searching the data tree finding halos in CS model with DR_0 =1.
DR0=1.
good_dr=[]
for n in np.arange(0,len(rand_ids)):
        halo=rand_ids[n]
	#searching within 10 Mpc of halos
        halo_neighbors=data_tree.query_radius(data[halo,:], 10., count_only = False, return_distance = False)
        concat_halos=np.concatenate(halo_neighbors)
        pruned_halos=(mass[concat_halos]>.5*mass[halo]).nonzero()[0]
        SB=concat_halos[pruned_halos]
        if (len(SB)<2): # if no halo is found within 10 Mpc radius, analyze the next halo candidate
                continue
        d=np.ones(len(SB))*1000
        i=0
        for sample in SB:
                d[i]=( (x[sample]-x[halo])**2+(y[sample]-y[halo])**2 + (z[sample]-z[halo])**2 )**0.5
                d[i]/=r_vir[sample]
                i+=1
        DR=min(d[d>0])
        exc=((d<1)&(d>0)).nonzero()[0]
        MM=mass[SB[exc]]
        MM_ind=(MM>1e13).nonzero()[0] 
        if len(MM_ind)>0: # excluding the halo if it resides in 1e13 M_sun halo due to lack of cold gas reservoir
                continue
        if (DR<DR0): # PDF(DR)=1 for DR<DR0
                good_dr.append(halo)
                continue

        if DR>DR0: #PDF(DR) =  (DR0./DR)^3 for DR>DR0
                random_n=np.random.uniform(0,1)
                if random_n<(1./DR)**3 :
                        good_dr.append(halo)
                        continue
#good_dr contains the list of the CS candidate halos.
#writing the position, mass and ID of the CS candidates into a file
file=open('CS_model_halos.dat','w+')
count=0
for indx in good_dr:
        print>>file,count,',',x[indx],',',y[indx],',',z[indx],',',mass[indx],',',Halo_ID[indx]
        count+=1
file.close()
