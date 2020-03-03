import os
import numpy as np
from scipy import stats
import sys
from func.helpers import *
from func.network import *
import matplotlib.pyplot as plt
na = np.array
import seaborn as sns
import pandas as pd

# %


params= {'J': 2.0,#10.0,
 'g': 4.0,
 'N': 1000,
 'epsilon': 0.2, # percentage of inhibitory neurons
 'eta': 0.0, #external input as nu_ext/nu_thr (Brunel 2000), if 0 not used
 'p_rate': 700.,#external inout rate
 'J_ext': 1.,#mini size
 'tauMem': 20.0,#membrane timescale
 'CMem': 1.0,#membrane capacitance
 'theta': 20.0,#firing threshold
 'V_res': 10.0,#reset threshold
 'Ks': [80,20],#indegrees (excitatory, inhibitory)
 'V_m': 0.0,#resting potential (not working with current adLIF model)
 'b': 0.3,# adaptation increment
 'a': 0.0,#subthrreshold adaptation (not working with current adLIF)
 'tau_w': 4000.,#adaptation timespeace
 'p': 0.1,#propability of connection (when Ks is not specified)
 't_ref': 2.0, #refactory period
 'd': 3.5,#synaptic delay
 'N_rec': 1000,#number of recorded neurons 
 'voltage': False,# record voltage
 'chunk': False,#chunk the simulation (slow implementation at the moment)
 'chunk_size': 1000.0,#size of each chunk
 'directory': 'sim/ABC_small/',#saving direcotry
 'simulation': 'hash',#simualtion name, if 'hash' use the hashed parameters 
 'simtime':200000.0,#simulation length (ms)
 'master_seed': 1000,#master seed for NEST
 'dt': 0.5,#integration timestep
 'threads': 20}#number of threads used

 # %%
A = adLIFNet(params)
A.build()
A.connect()
A.run()
# %%
name = get_hash(params)
st,gid = read_gdf('sim/ABC_small/',name,(5000,params['simtime']),threads = params['threads'])

# %%
plt.figure()
plt.plot(st,gid,'.', markersize = 0.8)
#plt.show()
plt.savefig('foo.png', bbox_inches='tight')
print(st[1:10])
print('done')
# %%
