import nest
import nest.raster_plot
import os
os.environ['LD_LIBRARY_PATH'] = '/home/ovinogradov/.local'
nest.Install('myModels')

import time
from numpy import exp
import numpy as np
import h5py as h5py
import os 
import json 
import hashlib

class ChunkError(Exception): pass
class BuildError(Exception): pass

class adLIFNet(object):
    """ Network of LIF neurons. 
    inspired by Gewaltig et al. 2012"""
    built = False
    connected = False
    verbose = False
#     threads = 7
#     dt      = float(0.5)    # the resolution in ms for the clock-based
#     dt = 0.1
    def __init__(self,params):
        """
                #simulation = 'test',
                #directory = 'test',
                #g=7, # inhibitory strenght
                #eta = 3.5, #Poisson rate ratio
                #d=[3.5], # synaptic delay
                #J=0.1, #synaptic strength
                #NE =8000, # fraction of inh neurons
                #NI =2000,
                #N_rec = 50,
                #epsilon = 0.1,
                #tauMem= 20.0,
                #simtime=1000,
                #master_seed = 1000,
                #verbose = True,
                #chunk = False,
                #chunk_size = 10,
                #voltage= False):
        """
        """Initialize the simulation , setup data directory"""
        self.name=self.__class__.__name__
        if params['simulation'] =='hash':
            params_json = json.dumps(params)
            hash_name = hashlib.sha256(params_json.encode('utf-8')).hexdigest()
            self.simulation = hash_name
        else:
            self.simulation =params['simulation']
        self.data_path=params['directory']+'/'+self.simulation
        nest.ResetKernel()
        #if self.verbose ==True:
        #    nest.set_verbosity('M_FATAL')#"M_ALL")
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            print('”Writing data to :'+self.data_path)
        self.dt = params['dt']
        if 'threads' in params:
            self.threads = params['threads']
        else:
            self.threads = 40
            
        nest.SetKernelStatus({'resolution': self.dt, 'print_time': False,
                              'overwrite_files':True,
                              'local_num_threads':self.threads,
                              'data_path': self.data_path})
        self.g = params['g']
        self.eta = params['eta']
        self.delay = params['d']
        self.J = params['J']
        self.epsilon = params['epsilon']
        # Neurons and degrees
        self.N = int(params['N'])
        self.NI = int(self.N *self.epsilon)
        self.NE = int(self.N- self.NI)
        self.p = params['p']
        self.KE    = int(self.p*self.NE) # number of excitatory synapses per neuron
        if 'K_balance' in params:
            self.KI    = int(self.KE/self.g)
        else:
            self.KI    = int(self.p*self.NI) # number of inhibitory synapses per neuron  
        if 'Ks' in params:
            self.KE    = params['Ks'][0]
            self.KI    = params['Ks'][1]
            
        self.K_tot = int(self.KI+self.KE)
        
        self.N_rec = params['N_rec']
        # Neurons' params 
        self.tauMem = params['tauMem']
        self.CMem =params['CMem']
        self.V_m = params['V_m']
        self.V_res = params['V_res']#10
        self.t_ref = params['t_ref']
        self.theta = params['theta']
        self.a = params['a']
        self.b = params['b']
        self.tau_w = params['tau_w']
        #Simulation params
        self.simtime = params['simtime']
        self.chunk = params['chunk']
        if self.chunk:
            try:
                self.chunk_size = params['chunk_size']
            except:
                raise ChunkError('check the simulation length') 
            
        self.record_vol = params['voltage']
        #
        self.p_rate = params['p_rate']
        self.eta = params['eta']
        self.J_ext = params['J_ext']
        #Some statistics
        self.in_deg_saved = False
        self.out_deg_saved = False
        if 'i_noise_off' in params.keys():
            self.i_noise_off = True
        else:
            self.i_noise_off = False
            
        if self.verbose:
            #neurons output
            print('Number of neurons (E/I):')
            print(self.NE,self.NI)
            print('Connection Probability: %s'%(self.p))

        #Set the RNG seeds for each thread
        self.master_seed = params['master_seed']#2000#1000
        self.n_vp = nest.GetKernelStatus('total_num_virtual_procs')
        self.msdrange1 = range(self.master_seed, self.master_seed+self.n_vp)
        self.pyrngs = [ np.random.RandomState(s) for s in self.msdrange1]
        self.msdrange2 = range(self.master_seed+self.n_vp+1, self.master_seed+1+2*self.n_vp)
        nest.SetKernelStatus({'grng_seed': self.master_seed+self.n_vp,
                              'rng_seeds' :self.msdrange2}) 
        
    def build(self):
        #Initialization of the parameters of the integrate and fire neuron and the synapses. The parameter of the neuron are stored in a dictionary.
        self.neuron_params = {'C_m':self.CMem,
                              'tau_m':self.tauMem,
                              't_ref':self.t_ref,
                              'E_L': 0.0,
                              'V_reset': self.V_res,
                              'V_abs': self.V_m,
                              'Theta':self.theta,
                              'with_refr_input':False,
                              'tau_w':self.tau_w,
                              'a':self.a,
                              'b':self.b}
                             #}
        if self.verbose:
            print('Check Neurons params \n')
            print(self.neuron_params)


        self.J_ex  =self.J       # amplitude of excitatory postsynaptic potential
        self.J_in  = -self.g*self.J_ex # amplitude of inhibitory postsynaptic potential
        if self.verbose:
            print('J_i = %s'%(self.J_in))

        #Poisson Rate
        if self.eta>0:
            
            self.nu_th =self.theta / (self.J * self.KE * self.tauMem)# (theta * CMem) / (J_ex * CE * exp(1) * tauMem * tauSyn)
            self.nu_ex = self.eta * self.nu_th
            self.p_rate = 1000.0 * self.nu_ex * self.KE
        else:
            self.p_rate =self.p_rate
            
#         print('p_rate')    
#         print(self.p_rate)        
        
        nest.SetDefaults("adapt_lif", self.neuron_params)
        nest.SetDefaults("poisson_generator",{"rate":self.p_rate})
        
        self.nodes_ex = nest.Create("adapt_lif",self.NE)
        self.nodes_in = nest.Create("adapt_lif",self.NI)
        self.noise    = nest.Create("poisson_generator")
        self.espikes  = nest.Create("spike_detector")
        if self.record_vol:
            self.voltmeter = nest.Create("multimeter")
        #self.ispikes  = nest.Create("spike_detector")
        self.nodes_al = self.nodes_ex +self.nodes_in
        nest.SetStatus(self.espikes,[{
                                #i"label": self.simulation,
                                 "withtime": True,
                                 "withgid": True,#%(n_i_),
                                 "to_file": True}])
            
        nest.SetStatus(self.nodes_al, "w",np.random.normal(0,0.5,size = len(self.nodes_al)))#np.ones(len(self.nodes_al))*0.5)    
#         nest.SetStatus(self.nodes_al, "w", np.ones(len(self.nodes_al))*0.5)    
        if self.record_vol:
            print('record from V_abs')
            nest.SetStatus(self.voltmeter,[{"label": self.simulation+'_voltage',
                                    'record_from': ['w','V_abs'],
                                    'interval':1.,
                                     "withtime": True,
                                     "withgid": True,#%(n_i_),
                                     "to_file": True,
                                      "to_memory":False}])

        self.built = True
    def connect(self):
#         print(nest.GetStatus([1],'C_m'))
        """Connect nodes"""
        if self.built ==False:
            raise BuildError('Build the network first')
        nest.CopyModel("static_synapse","excitatory",{"weight":self.J_ex})
        nest.CopyModel("static_synapse","inhibitory",{"weight":self.J_in})
        nest.CopyModel("static_synapse","external",{"weight":self.J_ext})

        self.syn_dict = {'model': 'excitatory',
                'delay':self.delay}
        self.syn_dict_in = {'model': 'inhibitory',
                'delay': self.delay}
        self.syn_dict_ext = {'model': 'external',
                'delay': self.delay}
        
        nest.Connect(self.noise,self.nodes_ex, syn_spec=self.syn_dict_ext)
#         if self.i_noise_off==False:
#             print('in noise connected')
        nest.Connect(self.noise,self.nodes_in, syn_spec=self.syn_dict_ext)

        nest.Connect(self.nodes_al[:self.N_rec],self.espikes, syn_spec=self.syn_dict)
        if self.record_vol:
            nest.Connect(self.voltmeter, self.nodes_al[1:100])#self.N_rec
#         print("Connecting network")

#         print("Excitatory connections")
        self.conn_params_ex = {'rule': 'fixed_indegree', 'indegree': self.KE}
        nest.Connect(self.nodes_ex, self.nodes_ex+self.nodes_in,
                     self.conn_params_ex, syn_spec =self.syn_dict)

#         print("Inhibitory connections")
        self.conn_params_in = {'rule': 'fixed_indegree', 'indegree': self.KI}
        nest.Connect(self.nodes_in, self.nodes_ex+self.nodes_in,
                     self.conn_params_in, syn_spec =self.syn_dict_in)

    def run(self,stim=0,stim_eta = 0.5,stim_tim = 5000,n_trials = 1):
        """run the simulation
        Chunk - to prevent from running unnecessary
        stim - with extra Poiss inoput"""
#         print("Simulating")
        # Simulate in chunk and check a special condition in between
        # Thus we avoid unwanted condition
        if self.chunk:
            self.chunk_times = np.arange(0,self.simtime+1,self.chunk_size)[1::]
            if len(self.chunk_times) != int(self.simtime/self.chunk_size):
                raise ChunkError('check the simulation length')
            for ch in self.chunk_times:
                nest.Simulate(self.chunk_size)
                self.events_ex = nest.GetStatus(self.espikes,"n_events")[0]
                t = nest.GetKernelStatus('time')
                self.rate_ex   = self.events_ex/t*1000.0/self.N_rec
                if self.rate_ex>50.0:
                    print('Rate is too high. Finishing simulation at %s ms'%(self.chunk_size))
                    self.simtime =t#self.chunk_size
                    break
                    
        elif stim==1:
            self.nu_th= self.theta / (self.J * self.KE * self.tauMem)## (theta * CMem) / (J_ex * CE * exp(1) * tauMem * tauSyn)
            self.new_nu_ex = stim_eta * self.nu_th
            self.new_p_rate = 1000.0 * self.new_nu_ex  * self.K
            
            
            self.add_noise    = nest.Create("poisson_generator")
            nest.Simulate(self.simtime)
            nest.SetStatus(self.add_noise,{"rate":self.new_p_rate})
            nest.Connect(self.add_noise,self.nodes_ex[:100], syn_spec=self.syn_dict)
            nest.Connect(self.add_noise,self.nodes_in[:100], syn_spec=self.syn_dict)
            for trial in range(n_trials):   
                nest.SetStatus(self.add_noise,{"rate":self.new_p_rate})
                nest.Simulate(stim_tim)
                nest.SetStatus(self.add_noise,{"rate":0.0})
                nest.Simulate(10000)
        else:
            nest.Simulate(self.simtime)
        self.events_ex = nest.GetStatus(self.espikes,"n_events")[0]
        #self.events_in = nest.GetStatus(self.ispikes,"n_events")[0]

        self.rate_ex   = self.events_ex/self.simtime*1000.0/self.N_rec
        #self.rate_in   = self.events_in/self.simtime*1000.0/self.N_rec


        self.num_synapses = (nest.GetDefaults("excitatory")["num_connections"] +
                       nest.GetDefaults("inhibitory")["num_connections"]+ 
                       nest.GetDefaults("external")["num_connections"])

        if self.verbose:
            print("Brunel network simulation (Python)")
            print("Number of neurons : {0}".format(self.N))
            print("Number of synapses: {0}".format(self.num_synapses))
            print("       Exitatory  : {0}".format(int(self.KE *self.N)
                                                   +self.N))
            print("       Inhibitory : {0}".format(int(self.KI * self.N)))
            print("Rate   : %.2f Hz" % self.rate_ex)
            #print("Inhibitory rate   : %.2f Hz" % self.rate_in)
            print("Simulation time   : %.2f s" % self.simtime)

    def get_Nin_deg(self):
        self.out_deg = np.zeros(self.N_rec)
        #out_deg = []#
        if self.in_deg_saved == False:
            for indx,i in enumerate(np.arange(1,self.N_rec+1)):
                conn = nest.GetConnections(target=[i])
                self.out_deg[indx] = len(nest.GetStatus(conn))
                #out_deg.append(nest.GetStatus(conn))
            with h5py.File(self.data_path+'/'+self.simulation+'-in_deg','w') as f:
                f['in_deg'] = self.in_deg
            print('saved as hdf5')
            self.in_deg_saved = True
        else:
            with h5py.File(self.data_path+'/'+self.simulation+'-in_deg','r') as f:
                self.in_deg=np.array(f['in_deg']) 
        return self.in_deg 

    def get_Nout_deg(self):
        self.out_deg = np.zeros(self.N_rec)
        #out_deg = []#
        if self.out_deg_saved == False:
            for indx,i in enumerate(np.arange(1,self.N_rec+1)):
                conn = nest.GetConnections(source=[i])
                self.out_deg[indx] = len(nest.GetStatus(conn))
                #out_deg.append(nest.GetStatus(conn))
            with h5py.File(self.data_path+'/'+self.simulation+'-out_deg','w') as f:
                f['out_deg'] = self.out_deg
            print('saved as hdf5')
            self.out_deg_saved = True
        else:
            with h5py.File(self.data_path+'/'+simulation+'-out_deg','r') as f:
                self.out_deg=np.array(f['out_deg']) 
        return self.out_deg 
    def get_connectivity(self):
        self.conn = nest.GetConnections()#source = range(1,self.N_rec+1)
        self.conn = np.array(self.conn)[:,0:2]
        with h5py.File(self.data_path+'/'+self.simulation+'-conn','w') as f:
            f['conn'] = self.conn
        print('saved as hdf5')
        return self.conn


    def conc_files(self):
        """ 
        Helper to concatinate the output files
        (Takes a lot of time; not recommended)
        Args:
        """

        import fileinput
        import glob

        file_list = glob.glob(self.data_path+'/'+self.simulation+'*')

        with open(self.data_path+'/'+self.simulation+'-all.gdf', 'w') as file:
                input_lines = fileinput.input(file_list)
                file.writelines(input_lines)
        print('files concatinated')

        #filenames = [mypath +i for i in listdir(self.data_path) if simulation in i and 'gdf' in i]
        #with open(self.data_path+'/'+self.simuation+'-all.gdf', 'w') as outfile:
        #    for fname in filenames:
        #        with open(fname) as infile:
        #            for line in infile:
        #                outfile.write(line)

        
    
class TimeSep_exp(adLIFNet):
    def __init__(self, params):
        # super(TimeSep_exp, self).__init__(params)
        self.w_init = params['w_init']
        self.mu = params['mu']
        self.sigma=params['sigma']
        # 
    def build(self):
        #Initialization of the parameters of the integrate and fire neuron and the synapses. The parameter of the neuron are stored in a dictionary.
        self.neuron_params = {'C_m':self.CMem,
                              'tau_m':self.tauMem,
                              't_ref':self.t_ref,
                              'E_L': 0.0,
                              'V_reset': self.V_res,
                              'V_abs': self.V_m,
                              'Theta':self.theta,
                              'with_refr_input':False,
                              'tau_w':self.tau_w,
                              'a':self.a,
                              'b':self.b}

        if self.verbose:
            print('Check Neurons params \n')
            print(self.neuron_params)


        self.J_ex  =self.J       # amplitude of excitatory postsynaptic potential
        self.J_in  = -self.g*self.J_ex # amplitude of inhibitory postsynaptic potential
        if self.verbose:
            print('J_i = %s'%(self.J_in))

        #Poisson Rate
        if self.eta>0:
            
            self.nu_th =self.theta / (self.J * self.KE * self.tauMem)# (theta * CMem) / (J_ex * CE * exp(1) * tauMem * tauSyn)
            self.nu_ex = self.eta * self.nu_th
            self.p_rate = 1000.0 * self.nu_ex * self.KE
        else:
            self.p_rate =self.p_rate
            
        print('p_rate')    
        print(self.p_rate)        
        
        nest.SetDefaults("adapt_lif", self.neuron_params)
        nest.SetDefaults("poisson_generator",{"rate":self.p_rate})
        
        self.nodes_ex = nest.Create("adapt_lif",self.NE)
        self.nodes_in = nest.Create("adapt_lif",self.NI)
        self.noise    = nest.Create("poisson_generator")
        self.espikes  = nest.Create("spike_detector")
        if self.record_vol:
            self.voltmeter = nest.Create("multimeter")
        #self.ispikes  = nest.Create("spike_detector")
        self.nodes_al = self.nodes_ex +self.nodes_in
        nest.SetStatus(self.espikes,[{"label": self.simulation,
                                 "withtime": True,
                                 "withgid": True,#%(n_i_),
                                 "to_file": True}])
        self.vinit = np.random.normal(self.mu,self.sigma,len(self.nodes_al))
        
        print('Initialize random Vm with:')
        print(self.mu,self.sigma)
        
        nest.SetStatus(self.nodes_al, "V_abs", self.vinit)
        nest.SetStatus(self.nodes_al, "w", np.ones(len(self.nodes_al))*self.w_init)
        if self.record_vol:
            print('record from V_abs')
            nest.SetStatus(self.voltmeter,[{"label": self.simulation+'_voltage',
                                    'record_from': ['w','V_abs'],
                                    'interval':self.dt,
                                     "withtime": True,
                                     "withgid": True,#%(n_i_),
                                     "to_file": True}])

        self.built = True

#         nest.SetStatus(self.nodes_ex, [{'V_m':np.random.normal(0,1,self.NE)}])
#         nest.SetStatus(self.nodes_in, [{'V_m':np.random.normal(0,1,self.NI)}])

class fixedW_adLIFNet(adLIFNet):
    def __init__(self, params):
        super(fixedW_adLIFNet, self).__init__(params)
        self.fixed_w = params['fixed_w']
        self.mu = params['mu']
        self.sigma=params['sigma']
        
    def build(self):
        #Initialization of the parameters of the integrate and fire neuron and the synapses. The parameter of the neuron are stored in a dictionary.
        self.neuron_params = {'C_m':self.CMem,
                              'tau_m':self.tauMem,
                              't_ref':self.t_ref,
                              'E_L': 0.0,
                              'V_reset': self.V_res,
                              'V_m': self.V_m,
                              'V_th':self.theta,
                              'I_e':self.fixed_w,
                              'refractory_input':False}
                             #}
        if self.verbose:
            print('Check Neurons params \n')
            print(self.neuron_params)


        self.J_ex  =self.J       # amplitude of excitatory postsynaptic potential
        self.J_in  = -self.g*self.J_ex # amplitude of inhibitory postsynaptic potential
        if self.verbose:
            print('J_i = %s'%(self.J_in))

        #Poisson Rate
        if self.eta>0:
            
            self.nu_th =self.theta / (self.J * self.KE * self.tauMem)# (theta * CMem) / (J_ex * CE * exp(1) * tauMem * tauSyn)
            self.nu_ex = self.eta * self.nu_th
            self.p_rate = 1000.0 * self.nu_ex * self.KE
        else:
            self.p_rate =self.p_rate
            
        print('p_rate')    
        print(self.p_rate)        
        
        nest.SetDefaults("iaf_psc_delta", self.neuron_params)
        nest.SetDefaults("poisson_generator",{"rate":self.p_rate})
        
        self.nodes_ex = nest.Create("iaf_psc_delta",self.NE)
        self.nodes_in = nest.Create("iaf_psc_delta",self.NI)
        self.noise    = nest.Create("poisson_generator")
        self.espikes  = nest.Create("spike_detector")
        
        if self.record_vol:
            self.voltmeter = nest.Create("multimeter")
        #self.ispikes  = nest.Create("spike_detector")
        self.nodes_al = self.nodes_ex +self.nodes_in
        nest.SetStatus(self.espikes,[{"label": self.simulation,
                                 "withtime": True,
                                 "withgid": True,#%(n_i_),
                                 "to_file": True}])
        self.vinit = np.random.normal(self.mu,self.sigma,len(self.nodes_al))
        print('Initialize random Vm with:')
        print(self.mu,self.sigma)
        
        nest.SetStatus(self.nodes_al, "V_m", self.vinit)
#         nest.SetStatus(self.nodes_ex, [{'V_m':np.random.normal(0,1,self.NE)}])
#         nest.SetStatus(self.nodes_in, [{'V_m':np.random.normal(0,1,self.NI)}])
        
        if self.record_vol:
            print('record from V_abs')
            nest.SetStatus(self.voltmeter,[{"label": self.simulation+'_voltage',
                                    'record_from': ['w','V_abs'],
                                    'interval':self.dt,
                                     "withtime": True,
                                     "withgid": True,#%(n_i_),
                                     "to_file": True}])

        self.built = True

class meta_brunel(object):
    """ Network of LIF neurons. 
    inspired by Gewaltig et al. 2012"""
    threads = 10
    built = False
    connected = False
    dt      = float(0.001)    # the resolution in ms for the clock-based
    CMem =1.0
    V_m = 0.0
    V_res = 10.0#10
    t_ref = 2.0
    theta = 20.0

    def __init__(self,
                simulation = 'test',
                directory = 'test',
                g=7, # inhibitory strenght
                eta = 3.5, #Poisson rate ratio
                d=[3.5], # synaptic delay
                J=0.1, #synaptic strength
                NE =80, # fraction of inh neurons
                NI =20,
                N_rec = 100,
                epsilon = 0.1,
                tauMem= 20.0,
                simtime=1000,
                master_seed = 1000,
                verbose = True,
                chunk = False,
                chunk_size = 10,
                voltage= False):
        """Initialize the simulation , setup data directory"""
        self.name=self.__class__.__name__
        self.data_path=directory
        self.simulation = simulation
        nest.ResetKernel()
        #if verbose ==False:
        #    nest.set_verbosity("M_ALL")
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            print('”Writing data to :'+self.data_path)
        
        nest.SetKernelStatus({'resolution': self.dt, 'print_time': True,
                              'overwrite_files':True,
                              'local_num_threads':self.threads,
                              'data_path': self.data_path})
        self.g = g
        self.eta = eta
        self.delay = d
        self.J = J
        self.NE =int(NE)
        self.NI = int(NI)
        self.N_neurons = NE + NI
        self.epsilon = epsilon
        self.CE    = int(epsilon*NE) # number of excitatory synapses per neuron
        self.CI    = int(epsilon*NI) # number of inhibitory synapses per neuron  
        self.C_tot = int(self.CI+self.CE)
        self.N_rec = N_rec
        self.tauMem = tauMem
        self.simtime = simtime
        self.verbose = verbose
        self.chunk = chunk
        self.chunk_size = chunk_size
        self.record_vol = voltage
        #Some statistics
        self.in_deg_saved = False
        self.out_deg_saved = False
        if verbose:
            #neurons output
            print('Number of neurons (E/I):')
            print(NE,NI)
            print('Connection Probability: %s'%(epsilon))

        #Set the RNG seeds for each thread
        self.master_seed = master_seed#2000#1000
        self.n_vp = nest.GetKernelStatus('total_num_virtual_procs')
#         print(self.n_vp)
        self.msdrange1 = range(self.master_seed, self.master_seed+self.n_vp)
        self.pyrngs = [ np.random.RandomState(s) for s in self.msdrange1]
        self.msdrange2 = range(self.master_seed+self.n_vp+1, self.master_seed+1+2*self.n_vp)
        nest.SetKernelStatus({'grng_seed': self.master_seed+self.n_vp,
                              'rng_seeds' :self.msdrange2}) 
        
    def build(self):
        
        #Initialization of the parameters of the integrate and fire neuron and the synapses. The parameter of the neuron are stored in a dictionary.
        self.neuron_params = {"C_m":self.CMem,
                         "tau_m":self.tauMem,
                         "t_ref":self.t_ref,
                         "E_L": 0.0,
                         "V_reset": self.V_res,
                         "V_m": self.V_m,
                         "V_th":self.theta,
                         "refractory_input": False}
        if self.verbose:
            print(self.neuron_params)


        self.J_ex  =self.J       # amplitude of excitatory postsynaptic potential
        self.J_in  = -self.g*self.J_ex # amplitude of inhibitory postsynaptic potential
        if self.verbose:
            print('J_i = %s'%(self.J_in))

        #Poisson Rate
        self.nu_th =self.theta / (self.J * self.CE * self.tauMem)# (theta * CMem) / (J_ex * CE * exp(1) * tauMem * tauSyn)
        self.nu_ex = self.eta * self.nu_th
        self.p_rate = 1000.0 * self.nu_ex * self.CE
#         print(self.p_rate, self.nu_ex)        
        #print(self.p_rate, self.nu_ex)        
        nest.SetDefaults("iaf_psc_delta", self.neuron_params)
        nest.SetDefaults("poisson_generator",{"rate":self.p_rate})
        self.nodes_ex = nest.Create("iaf_psc_delta",self.NE)
        self.nodes_in = nest.Create("iaf_psc_delta",self.NI)
        self.noise    = nest.Create("poisson_generator")
        self.espikes  = nest.Create("spike_detector")
        if self.record_vol:
            self.voltmeter = nest.Create("multimeter")
        #self.ispikes  = nest.Create("spike_detector")
        self.nodes_al = self.nodes_ex +self.nodes_in
        nest.SetStatus(self.espikes,[{"label": self.simulation,
                                 "withtime": True,
                                 "withgid": True,#%(n_i_),
                                 "to_file": True}])
        if self.record_vol:
            nest.SetStatus(self.voltmeter,[{"label": self.simulation+'_voltage',
                                    'record_from': ['V_m'],
                                    'interval':self.dt,
                                     "withtime": True,
                                     "withgid": True,#%(n_i_),
                                     "to_file": True}])
        #nest.SetStatus(self.ispikes,[{"label": "brunel-py-in",
        #                         "withtime": True,
        #                         "withgid": True}])#%(n_i_)"to_file": True
        self.built = True
    def connect(self):
        """Connect nodes"""
        if self.built ==False:
            raise BuildError('Build the network first')

        nest.CopyModel("static_synapse","excitatory",{"weight":self.J_ex})
        nest.CopyModel("static_synapse","inhibitory",{"weight":self.J_in})

        if len(self.delay)<2:
            self.delay = self.delay[0]
            self.syn_dict = {'model': 'excitatory',
                    'delay':self.delay,
                   }
            self.syn_dict_in = {'model': 'inhibitory',
                    'delay':self.delay,
                   }
        else:
            self.d_min = self.delay[0]#3.5
            self.d_max = self.delay[1]#5.5
            self.syn_dict = {'model': 'excitatory',
                    'delay':{'distribution': 'uniform', 'low': self.d_min,
                             'high': self.d_max},
                   }
            #print('uniform delay')

            self.syn_dict_in = {'model': 'inhibitory',
                    'delay': {'distribution': 'uniform', 'low': self.d_min,
                              'high': self.d_max},
                   }
        nest.Connect(self.noise,self.nodes_ex, syn_spec=self.syn_dict)
        nest.Connect(self.noise,self.nodes_in, syn_spec=self.syn_dict)

        nest.Connect(self.nodes_al[:self.N_rec],self.espikes, syn_spec=self.syn_dict)
        #nest.Connect(self.nodes_in[:self.N_rec] ,self.ispikes, syn_spec=self.syn_dict)
        if self.record_vol:
            nest.Connect(self.voltmeter, self.nodes_al[:self.N_rec])


        self.conn_params_ex = {'rule': 'fixed_indegree', 'indegree': self.CE}
        #nest.Connect(self.nodes_ex, self.nodes_ex+self.nodes_in,
        #             self.conn_params_ex, syn_spec =self.syn_dict)


        self.conn_params_in = {'rule': 'fixed_indegree', 'indegree': self.CI}
        #nest.Connect(self.nodes_in, self.nodes_ex+self.nodes_in,
        #             self.conn_params_in, syn_spec =self.syn_dict_in)

    def run(self,stim=0,stim_eta = 0.5,stim_tim = 5000,n_trials = 1):
        """run the simulation
        Chunk - to prevent from running unnecessary
        stim - with extra Poiss inoput"""
        # New option to simulate in chunk and check a special condition in between
        # Thus we avoid unwanted condition
        if self.chunk:
            self.chunk_times = np.arange(0,self.simtime+1,self.chunk_size)[1::]
            if len(self.chunk_times) != int(self.simtime/self.chunk_size):
               raise ChunkError('check the simulation length') 
            for ch in self.chunk_times: 
                nest.Simulate(self.chunk_size)
                self.events_ex = nest.GetStatus(self.espikes,"n_events")[0]
                self.rate_ex   = self.events_ex/(self.chunk_size*1000.0)/self.N_rec
                if self.rate_ex>60.0:
                    print('Rate is too high. Finishing simulation at %s ms'%(self.ch))
                    self.simtime = self.chunk_size
                    break
        elif stim==1:
            self.nu_th= self.theta / (self.J * self.CE * self.tauMem)## (theta * CMem) / (J_ex * CE * exp(1) * tauMem * tauSyn)
            self.new_nu_ex = stim_eta * self.nu_th
            self.new_p_rate = 1000.0 * self.new_nu_ex  * self.CE
            
            
            self.add_noise    = nest.Create("poisson_generator")
            nest.Simulate(self.simtime)
            nest.SetStatus(self.add_noise,{"rate":self.new_p_rate})
            nest.Connect(self.add_noise,self.nodes_ex[:100], syn_spec=self.syn_dict)
            nest.Connect(self.add_noise,self.nodes_in[:100], syn_spec=self.syn_dict)
            for trial in range(n_trials):   
                nest.SetStatus(self.add_noise,{"rate":self.new_p_rate})
                nest.Simulate(stim_tim)
                nest.SetStatus(self.add_noise,{"rate":0.0})
                nest.Simulate(10000)
        else:
            nest.Simulate(self.simtime)
        self.events_ex = nest.GetStatus(self.espikes,"n_events")[0]
        #self.events_in = nest.GetStatus(self.ispikes,"n_events")[0]

        self.rate_ex   = self.events_ex/self.simtime*1000.0/self.N_rec
        #self.rate_in   = self.events_in/self.simtime*1000.0/self.N_rec


        self.num_synapses = (nest.GetDefaults("excitatory")["num_connections"] +
                       nest.GetDefaults("inhibitory")["num_connections"])

        if self.verbose:
            print("Brunel network simulation (Python)")
            print("Number of neurons : {0}".format(self.N_neurons))
            print("Number of synapses: {0}".format(self.num_synapses))
            print("       Exitatory  : {0}".format(int(self.CE *self.N_neurons)
                                                   +self.N_neurons))
            print("       Inhibitory : {0}".format(int(self.CI * self.N_neurons)))
            print("Rate   : %.2f Hz" % self.rate_ex)
            #print("Inhibitory rate   : %.2f Hz" % self.rate_in)
            print("Simulation time   : %.2f s" % self.simtime)

    def get_Nin_deg(self):
        self.in_deg = np.zeros(self.N_rec)
        #out_deg = []#
        if self.in_deg_saved == False:
            for indx,i in enumerate(np.arange(1,self.N_rec+1)):
                conn = nest.GetConnections(target=[i])
                self.out_deg[indx] = len(nest.GetStatus(conn))
                #out_deg.append(nest.GetStatus(conn))
            with h5py.File(self.data_path+'/'+self.simulation+'-in_deg','w') as f:
                f['in_deg'] = self.in_deg
            print('saved as hdf5')
            self.in_deg_saved = True
        else:
            with h5py.File(self.data_path+'/'+self.simulation+'-in_deg','r') as f:
                self.in_deg=np.array(f['in_deg']) 
        return self.in_deg 

    def get_Nout_deg(self):
        self.out_deg = np.zeros(self.N_rec)
        #out_deg = []#
        if self.out_deg_saved == False:
            for indx,i in enumerate(np.arange(1,self.N_rec+1)):
                conn = nest.GetConnections(source=[i])
                self.out_deg[indx] = len(nest.GetStatus(conn))
                #out_deg.append(nest.GetStatus(conn))
            with h5py.File(self.data_path+'/'+self.simulation+'-out_deg','w') as f:
                f['out_deg'] = self.out_deg
            print('saved as hdf5')
            self.out_deg_saved = True
        else:
            with h5py.File(self.data_path+'/'+simulation+'-out_deg','r') as f:
                self.out_deg=np.array(f['out_deg']) 
        return self.out_deg 
    def get_connectivity(self):
        self.conn = nest.GetConnections()#source = range(1,self.N_rec+1)
        self.conn = np.array(self.conn)[:,0:2]
        with h5py.File(self.data_path+'/'+self.simulation+'-conn','w') as f:
            f['conn'] = self.conn
        print('saved as hdf5')
        return self.conn

