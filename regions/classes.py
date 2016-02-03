#! /usr/bin/env python
from pylab import is_numlike
from pylab import is_string_like
import sys
from tools import *
import defaults
from configobj import ConfigObj
from validate import Validator

## Custom modules and functions
from storage import one_way_storage, two_way_storage

# specify format of config file
configspec=ConfigObj(defaults.configspec,
                     list_values=False, file_error=True, _inspec=True)

# read config
regionConfig = ConfigObj(defaults.regionspec, list_values=True, configspec=configspec)
regionConfig.validate(Validator()) # type conversion happens only here!

region_name = regionConfig['region']
optimal_alphas = regionConfig['optimal_alphas']
node_names = regionConfig['node_names']
node_filenames = regionConfig['node_filenames']
node_data_path = regionConfig['node_data_path']
node_results_path = regionConfig['node_results_path']
admittance_matrix_path = regionConfig['admittance_matrix_path']


class node:
    """ The node class loads data from the '.npz' countryfiles and stores it in these
        predefined structures.

        NOTE: the unnormalized load signal can be queried by typing
        n.load*n.mean, but we have functions for querying wind and solar.
        Type n.get_wind() or n.get_solar.

        Individual values for gamma and alpha can also be set by n.set_alpha().
        DON'T MODIFY THEM DIRECTLY IN n.gamma OR n.alpha AS THEY WONT BE
        RECALCLUATED UNLESS YOU TYPE self._update_()"""
    def __init__(self,path,fileName,ID):
        self.id = ID
        data = np.load(path + fileName)
        self.gamma = 1.0
        self.alpha = 0.7
        if data['L'].shape == (): # We have an offshore node!
            self.normwind = np.array(map(np.double,data['Gw']))
            self.load = 0 * self.normwind
            self.normsolar = 0 * self.normwind
            self.nhours=len(self.normwind)
            self.mean=1
        else:
            self.load = 1000.*np.array(map(np.double,data['L']))
            self.nhours = len(self.load)
            self.normwind = np.array(map(np.double,data['Gw']))
            self.normsolar = np.array(map(np.double,data['Gs']))
            self.mean = np.mean(self.load)
        self.balancing = np.zeros(self.nhours)
        self.curtailment = np.zeros(self.nhours)
        self.label = data['datalabel']
        self.mismatch = None
        self.colored_import = None #Set using self.set_colored_i_import()
        self.power_mix = None
        # capacity factors for wind and solar (before they were normalized)
        try:
            self.CFw = data['CFw']
            self.CFs = data['CFs']
        except:
            pass
        data.close()
        self.ESS = ESS(self.nhours)

        self.storage_default_output = None ## Default output from storage. Set by self._update_().
        self.storage_discharge = np.zeros(self.nhours) ## Positive when the storage is discharging.
        self.storage_level = None ## Set by self._update_().
        self.storagePower=np.zeros(self.nhours)
        self.storageLevel=np.zeros(self.nhours+1)
        self.solved = False ## Set to true after flow-calculation.
        self._update_()

    def _update_(self,load_nodes=False):
        if not load_nodes:
            self.storage_default_output = self.ESS.get_default_output()
            self.storage_level = self.ESS.get_level()
        self.mismatch = self.get_RES() - self.load

    def get_mismatch(self):
        """Return the local (raw) mismatch."""
        return self.mismatch

    def get_mismatch_r(self):
        """Returns the reduced mismatch. That is the local mismatch after the storage has been used."""
        return self.get_mismatch() + self.get_storage_discharge()

    def get_import(self):
        """Returns import power time series in units of MW. It includes all types of power which is imported."""
        return get_positive(-(self.get_mismatch_r() + self.get_balancing()  - self.get_curtailment()))

    def get_export(self):
        """Returns export power time series in units of MW. It includes all types of power which is exported."""
        return get_positive(self.get_mismatch_r() + self.get_balancing()  - self.get_curtailment())

    def get_localRES(self):
        """Returns the local use of RES power time series in units of MW."""
        return self.get_RES() - self.get_curtailment() - self.get_export() + self.storage_discharge

    def get_localBalancing(self):
        """Returns the local use of balancing power time series in units of MW."""
        return get_positive(-self.get_mismatch_r()) - self.get_import()

    def get_wind(self):
        """Returns wind power time series in units of MW."""
        return self.mean*self.gamma*self.alpha*self.normwind

    def get_solar(self):
        """Returns solar power time series in units of MW."""
        return self.mean*self.gamma*(1.-self.alpha)*self.normsolar

    def get_curtailment(self):
        return self.curtailment

    def get_balancing(self):
        return self.balancing

    def get_RES(self):
        return self.get_wind() + self.get_solar() + self.get_storage_default_output()

    def get_storage_default_output(self):
        return self.storage_default_output

    def get_storage_discharge(self):
        """Positive when storage is discharging, negative when it is charging. """
        return self.storage_discharge

    def set_gamma(self,gamma,operation='='):
        if operation == '=':
            self.gamma = gamma
        else:
            self.gamma *= gamma
        self._update_()

    def set_alpha(self,alpha):
        self.alpha=alpha
        self._update_()

    def get_colored_import(self):
        """ Wraper functions for R's color code."""
        if self.colored_import == None:
            imports = self.get_import()
            self.colored_import = zeros_like(self.power_mix)
            for i in arange(len(self.colored_import)):
                if i!=self.id:
                    self.colored_import[i] = imports*self.power_mix[i]/sum(self.power_mix,axis=0)
            self.colored_import[isnan(self.colored_import)]=0.

        return self.colored_import

    def add_hydro_storage_lake(self, P_out, volume, SoC_0=0, inflow=None, median_level=None):
        """ Wraper for adding a hydro storage lake to the node. Give P_out in units of mean load and volume in units of hours at P_out."""
        self.ESS.add_hydro_storage_lake(self.mean, P_out, volume, SoC_0, inflow, median_level)
        self._update_() ## Run to include default output in mismatch.

    def add_storage(self, P_in, P_out, volume, SoC_0=0):
        """ Wraper for adding a normal storage to the node. Give P_in, P_out in units of mean load and volume in units of hours at P_out."""
        self.ESS.add_storage(self.mean, P_in, P_out, volume, SoC_0)


class ESS:

    def __init__(self,nhours):
        self.nhours = nhours
        self.all_ess=[] ## Container for all initialized storage objects.
        self.all_ess_type = [] ## Type of storage.
        self.default_output = None ## Container for default output used by one_way_storage, e.g. run-of-river.

    def add_hydro_storage_lake(self, load_mean, P_out, volume, SoC_0=0, inflow=None, median_level=None):
        """Give P_out in units of mean load and volume in units of hours at P_out."""

        P_out = P_out*load_mean ## Convert P_out to real units (MW)
        volume = volume*P_out ## Convert volume to real units (MWh)
        SoC_0 = SoC_0*volume ## Convert initial state of charge (SoC) to real units (MWh)

        if inflow==None:
            inflow = P_out*np.ones(self.nhours) ## Default inflow.
        else:
            inflow = inflow*load_mean ## Convert inflow to real units (MW)

        if median_level != None:
            median_level = median_level*volume ## Convert median level to real units (MWh)

        ## Initialize the storage class
        ess = one_way_storage(volume=volume, P_out=P_out, inflow=inflow, median_level=median_level, SoC_0=SoC_0)

        ## Append the storage object to the collection of storages and update aggregated storage values.
        self.all_ess.append(ess)
        self.all_ess_type.append('one_way')
        if self.default_output==None:
            self.default_output = ess.default_output
        else:
            self.default_output += ess.default_output

    def add_storage(self,load_mean, P_in, P_out, volume, SoC_0=0):
        """Give P_in and P_out in units of mean load and volume in units of hours at P_out."""

        P_in = P_in*load_mean ## Convert P_in to real units (MW)
        P_out = P_out*load_mean ## Convert P_out to real units (MW)
        volume = volume*P_out ## Convert volume to real units (MWh)
        SoC_0 = SoC_0*volume ## Convert initial state of charge (SoC) to real units (MWh)

        ## Initialize the storage class
        ess = two_way_storage(volume=volume, P_in=P_in, P_out=P_out, level=np.zeros(self.nhours), SoC_0=SoC_0)

        ## Append the storage object to the collection of storages and update aggregated storage values.
        self.all_ess.append(ess)
        self.all_ess_type.append('two_way')

    def get_power_out(self):
        """ Return aggregated maximum discharging power. """
        P_out = 0
        if len(self.all_ess)==0:
            P_out = 1e-6 ## Minimum discharging power (required for flow solver convergens).
        else:
            for i in self.all_ess:
                P_out += i.get_power_out() ## Sum all available dischargeing power.

        return np.double(P_out)

    def get_power_in(self):
        """ Return aggregated maximum charging power. """
        P_in = 0
        if len(self.all_ess)==0:
            P_in = -1e-6 ## Minimum charging power (required for flow solver convergens).
        else:
            for i in self.all_ess:
                P_in += i.get_power_in() ## Sum all available chargeing power.

        return np.double(P_in)

    def charge(self,value):
        """ Charge/discharge storages in order of priority. """
        ## NOTE: It is important that ess_i.charge() is called for all storages in order to increment time correctly.

        if value <= 0:  ## Then we were dicsharging

            ## Discharge all storages in order of priority.
            ## NOTE: Some storages may require a minimum of charging. These should be satisfied first. This is not acounted for yet.
            for ess_i in self.all_ess:
                charge_ess_i = - np.amin([-value,ess_i.get_power_out()])
                ess_i.charge(charge_ess_i)
                value -= charge_ess_i

        else: ## Then we were charging

            ## Charge all storages in order of priority.
            for ess_i in self.all_ess:
                charge_ess_i = np.amin([value,ess_i.get_power_in()])
                ess_i.charge(charge_ess_i)
                value -= charge_ess_i

        if value>0.01: print "WARNING (k32klfsd): Charge assignment error."

    def get_default_output(self):
        """ Returns default power output (from e.g. hydro storage lake) """

        if self.default_output == None:
            return np.zeros(self.nhours)
        else:
            return self.default_output

    def get_level(self):

        level = np.zeros(self.nhours)
        for ess_i in self.all_ess:
            level = ess_i.get_level()

        return level


class Nodes:
    """ Contains a collection of Node() objects and tools for managing them.

        When called as Europe=Nodes() (or just, N=Nodes()) it builds nodes
        from the country data files. If called as N=Nodes(filename) the it looks
        for a pre-solved filename with balancing, curtailment, etc.

        For the time being, Flows are stored separately in npy files, which are
        simply opened with np.load. Nodes can be saved with the inbuilt function save_nodes()"""


    def __init__(self, admat=admittance_matrix_path, region_name=region_name, 
                 path=node_data_path, prefix='', files=node_filenames, 
                 load_path=node_results_path, load_filename=None, full_load=False,
                 alphas=optimal_alphas, gammas=1.):
        
        self.cache = []
        self.pathadmat = admat
        self.region = region_name
        self.path = path
        self.prefix = prefix
        self.files = files
        self.storage_factor = 1.0
        for i in range(len(files)):
            n=node(path,prefix + files[i],i)
            self.cache=np.append(self.cache,n)
        F=np.zeros((np.size(files),self.cache[0].nhours))

        self.set_alphas(alphas)
        self.set_gammas(gammas)

        if load_filename != None:
            self._load_nodes_(load_filename, full_load, path=load_path)
            for n in self.cache:
                n._update_()

    def __getitem__(self,x):
        return self.cache[x]

    def __len__(self):
        return len(self.cache)

    def set_gammas(self,value):
        # to change a single node's gamma, just write XX.set_gamma(yy)
        # 'value' can be a single number or a vector
        if np.size(value)==1:
            for i in self.cache: i.set_gamma(value)
        elif np.size(value)!=np.size(self.cache):
            print "Wrong gamma vector size. ", np.size(value,0)," were received, ",np.size(self.cache)," were expected."
        else:
            for i in self.cache:
                i.set_gamma(value[i.id])

    def set_alphas(self,value):
        # to change a single node's alpha, just write XX.set_gamma(yy)
        # 'value' can be a single number or a vector
        if np.size(value)==1:
            for i in self.cache: i.set_alpha(value)
        elif np.size(value)!=np.size(self.cache):
            print "Wrong gamma vector size. ", np.size(value,0)," were received, ",np.size(self.cache)," were expected."
        else:
            for i in self.cache:
                i.set_alpha(value[i.id])

    def get_P_charge(self,t):
        ppos=np.zeros(np.size(self.cache),dtype='d')
        for i in self.cache:
            ppos[i.id]= i.ESS.get_power_in()
        return np.double(ppos)

    def get_P_discharge(self,t):
        pneg=np.zeros(np.size(self.cache),dtype='d')
        for i in self.cache:
            pneg[i.id]=i.ESS.get_power_out()
        return np.double(pneg)


    def save_nodes(self,filename,path='./results/'):
        """Saves the contents of a Nodes instance to a npz file."""

        attribute = dir(self[0])
        save_str = []
        #Determine which attributes to be saved
        for attribute in dir(self[0]):
            if attribute[0]=='_':
                continue
            elif is_numlike(getattr(self[0],attribute)) or is_string_like(getattr(self[0],attribute)):
                save_str.append(attribute + '=' + 'np.array([self[i].'+attribute+' for i in np.arange(len(self))])')

        save_str.append('pathadmat=self.pathadmat')
        #Write save file
        eval('np.savez(path+filename,'+','.join(save_str)+')')

        print 'Saved nodes to file: ', path+filename
        sys.stdout.flush()

    def _load_nodes_(self,load_filename, full_load, path='./results/'):
        """Loads a Nodes instance from an npz file."""

        npzobj = np.load(path+load_filename)
        if full_load == False:
            for attribute in npzobj.files: ## this is real
                # for loading data for the individual Node-objects in the Nodes-object
                if attribute in ['balancing', 'curtailment', 'storage_level', 'solved','alpha','gamma', 'balancing_upper_bound']: ## not full load
                    for i in np.arange(len(self)):#said self earlier
                        setattr(self.cache[i],attribute,npzobj[attribute][i])

                # for loading data for the Nodes-object
                if attribute in ['pathadmat']:
                    setattr(self, attribute, str(npzobj[attribute]))

        if full_load == True:
            for attribute in npzobj.files:
                for i in np.arange(len(self)):#same here
                    setattr(self.cache[i],attribute,npzobj[attribute][i])

        for n in self.cache:
            n._update_(load_nodes=True)
        npzobj.close()
        print 'Loaded nodes from file: ', path+load_filename
        sys.stdout.flush()
