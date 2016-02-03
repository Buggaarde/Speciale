#from pylab import *
#from scipy import *

#Custom functions
from tools import *

class two_way_storage:
    """Cyclic storage that can be charged and discharged at will. The difference between first and last hour is equal to the change during the first hour."""

    def __init__(self,volume=1,P_in=1,P_out=1,level=np.zeros(10),prefilling=None,median_level=None,SoC_0=0):
        
        ## Set the storage volume and initialize the storage filling level time series.
        self.volume = volume
        self.level = level ## NOTE: level should perhaps be removed from the input variables.
        self.level[-1] = SoC_0 ## Set the initital state of charge.
        
        if prefilling==None:
            self.prefilling = np.zeros_like(level)
        else:
            self.prefilling = prefilling
        
        ## Set the installed power in and output capacities. If None the capacities are assumed to be unlimited.
        ## The capacity can be constant in time or vary.
        if P_in==None:
            self.power_cap_in = NaN*np.ones_like(level)
        else:
            if np.size(P_in)==1:
                self.power_cap_in = P_in*np.ones_like(level)
            else:
                self.power_cap_in = np.array(P_in,ndmin=1)
        
        if P_out==None:
            self.power_cap_out = NaN*np.ones_like(level)
        else:
            if np.size(P_out)==1:
                self.power_cap_out = P_out*np.ones_like(level)
            else:
                self.power_cap_out = np.array(P_out,ndmin=1)
        
        ## Initialize internal time/index and cyclicity counter.
        self.index = 0 #Current index when iterating.
        self.cyclic = np.zeros(level.shape)
    
        ### Below the use of a target median level is setup.
        self.median_level = median_level # Set the requested median level (if any). Acceptet inputs are: None and an array like inflow.
        
        ## Calculate the charge and discharge power that will result in the median level. Check if these levels conflict with the charge and discharge power of the storage.
        ## NB! The check might not be sufficient as it doesn't take the prefilling level into account.
        ## Setup an array for keeping track of the power balance. Only if the power balance is positive is it possible to discharge more than the default_charge value.
        if self.median_level!=None:
        
            ## NOTE: The minimum level should be capped by the volume.
            self.min_level = zeros_like(self.median_level)
            
            for i in arange(len(self.median_level)):
                self.min_level[i] = -amin(self.median_level - self.median_level[i])   + 0.1*self.volume
                self.ekstra_discharge = zeros_like(self.min_level)
        
            self.default_charge = diff(concatenate([self.median_level - self.prefilling,[(median_level - self.prefilling)[0]]]))
            if amax(self.default_charge - self.power_cap_in) > 0:
                print "Warning (kdaf3): Median level requires too large charging power."
                index = find((self.default_charge - self.power_cap_in) > 0)
                self.default_charge[index] = self.power_cap_in[index]
            if amax(-self.default_charge - self.power_cap_out) > 0:
                print "Warning (lajkf4835): Median level requires too large discharging power." 
                index = find((-self.default_charge - self.power_cap_out) > 0)
                self.default_charge[index] = -self.power_cap_out[index]
            
            self.power_balance = np.zeros_like(level)
        else:
            self.default_charge = None
            self.power_balance = None
    
    def __getitem__(self,x):
        return self.level[x]
        
    def __len__(self):
        return len(self.volume)
        
    def next(self):
        """Advance the internal time by one time step. When comming to the end, the storage wraps around to the beginning."""
        self.index = np.mod(self.index+1,len(self.level))
        
    def iscyclic(self):
        """Check if the storage became cyclic in the previous time step."""
        return self.cyclic[self.index-1]
        
    def reset(self):
        """Reset the internal time and the cyclic counter. power capacities, volume, filling level and prefilling are not reset."""
        self.index = 0 #Current index when iterating.
        self.cyclic = 0*self.cyclic
        
    def get_power_in(self):
        """Maximum possible input power at current hour."""        
        
        if self.default_charge!=None:
            ## One is always allowed to fill as hard as possible. So this is just to keep track off what extra filling would have been possible.
            
            self.get_ekstra_discharge()
            
        return np.nanmin([self.power_cap_in[self.index],self.volume-self.level[self.index-1],self.volume-self.prefilling[self.index]])

    def get_power_out(self):
        """Maximum possible output power at current hour."""
        if self.default_charge!=None:
            
            self.get_ekstra_discharge()
            
            ## If the default charge is positive, the max_allowed_discharge can be positive too. This means that the storage is forced to fill! This also means that max_allowed_discharge must be smaller than the max charging power.
            max_allowed_discharge = amin([self.default_charge[self.index] - self.ekstra_discharge[self.index],self.get_power_in()])
            
            ## Compare the max_allowed_discharge, the maximum output power, the remaining level.
            ## The minus in front is due to sign convention that get_power_out() returns a positive number if one is allowed to discharge. However, in this case, it may return a negative number indicating that one has to charge.
            return -np.nanmax([max_allowed_discharge,-self.power_cap_out[self.index],-self.level[self.index-1]])
        
        else:
            return np.nanmin([self.power_cap_out[self.index],self.level[self.index-1]])
    
    def get_ekstra_discharge(self):
        
        delta = self.level[self.index-1] - self.min_level[self.index-1]
        delta_0 = self.median_level[self.index-1] - self.min_level[self.index-1]
    
        self.ekstra_discharge[self.index] = 2*self.volume/(365.*24.)*sign(delta)*(delta/delta_0)**2
        
    def get_default_charge(self):
        """Default charge at current hour (if any). Requires the target median level to be set and will return None if it is not."""
        if self.default_charge!=None:
            return self.default_charge[self.index]
        else:
            return None
                
    def charge(self,value):
        """Charge or discharge storage at current hour. Advance to next hour."""
    
        old_level = self.level[self.index]
        if self.median_level!=None:
            old_power_balance = self.power_balance[self.index]
    
        if value<0:
            ## Discharge storage
            if abs(value)<=self.get_power_out():
                self.level[self.index] = self.level[self.index-1] + value
            else:
                print "WARNING! Storage cannot supply the required power output. Output set to maximum possible. (3nsd32)"
                self.level[self.index] = self.level[self.index-1] - self.get_power_out()
        else:
            ## Charge storage
            if abs(value)<=self.get_power_in():
                self.level[self.index] = self.level[self.index-1] + value
            else:
                print "WARNING! Storage cannot absorb the required power input. Input set to maximum possible. (lkdf894)"
                self.level[self.index] = self.level[self.index-1] + self.get_power_in()
        
        if self.default_charge!=None:
            ## SLIGHTLY WRONG EXPLANAITION: If the storage level increase the power balance increase by the change. If the level decrease the power balance decrease too.
            self.power_balance[self.index] = self.power_balance[self.index-1] + (self.level[self.index]-self.level[self.index-1]) - self.default_charge[self.index]
        
        ## Check if cyclic. On the first round of iterations cyclic is set to false.                
        if self.cyclic[self.index] == 0:
            self.cyclic[self.index] = -1 #First time the value is set to false as there is no old level to compare against.
        else:
            if self.default_charge!=None:
                if abs(self.power_balance[self.index] - old_power_balance) < self.volume*1e-3:
                    self.cyclic[self.index-1] = 1
                    print self.index, "Power balance cyclic: ", old_power_balance, self.power_balance[self.index]
            elif abs(self.level[self.index] - old_level) < self.volume*1e-2:
                self.cyclic[self.index] = 1
                #print "Storage cyclic: ", old_level, self.level[self.index]
                
        ## Advance to next hour.        
        self.next()

    def get_level(self):
    
        return self.level

class one_way_storage():
    """A storage with forced filling and controllable discharge. Like a storage lake. The storage is modeled as a virtual two-way storage. It has a default maximum output power that can be regulated through charging and discharging of the virtual storage."""
    
    def __init__(self,volume=1,P_out=1,inflow=np.ones(10),median_level=None,SoC_0=0):
        
        ## Prespecified proporties and time series
        self.volume = volume # Storage volume.
        self.power_cap_out = P_out # Maximum power output capacity/ installed capacity.
        self.inflow = inflow # Inflow to the storage volume.
        
        ## Derived proporties and time series. These are filled in below.
        forced_filling = np.zeros_like(inflow) # If the inflow exceeds the output power then filling is forced until the reservoir is full.
        self.forced_overflow = np.zeros_like(inflow) # If the inflow exceeds the output capacity and the storage volume is full, overflow occurs. No power is produced from overflow.
        forced_unfilling = np.zeros_like(inflow) # If the reservoir is not empty and some and some or all output power is available, discharge at the available power is forced.
        
        ## Filling of the reservoir that cannot be avoided and fastest power output is calculated below.
        max_forced_filling = get_positive(self.inflow - self.power_cap_out) # Forced filling takes these values unless the storage is full.
        run_of_river = self.inflow - max_forced_filling # The amount of inflow that can pass directly through the turbines.
        max_forced_unfilling = self.power_cap_out - run_of_river # Available power capacity after run-of-river.
        
        ## Initialize temporal virtual two-way storage.
        virtual_two_way_storage_temp = two_way_storage(volume=self.volume,P_in=None,P_out=self.power_cap_out,level=np.zeros_like(self.inflow)) #

        ## Loop until two-way storage is cyclic.
        i=0;
        while virtual_two_way_storage_temp.iscyclic()!=True:
        
            if max_forced_filling[i]>0:
                forced_filling[i] = amin([max_forced_filling[i],virtual_two_way_storage_temp.get_power_in()])
                virtual_two_way_storage_temp.charge(forced_filling[i])
                self.forced_overflow[i] = max_forced_filling[i] - forced_filling[i]
            else:
                forced_unfilling[i] = amin([max_forced_unfilling[i],virtual_two_way_storage_temp.get_power_out()])
                virtual_two_way_storage_temp.charge(-forced_unfilling[i])

            i = np.mod(i+1,len(self.inflow)) #Update i, start over if i becomes to large.

        ## Default power output.
        self.default_output = run_of_river + forced_unfilling
        
        ### NEW
        self.virtual_two_way_storage = two_way_storage(volume=self.volume,P_in=self.default_output,P_out=self.power_cap_out - self.default_output,level=np.zeros_like(self.inflow),prefilling=copy(virtual_two_way_storage_temp.level),median_level=median_level,SoC_0=SoC_0) #
        
        # ## Update virtual storage by setting max in and output powers to reflect the available capacity.
        # self.virtual_two_way_storage.prefilling = copy(self.virtual_two_way_storage.level)
        # self.virtual_two_way_storage.power_cap_in = self.default_output  # Filling is only possible by not letting water out.
        # self.virtual_two_way_storage.power_cap_out = self.power_cap_out - self.default_output # Additional output is possible if there is available capacity left.
        # self.virtual_two_way_storage.reset() # Reset the internal clock to 0 and the cyclicity counter to its initial value(s).
                
    def get_power_in(self):
        """Maximum input power at hour i."""
        return self.virtual_two_way_storage.get_power_in()

    def get_power_out(self):
        """Maximum output power at hour i."""
        return self.virtual_two_way_storage.get_power_out()
        
    def charge(self,value):
        """Charge or discharge storage at hour i."""
        self.virtual_two_way_storage.charge(value)                
    
    def iscyclic(self):
        """Check if the storage became cyclic in the previous time step."""
        return self.virtual_two_way_storage.iscyclic()
        
    def get_level(self):
        return self.virtual_two_way_storage.get_level()
        
    def get_power_balance(self):
        return self.virtual_two_way_storage.get_power_balance()
