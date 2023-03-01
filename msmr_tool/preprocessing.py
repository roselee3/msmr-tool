""" 
Methods to load and interpolate experimental data from different galvanostatic cyclers.
"""

import math
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter as sf

def load_experiment_data(filepath, cycler, cycle_num, charge=True):

    """
    Loads the voltage and capacity from constant-current/galvanostatic experiments.

    Parameters:

    filepath: (str) Filepath to the slow-scan data experimental data
    cycler: (str) Name of the galvanostatic cycler that was used in the experiment.
    cycle_num: (int) Number of the cycle to be modelled. 
    charge: (boolean) default = True. If False, the voltage and currents will be reversed so that
    it looks similar to the capacity-voltage profile of the charge step.

    Returns:

    exp_voltage (data type): Cell voltage from experimental data. Units of Volts
    exp_capacity (data type): Cell capacity from experimental data. Units of mAh

    """
    data = pd.read_csv(filepath)
    
    exp_capacity = np.array([])
    exp_voltage = np.array([])
    
    if cycler == 'Victor':
        # victor had files for each individual charge/discharge step
        # so no need to sort
        exp_capacity = np.array(data['Capacity(Ah)'])
        exp_voltage = np.array(data['Voltage(V)'])
        
    elif cycler == 'Arbin':
        # this is for Martin's data, i dont remember if it was pretreated
        for i in range(data['Cycle_Index'].size):
            if data['Cycle_Index'][i] != cycle_num:
                continue
            else:
                exp_capacity = np.append(exp_capacity, data['Charge_Capacity(Ah)'][i])
                exp_voltage = np.append(exp_voltage, data['Voltage(V)'][i])
            
    elif cycler == 'BioLogic':
        # this is for Luis's pretreated data (pretreated by me)
        for i in range(data['Cycle'].size):
            if data['Cycle'][i] != cycle_num:
                continue
            else:
                exp_capacity = np.append(exp_capacity, data['Capacity (mAh)'][i])
                exp_voltage = np.append(exp_voltage, data['Voltage (V)'][i])
   
    elif cycler == 'Other':
        # i'm thinking the scientists can pre-treat their own data
        # col 0: cycle number (if multiple cycles)
        # col 1: voltage
        # col 2: capacity
        pass
    
    if charge == False:
        exp_voltage = np.flip(exp_voltage)
        
    return exp_capacity, exp_voltage

def clean_exp_data(exp_voltage, exp_capacity, constant_current, timestep, interp_voltage_range, sf_window_length):
    """ 
    This function will...
    
    1. Calculate the differential voltage from the experimental data using a 
    Savitzy-Golay filter to smooth out the derivative.
    
    2. Interpolate the experimental capacity and differential voltage 
    given a user-specified voltage range 
    
    Inputs:
    
    exp_voltage: 
    exp_capacity: 
    constant_current: (float) The constant-current value in which the experiment is using in Amps
    timestep: (float or int) Time step in between data collection in seconds
    sf_window_length: (int) Window length for the Savitzy Golay Filter (must be an odd number)
    interpolated_voltage_range: (tuple) Interpolated voltage range to be selected for optimizing
    
    
    Returns:
    
    dudq: Differential voltage
    data_cap_interp: Interpolated cell capacity that corresponds with the interpolating voltage window
    data_dudq_interp: Interpolated differential voltage that corresponds with the interpolating voltage window
    """
    
    # Calculating the derivative using a Savitzy-Golay Filter
    dudt = sf(exp_voltage, window_length = sf_window_length, deriv = 1, delta = timestep, polyorder = 3) 
    dudq = dudt / (constant_current/3600)
    
    # Interpolation
    f_data_cap_interp = interp1d(exp_voltage, exp_capacity)
    f_data_dudq_interp = interp1d(exp_voltage, dudq)
    
    exp_cap_interp = f_data_cap_interp(interp_voltage_range)
    exp_dudq_interp = f_data_dudq_interp(interp_voltage_range)
    
    return dudq, exp_cap_interp, exp_dudq_interp