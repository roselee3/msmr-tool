import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from scipy.optimize import fmin_slsqp
from scipy.optimize import LinearConstraint
from scipy.optimize import least_squares
from scipy.stats import rv_histogram

from utilities import preprocessing
from utilities import msmr_model

sns.set_context('poster')


# 1. Upload experimental data -------------------------------------
cycler_options = ['Victor', 'Arbin', 'BioLogic', 'Other']
cycle_direction = ['Charge', 'Discharge']

exp_data = st.file_uploader(label = 'Upload experimental galvanostatic cycling data')

with st.form(key = 'exp information'):
    # User specifies the cycler
    user_cycler = st.selectbox('Select battery cycler: ', 
                               options = cycler_options)
    
    # User inputs constant-current value
    # Will eventually have a unit converter lol
    current = st.number_input(label = 'Constant-current value: ', 
                              min_value = 0.0,
                              step = 0.00001,
                              format = '%.5f')
    
    # User inputs time between data point collection (s)
    timestep = st.number_input(label = 'Time between data points (s): ', 
                                min_value = 0.0)
    
    # User inputs LCV and UCV
    LCV = st.number_input(label = 'Lower cutoff voltage: ',
                          format = '%.3f')
    
    UCV = st.number_input(label = 'Upper cutoff voltage: ',
                          format = '%.3f')
    
    v_range = np.linspace(LCV, UCV, 1000)
    
    # User specifies cycle number of interest
    cycle_num = st.number_input(label = 'Cycle number to analyze: ', 
                                min_value = 1, 
                                step = 1)
    
    # User specifies if this is a charge or discharge curve
    charge_or_discharge = st.selectbox(label = 'Charge or Discharge? ',
                                      options = cycle_direction)
    
    if charge_or_discharge == 'Discharge':
        charge_check = False
    else:
        charge_check = True
 
    # eventually, i'll have a slider to adjust the sf window length
    
    data_submit = st.form_submit_button(label = 'Go!')
    
with st.form(key = 'process exp data'): 
    # Experimental data is loaded 
    path = '../data/' + exp_data.name # change this line once we figure out file path stuff
    exp_capacity, exp_voltage = preprocessing.load_experiment_data(filepath = path, 
                                                                   cycler = user_cycler, 
                                                                   cycle_num = cycle_num, 
                                                                   charge = charge_check)
    
    
    # Experimental data is processed
    exp_dudq, exp_cap_interp, exp_dudq_interp = preprocessing.clean_exp_data(exp_voltage, 
                                                                             exp_capacity, 
                                                                             constant_current = current, 
                                                                             timestep = timestep, 
                                                                             interp_voltage_range = v_range, 
                                                                             sf_window_length = 99)
    
    fig1, ax = plt.subplots(1,2, figsize = (12,6), tight_layout = True)
    ax[0].set_title('Experimental')
    ax[0].set_xlabel('Capacity (mAh)')
    ax[0].set_ylabel('Potential vs Na/Na+ (V)')
    ax[0].plot(exp_capacity, exp_voltage)
    #ax[0].plot(Q_1400_interp, v_range, '--', label = 'Interpolated')
    #ax[0].legend()

    #ax[1].set_title('interpolated data')
    ax[1].set_xlabel('dudq (V/mAh)')
    ax[1].set_ylabel('Potential vs Na/Na+ (V)')
    ax[1].plot(exp_dudq, exp_voltage)

    plt.show()
    st.pyplot(fig1)
    
    data_process = st.form_submit_button(label = 'Go!')

# st.experimental_data_editor

# feed path name into load exp data




# 2. Make initial model -------------------------------------------


# 3. Fit model to data --------------------------------------------