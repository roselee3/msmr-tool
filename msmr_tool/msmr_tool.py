import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from utilities import preprocessing
from utilities import msmr_model

sns.set_context('paper')

st.set_page_config(layout="wide")

# Sidebar with information (maybe) -------------------------------
with st.sidebar:
    st.title('MSMR Tool')
    st.text('This is totally just a placeholder!! will contain instructions later')
    
    
# 1. Upload experimental data -------------------------------------
cycler_options = ['Victor', 'Arbin', 'BioLogic', 'Other']
cycle_direction = ['Charge', 'Discharge']


with st.form(key = 'exp information'):
    data_in, data_out = st.columns(2)
    
    with data_in:
        st.header('1. Load experimental data')
        exp_data = st.file_uploader(label = 'Upload Experimental Galvanostatic Cycling Data')
        col1, col2 = st.columns(2)
        
        with col1:
           
            # User specifies the cycler
            user_cycler = st.selectbox('Select battery cycler: ', 
                                       options = cycler_options)

            # User specifies cycle number of interest
            cycle_num = st.number_input(label = 'Cycle number to analyze: ', 
                                        min_value = 1, 
                                        step = 1)

            # User specifies if this is a charge or discharge curve
            charge_or_discharge = st.selectbox(label = 'Charge or Discharge? ',
                                              options = cycle_direction)
            
            temp = st.number_input(label = 'Temperature (K): ', 
                                      min_value = 0.0, 
                                      value = 298.0,
                                      format = '%.1f')
        
        with col2:
            # User inputs constant-current value
            # Will eventually have a unit converter lol
            current = st.number_input(label = 'Constant-current value (mA): ', 
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
        
        if charge_or_discharge == 'Discharge':
            charge_check = False
        else:
            charge_check = True
 
    # eventually, i'll have a slider to adjust the sf window length

    with data_out:
        if exp_data is not None:
            #path = '../data/' + exp_data.name # change this line once we figure out file path stuff

            df = pd.read_csv(exp_data)
            exp_capacity, exp_voltage = preprocessing.load_experiment_data(filepath = df, 
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

            fig1, ax = plt.subplots(2,1, figsize = (3,6), tight_layout = True)
            #ax[0].set_title('Experimental')
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
            
            st.empty()
            st.pyplot(fig1)
        else: 
            pass
    
    data_submit = st.form_submit_button(label = 'Go!')


# 2. Make initial model -------------------------------------------

electrode_options = ['---', 'Graphite', 'Hard Carbon (Na)', 'LFP', 'NMC 622', 'Spinel LMO', 'Li metal', 'Na metal']

with st.form(key = 'make model'):
    st.header('2. Create initial MSMR model')
    # check if half or full cell
    
    model_in, model_out = st.columns(2)
    
    # feed selectbox into select_electrode
    # we eventually have to add something to remove the electrode option from the list if it's already taken lol
    with model_in:
        col3, col4 = st.columns(2)
        
        with col3:
            
#             def change_elec():
#                 if p_electrode != '---':
#                     p_electrode = msmr_model.select_electrode(p_electrode)
#                     st.dataframe(p_electrode, use_container_width = True)
                
            p_electrode = st.selectbox('Positive Electrode: ', key = 'PE',
                                       options = electrode_options)
            
            # Input PE capacity
            p_capacity = st.number_input(label = 'Electrode Capacity (mAh): ', 
                                         key = 'Positive electrode capacity',
                                         format = '%.3f')
            
            # Input PE lower lithiation limits
            p_Li_lim = st.number_input(label = 'Lower Li Limit (mAh): ', 
                                         key = 'PE lower Li lim',
                                         format = '%.3f')
            
            # Input PE lower cutoff potential
            p_V_low = st.number_input(label = 'Lower cutoff potential (V): ', 
                                         key = 'PE LCP',
                                         format = '%.3f')

            # Input PE upper cutoff potential
            p_V_up = st.number_input(label = 'Upper cutoff potential (V): ', 
                                         key = 'PE UCP',
                                         format = '%.3f')            
            if p_electrode != '---':
                p_elec_params = msmr_model.select_electrode(p_electrode)            
                st.dataframe(p_elec_params, use_container_width = True)
                p_elec_params['Xj'] = p_elec_params['Xj'] * p_capacity
                nor_pos = p_elec_params.shape[0]
            
        with col4:
            n_electrode = st.selectbox('Negative Electrode: ', key = 'NE', 
                                       options = electrode_options)    
            
            # Input NE capacity
            n_capacity = st.number_input(label = 'Electrode Capacity (mAh): ', 
                                         key = 'Negative electrode capacity',
                                         format = '%.3f')
            
            # Input NE lower lithiation limit
            n_Li_lim = st.number_input(label = 'Lower Li Limit (mAh): ', 
                                         key = 'NE lower Li lim',
                                         format = '%.3f')
            
            # Input NE lower cutoff potential
            n_V_low = st.number_input(label = 'Lower cutoff potential (V): ', 
                                         key = 'NE LCP',
                                         format = '%.3f')

            # Input NE upper cutoff potential
            n_V_up = st.number_input(label = 'Upper cutoff potential (V): ', 
                                         key = 'NE UCP',
                                         format = '%.3f')                   
            if n_electrode != '---':
                n_elec_params = msmr_model.select_electrode(n_electrode)
                st.dataframe(n_elec_params, use_container_width = True)
                n_elec_params['Xj'] = n_elec_params['Xj'] * n_capacity
                nor_neg = n_elec_params.shape[0]
            
    # Capacity multiplied to Xj to convert to extensive units        
#     p_electrode['Xj'] = p_electrode['Xj'] * p_capacity
#     n_electrode['Xj'] = n_electrode['Xj'] * n_capacity
    
    
    # obtain parameters, display on screen, 
    # show initial model vs exp data 
#     elec_params = np.append(p_electrode, n_electrode)
#     st.dataframe(elec_params)
    
    # edit parameters via st.experimental_data_editor
    # editing parameters should result in change of model
        
    # store final parameters to transfer to the fitting section
    
    model_submit = st.form_submit_button(label = 'Generate Model')

            # change usable cap to be p_cap if half-cell,
            # nominal/rated cap if full-cell

    if model_submit:
        elec_params = np.append(p_elec_params, n_elec_params)
        Q_IM, V_IM, dqdu_IM, dudq_IM = msmr_model.whole_cell(parameter_matrix = elec_params,
                                                                     temp = temp, 
                                                                     nor_pos = nor_pos, 
                                                                     nor_neg = nor_neg,
                                                                     pos_volt_range = (p_V_low, p_V_up),
                                                                     neg_volt_range = (n_V_low, n_V_up),
                                                                     pos_lower_li_limit = p_Li_lim,
                                                                     neg_lower_li_limit = n_Li_lim, 
                                                                     n_p = 1, 
                                                                     p_capacity = p_capacity,
                                                                     usable_cap = p_capacity, 
                                                                     Qj_or_Xj = 'Qj',
                                                                     all_output = True)
                
        fig2, ax = plt.subplots(2,1, figsize = (3,6), tight_layout = True)

        ax[0].set_xlabel('Capacity (mAh)')
        ax[0].set_ylabel('Potential (V vs Na/Na+)')
        ax[0].plot(np.flip(Q_IM[0]), V_IM[0], linewidth = 3)
        ax[0].set_ylim(0, 1.25)

        ax[1].set_xlabel('dudq')
        ax[1].set_ylabel('Potential (V vs Na/Na+)')
        ax[1].plot(dudq_IM[0], V_IM[0], linewidth = 3)
        ax[1].set_xlim(-20, 1)

        plt.show()
            
        with model_out:
            st.pyplot(fig2)

# 3. Fit model to data --------------------------------------------