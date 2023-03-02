"""
this is where we'll have a selection of electrodes
electrode response
full cell model
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from scipy.optimize import fmin_slsqp
from scipy.optimize import LinearConstraint
from scipy.optimize import least_squares
from scipy.stats import rv_histogram


def select_electrode():
    # user selects electrode from list
    # function pulls the parameters from a library and assigns the values automatically
    # if the electrode is not on the list
    # user will be prompted to add their own 
    # 
    return

def individual_reactions(U, U0, Xj, w, T):
    """
    Uses the MSMR model to calculate xj as a function of U, U0, Xj, w, and T 
    for a single insertion reaction.

    Parameters:

    U: (float, 1-D array) Measured potential value (V)
    U0: (float) Formal electrode potential for reaction j (V)
    Xj: (float) Fraction of intercalation sites available in reaction j, 
    relative to the total number of sites (intensive)
        Note: Xj * Q (electrode capacity) = Qj (maximum capacity of reaction j; extensive)
    w: (float) Thermodynamic ideality factor for reaction j
    T: (float) Temperature (K)

    Returns:
    xj: (1-D array) Fraction of filled intercalation sites (or capacity if extensive)
    dxjdu: (1-D array) Differential capacity


    """
    R, F = 8.314, 96485 # Gas constant, Faraday's constant
    f = F/(R*T)
    xj = Xj/(1 + np.exp(f*(U-U0)/w))
    
    try:
        dxjdu = (-Xj/w)*((f*np.exp(f*(U-U0)/w))/(1+np.exp(f*(U-U0)/w))**2)
    except OverflowError: 
        dxjdu = 0 # Approximates the value as zero in the case that an Overflow Error occurs
    
    return xj, dxjdu

def electrode_response(parameter_matrix, T, min_volt, max_volt, number_of_rxns, points=1000):
    """
    Wraps the individual solver and creates the cumulative OCV and dx/dU or dQ/dU curve. 
    The parameter matrix holds the Uo, Xj or Qj, and wj term for each of the 
    individual reactions, respectively.

    parameter_matrix: (1-D Array) Array of parameters for the MSMR model, that follow the repeating order,  
    standard electrode potential (U), lithium content or capacity (Xj or Qj), and thermodynamic factor
    (omega). parameter_matrix should be equal to the number_of_rxns * 3.
    T: (float) Temperature
    min_volt: (float) Minimum voltage for the MSMR model to solve
    max_volt: (float) Maximum voltage for the MSMR model to solve
    number_of_rxns: (int) Number of reactions
    points: (int) Number of points per volt

    """
    
    # Initialize the matrix with the first entry
    voltage = np.linspace(min_volt, max_volt, int((max_volt-min_volt)*points)+1)
    host_xj, host_dxjdu = individual_reactions(U=voltage, 
                                               U0=parameter_matrix[0], 
                                               Xj=parameter_matrix[1], 
                                               w=parameter_matrix[2], 
                                               T=T)
    
    # Add additional rows into the matrix for each separate reaction
    for i in range(1, number_of_rxns):
        row = int(i)
        xj_n, dxjdu_n = individual_reactions(U=voltage, 
                                             U0=parameter_matrix[int(0+i*3)], 
                                             Xj=parameter_matrix[int(1+i*3)], 
                                             w=parameter_matrix[int(2+i*3)], 
                                             T=T)

        host_xj = np.vstack((host_xj, xj_n))
        host_dxjdu = np.vstack((host_dxjdu, dxjdu_n))
    
    host_xj_sum = np.sum(host_xj, axis = 0)
    host_dxjdu_sum = np.sum(host_dxjdu, axis = 0)
    
    return voltage, host_xj_sum, host_dxjdu_sum

def whole_cell(parameter_matrix,
               temp, nor_pos, nor_neg, 
               pos_volt_range, neg_volt_range,
               pos_lower_li_limit, neg_lower_li_limit, 
               n_p, p_capacity, usable_cap, Qj_or_Xj,
               all_output = False):
    
    """ 
    Uses the MSMR model to generate a whole cell response and yields capacity, voltage, and differential
    voltage data. If prompted, this will also output results for the two individual electrodes in conjunction
    with the whole-cell response.

    Parameters

    parameter_matrix: (N,) A 1-D array of all parameter values of Uj, Qj or Xj, and wj for both electrodes
    temp: (int or float) Temperature
    nor_pos: (int) Number of reactions in the positive electrode. Helps determine how many parameters in
             the parameter_matrix are those of the positive electrde.
    nor_neg: (int) Number of reactions in the negative electrode. Helps determine how many parameters in
             the parameter_matrix are those of the negative electrde.
    pos_volt_range: (tuple) The voltage range to calculate the MSMR results for the positive electrode.
    neg_volt_range: (tuple) The voltage range to calculate the MSMR results for the negative electrode.
    pos_lower_li_limit: (float) The lower capacity bound of the positive electrode (assuming partial utilization)
    neg_lower_li_limit: (float) The lower capacity bound of the negative electrode (assuming partial utilization)
    n_p: (float) The ratio of the negative electrode capacity to the positive electrode capacity in whole cells.
         Only necessary if Qj_or_Xj == Xj
    p_capacity: (float) Positive electrode capacity. Only necessary if Qj_or_Xj == Xj
    usable_cap: (float) Usable capacity, rated capacity, or capacity available within a certain voltage window.
    Qj_or_Xj: (str) Must be "Qj" or "Xj" and determines if the MSMR model is computed with intensive (Xj) or
              extensive (Qj) properties.
    all_output: (boolean) If True, returns capacity, V, and dV/dQ for whole cell and both electrodes. If False,
                returns capacity, V, and dV/dQ for just whole cell response.

    Returns

    capacity_range: Calculated capacity values (within the usable capacity) that correspond with the following outputs 
    whole_cell_volt: Calculated voltage values
    whole_cell_dqdu: Calculated differential capacity values
    whole_cell_dudq: Calculated differential voltage values

    p_capacity_range: Capacity range that the positive electrode is operating through
    n_capacity_range: Capacity range that the negative electrode is operating through
    pos_volt_interp: Positive Electrode Voltages
    neg_volt_interp: Negative Electrode Voltages
    pos_dqdu_interp: Positive Electrode Differential Capacity
    neg_dqdu_interp: Negative Electrode Differential Capacity

    """

    int_points = 1000
    
    pos_matrix = parameter_matrix[0:3*nor_pos]
    neg_matrix = parameter_matrix[3*nor_pos:]
    
    # Unpacking Variables
    p_min_volt, p_max_volt = pos_volt_range
    n_min_volt, n_max_volt = neg_volt_range
    
    if Qj_or_Xj == 'Xj':
        # Generating the individual electrode responses, where p = positive
        pv, px, pdxdu = electrode_response(pos_matrix, temp, p_min_volt, p_max_volt, nor_pos) 
        nv, nx, ndxdu = electrode_response(neg_matrix, temp, n_min_volt, n_max_volt, nor_neg)
        
        # Applying N/P ratio to convert to the usable range of the electrodes into a nominal capacity
        pq = (px)*p_capacity
        nq = (nx)*p_capacity*n_p
        
        # Converting from dxdu to dQdu (for purposes of comparing with real data)
        pdqdu = pdxdu*p_capacity
        ndqdu = ndxdu*p_capacity*n_p

        # Interpolating capacities for proper Coulomb counting between p and n electrodes. 
        # Takes the lower ends of the two ranges, and adds usable capacities to get the same Q on both sides
        p_capacity_range = np.linspace(pos_lower_li_limit*p_capacity, (pos_lower_li_limit*p_capacity)+usable_cap, int_points)
        n_capacity_range = np.linspace(neg_lower_li_limit*p_capacity*n_p, (neg_lower_li_limit*p_capacity*n_p)+usable_cap, int_points)
        capacity_range = p_capacity_range - p_capacity_range.min()
        # ^^this normalizes the capacity range 

    elif Qj_or_Xj == 'Qj':
        # Generating the individual electrode responses, where p = positive
        pv, pq, pdqdu = electrode_response(pos_matrix, temp, p_min_volt, p_max_volt, nor_pos)
        nv, nq, ndqdu = electrode_response(neg_matrix, temp, n_min_volt, n_max_volt, nor_neg)
        
        # Interpolating capacities for proper Coulomb counting between p and n electrodes. 
        # Takes the lower ends of the two ranges, and adds usable capacities to get the same Q on both sides
        p_capacity_range = np.linspace(pos_lower_li_limit, pos_lower_li_limit+usable_cap, int_points)
        n_capacity_range = np.linspace(neg_lower_li_limit, neg_lower_li_limit+usable_cap, int_points)
        capacity_range = p_capacity_range - p_capacity_range.min()

    else:
        raise ValueError('Missing input for Qj_or_Xj')
    
    # Interpolating the positive and negative electrode data to ensure the capacity data are evenly spaced
    f_pos_cap_interp = interp1d(pq, pv, fill_value='extrapolate')
    f_pos_dx_interp = interp1d(pq, pdqdu, fill_value='extrapolate')
    f_neg_cap_interp = interp1d(nq, nv, fill_value='extrapolate')
    f_neg_dx_interp = interp1d(nq, ndqdu, fill_value='extrapolate')

    pos_volt_interp = f_pos_cap_interp(p_capacity_range)
    pos_dqdu_interp = f_pos_dx_interp(p_capacity_range)
    neg_volt_interp = f_neg_cap_interp(n_capacity_range)
    neg_dqdu_interp = f_neg_dx_interp(n_capacity_range)
    
    whole_cell_volt = np.flip(pos_volt_interp) - neg_volt_interp
    whole_cell_dqdu = np.flip(pos_dqdu_interp) + neg_dqdu_interp
    whole_cell_dudq = 1/np.flip(pos_dqdu_interp) + 1/neg_dqdu_interp
    
    if all_output == False:
        return capacity_range, whole_cell_volt, whole_cell_dqdu, whole_cell_dudq
    elif all_output == True:
        return ((capacity_range, p_capacity_range, n_capacity_range), 
                (whole_cell_volt, pos_volt_interp, neg_volt_interp), 
                (whole_cell_dqdu, pos_dqdu_interp, neg_dqdu_interp),
                (whole_cell_dudq, 1/pos_dqdu_interp, 1/neg_dqdu_interp))