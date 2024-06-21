import numpy as np
import pandas as pd
import skrf as rf
from skrf.calibration import OpenShort, SplitTee, AdmittanceCancel
from skrf.calibration import IEEEP370_SE_NZC_2xThru
import matplotlib.pyplot as plt
import os
import datetime
from operator import itemgetter
from dataclasses import dataclass
import re
from typing import Tuple
import matplotlib.cm as cm
import itertools
import scipy
from scipy.signal import medfilt
import scipy.fft as fft
import scipy.interpolate as interp
import copy
from scipy.signal import find_peaks
from scipy.linalg import sqrtm



# Define a class to store the VNA data alongside its ascociated with the filename, device index, and state
@dataclass
class S2PFile:
    network: rf.Network
    filename: str
    wafer_number: int #wafer number 1 or 2
    dev_row: int #row of device on wafer
    dev_col: int #column of device on wafer
    state: str
    run: int #run number to identify the measurements chronological position in the list
    
    @property #generating a unique label for the device that can be used in plots to identify it
    def label(self) -> str:
        return f"run{self.run}_r{self.dev_row}c{self.dev_col}_{self.state}"

############ Start of Functions

def import_data(data_path: str):
    # Function to import the data from the VNA files and store it in a list of S2PFile objects
    # Extracts the dat and time from the s2p file, uses it to sort the files in chronological order, 
    # assigns a key number to each file based off this and then extracts information from the filename
    # about the measurement device/state/type etc and stores it in the S2PFile object
    
    # Get a list of all .s2p files in the directory
    files = [f for f in os.listdir(data_path) if f.lower().endswith('.s2p')]
    ### Initiate a list to store the filenames and their associated dates     
    file_dates = []
    # Loop over all files in the directory
    for fi in files:
        with open(os.path.join(data_path, fi), 'r') as f:
            for line in f:
                if line.startswith('! VAR DATE='):
                    # Extract date and time from the line
                    date_str = line.split('=')[1].strip()
                    # Convert to datetime object
                    date = datetime.datetime.strptime(date_str, '%d/%m/%Y %H:%M:%S')
                    # Add to list
                    file_dates.append((fi, date))
                    break
                elif line.startswith('!Date:'):
                    # Extract date and time from the line
                    date_str = line.split(': ', 1)[1].strip()
                    # Convert to datetime object
                    date = datetime.datetime.strptime(date_str, '%A, %B %d, %Y %H:%M:%S')
                    # Add to list
                    file_dates.append((fi, date))
                    break
                elif line.startswith('! Date and time:'):
                    # Extract date and time from the line
                    date_str = line.split(': ', 1)[1].strip()
                    # Convert to datetime object
                    date = datetime.datetime.strptime(date_str, '%a %b %d %H:%M:%S %Y')
                    # Add to list
                    file_dates.append((fi, date))
                    break
    # Sort the list chronologically based on the datetime
    file_dates.sort(key=itemgetter(1))
  
    
    # for f, date in file_dates:
    #     print(f, date)
        
    # Extract the filenames in chronological order
    chron_files = [f[0] for f in file_dates]

    ### Read each file and store it in a chronological list of S2PFile objects
    s2p_files = []
    run_count = 1
    for f in chron_files:
        network = rf.Network(os.path.join(data_path,f))
        #network.frequency.drop_non_monotonic_increasing() - get rid of any non-monotonic increasing frequency points but then causes array mismatch error later
        #state keywords to look for in filename, by default thru/short etc means the tapered version and it will have "notaper" in the filename if it is the straight thru
        #note that smaller words that are substrings of larger words should be placed after the larger words in the list
        keywords = ['thrunotaper', 'opennotaper','shortnotaper', 'opensignotaper', 'openverynarrow','opennarrow', 'smallform', 'fullform', 'opensig', 'thruISS','pristine', 'formed', 'thru', 'open', 'short','set','reset','linelong','line']
        state = next((x for x in keywords if x in f.lower()), None) #returns the first keyword found in the state value, stops as soon as the first keyword is found
        # Extract the row, colum and wafer numbers from the filename (e.g. wafer 1 r1_c11) and store into position variable
        wafer_number = re.findall(r'Wafer(\d)', f, re.IGNORECASE)
        r_number = re.findall(r'_r(\d{1,2})_', f, re.IGNORECASE)
        c_number = re.findall(r'_c(\d{1,2})_', f, re.IGNORECASE)
        # Check for the case where there is no state matching the keywords
        if state is not None:
            state = state.lower()
        else:
            state = None  # or some default value
        #store the network and its associated metadata in the S2PFile object grouped toegher in the s2p_files list
        if not wafer_number or not r_number or not c_number:
            s2p_files.append(S2PFile(network, f, int(0), None, None, state, run_count)) #ISS is classes as wafer zero
        else:
            s2p_files.append(S2PFile(network, f, int(wafer_number[0]), int(r_number[0]), int(c_number[0]), state.lower(),run_count))
        run_count += 1
    return s2p_files


#--------------------------------------------------------------------------------
def duplicate_check(s2p_files):
    for f in s2p_files:
        print(f.run,f.label,f.filename)
        indx = f.network.frequency.drop_non_monotonic_increasing()
        # Remove the corresponding entries from the s-parameters array
        unique_s = np.delete(f.network.s, indx, axis=0)
        unique_z0 = np.delete(f.network.z0, indx, axis=0)
        # Assign the unique s-parameters array back to the network object
        f.network.s = unique_s
        f.network.z0 = unique_z0
    return s2p_files

def calibration_OS(open, short, thru, plot_cal = False):
    # Function to generate an OpenShort De-embedding calibration and apply it to the on-wafer thru measurements
    # Inputs multiple open and short measurements and compares them to see which give the most effective de-embedding
    # Outputs the best de-embedding protocol and plots of the thru measurements before and after de-embedding for each protocol
    # Prints the error and its integral for each de-embedding protocol to help select the best de-embedding data
  
    dm = [] #initialize an empty list to store the de-embedded data
    error = [] #initialize an empty list to store the error for each de-embedding protocol
    num_colors = len(open)*len(short)*len(thru)
    colors = plt.cm.jet(np.linspace(0,1,num_colors))
    color_count = 0
    for count_o, o  in enumerate(open, start=0):
        for count_sh, sh in enumerate(short, start=0):
            cal = OpenShort(dummy_open=o.network, dummy_short=sh.network, name='OpenShort Calibration')
            dm.append(cal)
            
            total_error = 0 #initialize variable to store the total error for each open/short pair
            for count_th,th in enumerate(thru, start=0):
                        
                if plot_cal:
                    plt.figure(f'Open Short De-embedding')
                    th.network.plot_s_mag(m=1, n=0, color=colors[color_count], linestyle='dashed',label = f'Raw: {th.label}; {sh.label}; O{count_o}S{count_sh}T{count_th}')  # Plot only s21 with colorblind colormap
                    cal.deembed(th.network).plot_s_mag(m=1, n=0, color=colors[color_count], label = f'OS: {th.label}; {sh.label}; O{count_o}S{count_sh}T{count_th}')  # Plot only s21 with colorblind colormap
                #store the error for each de-embedding protocol in the respective array index
                error_value = np.abs(np.sum(cal.deembed(th.network).s[:,1,0]))+np.abs(np.sum(cal.deembed(th.network).s[:,0,1]))
                total_error += error_value #sum the error for each open/short pair over every thru device
                color_count += 1
            error.append(total_error)
    
    
    #Find the best de-embedding protocol
    print(error)
    min_index = error.index(min(error))
    max_index = error.index(max(error))

    print(f"Best OS de-embedding protocol: dm[{min_index}] = {min(error)}, worst:dm[{max_index}] = {max(error)}")
    
    #Plot the best de-embedding protocol
    if plot_cal:
        plt.figure('Best De-embedding Protocol')
        thru[0].network.plot_s_mag(m=1, n=0, color='red', linestyle='dashed',label = f'Raw')  # Plot only s21 with colorblind colormap
        dm[min_index].deembed(thru[0].network).plot_s_mag(m=1, n=0, color='green', label = f'Best OS:dm[{min_index}]')  # Plot only s21 with colorblind colormap
        dm[max_index].deembed(thru[0].network).plot_s_mag(m=1, n=0, color='blue', label = f'Worst OS:dm[{max_index}]')  # Plot only s21 with colorblind colormap
   
    #only return the best de-embedding protocol
    return dm[min_index]



def calibration_2x(thru, plot_cal = False):
    # Function to generate 2x calibration from the thru measurements (though these are strictly speaking 2x + length of DUT so it isn't perfect)
    # Inputs thru devices and outputs the best de-embedding protocol based on the error of the de-embedding
    # plots of the thru measurements before and after de-embedding for each protocol
    # Prints the error and its integral for each de-embedding protocol to help select the best de-embedding data
  
    dm = [] #initialize an empty list to store the de-embedded data
    total_error = [0.0] * len(thru) #initialize an empty list of floats to store the total error for each de-embedding protocol
    num_colors = len(thru)*(len(thru)-1) #as each thru cal, len(thru), will be applied to all other thru measurements, (len(thru)-1)
    colors = plt.cm.jet(np.linspace(0,1,num_colors))
    color_count = 0
    
    # Generate calibration objects for each thru device
    for t  in thru:
        cal = IEEEP370_SE_NZC_2xThru(dummy_2xthru = t.network, name = '2xthru')
        dm.append(cal)
    
    for count_t, th  in enumerate(thru, start=0):
        for i in range(len(dm)):
            if i == count_t:
                continue # skip this iteration of the for loop if the de-embedding protocol is being applied to the thru device it was generated from
            # apply each deembedding to the thru device except for the one it was generated from
            thru_TX = dm[i].deembed(th.network)
            #store the error for each de-embedding protocol in the respective array index
            error_value = np.abs(np.sum(thru_TX.s[:,1,0]))+np.abs(np.sum(thru_TX.s[:,0,1]))
            total_error[i] += error_value #add the error for each de-embedding protocol applied to every thru to the total error for that protocol 
                    
            if plot_cal:
                plt.figure(f'2x De-embedding')
                th.network.plot_s_mag(m=1, n=0, color=colors[color_count], linestyle='dashed',label = f'Raw: {th.label}')  # Plot only s21 with colorblind colormap
                thru_TX.plot_s_mag(m=1, n=0, color=colors[color_count], label = f'TX: {th.label}')  # Plot only s21 with colorblind colormap
                color_count += 1
            
            
    print(total_error)
    
    #Find the best de-embedding protocol
    min_index = total_error.index(min(total_error))
    max_index = total_error.index(max(total_error))

    print(f"Best TX de-embedding protocol: dm[{min_index}] = {min(total_error)}, worst:dm[{max_index}] = {max(total_error)}")
    
    #Plot the best de-embedding protocol
    if plot_cal:
        plt.figure('Best De-embedding Protocol')
        thru[0].network.plot_s_mag(m=1, n=0, color='red', linestyle='dashed',label = f'Raw')  # Plot only s21 with colorblind colormap
        dm[min_index].deembed(thru[0].network).plot_s_mag(m=1, n=0, color='green', label = f'Best TX:dm[{min_index}]')  # Plot only s21 with colorblind colormap
        dm[max_index].deembed(thru[0].network).plot_s_mag(m=1, n=0, color='blue', label = f'Worst TX:dm[{max_index}]')  # Plot only s21 with colorblind colormap
   
    #only return the best de-embedding protocol
    return dm[min_index]


def calibration_ABCD(thru, plot_cal = False):
    # Function to Use ABCD parameters to de-embed the thru measurements
    
    # For ABCD parameters that cascade: [Measured_data] = [Half_thru]*[DUT]*[Half_thru]
    # Therefore, [DUT] = [Half_thru]^-1*[Measured_data]*[Half_thru]^-1
    # As [Thru] = [Half_thru]*[Half_thru] -> [Half_thru] = sqrt([Thru])
    # Therefore, [DUT] = sqrt([Thru])^-1*[Measured_data]*sqrt([Thru])^-1
    
    # Inputs all of the on wafer Thru devices
    # Outputs the best params to use for de-embedding - return = sqrt(ABCD)^-1 - such that you can later apply return*data*return to de-embed the data
    # Prints the error and its integral for each de-embedding protocol to help select the best de-embedding data
  
    dm = [] #initialize an empty list to store the de-embedded data
    total_error = [0.0] * len(thru) #initialize an empty list of floats to store the total error for each de-embedding protocol
    num_colors = len(thru)*(len(thru)-1) #as each thru cal, len(thru), will be applied to all other thru measurements, (len(thru)-1)
    colors = plt.cm.jet(np.linspace(0,1,num_colors))
    color_count = 0
    
    # If there is only one thru device then return the inverse square root of the ABCD matrix and skip the error calculations
    if len(thru) == 1:
        t = thru[0]
        cal = t.network.a
        cal_transformed = np.empty_like(cal)
        for i in range(cal.shape[0]):
            cal_transformed[i,:,:] = np.linalg.inv(sqrtm(cal[i,:,:]))
        return cal_transformed
    
    # Generate calibration ABCD matrices for each thru device
    for t in thru:
        cal = t.network.a
        # Initialize an empty array with the same shape as cal
        cal_transformed = np.empty_like(cal)
        # Iterate over the first dimension of cal to find the inverse square root of each 2x2 ABCD matrix and return it to cal_transformed
        for i in range(cal.shape[0]):
            # Compute the inverse square root of the 2x2 matrix at each f point
            cal_transformed[i,:,:] = np.linalg.inv(sqrtm(cal[i,:,:]))
        dm.append(cal_transformed)


    
    for count_t, th  in enumerate(thru, start=0):
        for i in range(len(dm)):
            # apply each deembedding to the thru device except for the one it was generated from
            if i == count_t:
                continue # skip this iteration of the for loop if the de-embedding protocol is being applied to the thru device it was generated from
            
            # pre and post multiply by the inverse square root of the ABCD matrix to de-embed the data
            thru_ABCD_mat = np.matmul(dm[i],np.matmul(th.network.a,dm[i]))
            # Generate a network object from the de-embedded data
            thru_ABCD = rf.Network(frequency = th.network.f, a = thru_ABCD_mat, z0 = th.network.z0)
            #store the error for each de-embedding protocol in the respective array index
            error_value = np.abs(np.sum(thru_ABCD.s[:,1,0]))+np.abs(np.sum(thru_ABCD.s[:,0,1]))
            total_error[i] += error_value #add the error for each de-embedding protocol applied to every thru to the total error for that protocol 
                    
            if plot_cal:
                plt.figure(f'ABCD De-embedding')
                th.network.plot_s_db(m=1, n=0, color=colors[color_count], linestyle='dashed',label = f'Raw: {th.label}')  # Plot only s21 with colorblind colormap
                thru_ABCD.plot_s_db(m=1, n=0, color=colors[color_count], label = f'ABCD_deembeding: {th.label};')  # Plot only s21 with colorblind colormap
                plt.ylim(-1,0.4)
                color_count += 1
            
            
    print(total_error)
    
    #Find the best de-embedding protocol
    min_index = total_error.index(min(total_error))
    max_index = total_error.index(max(total_error))

    print(f"Best ABCD de-embedding protocol: dm[{min_index}] = {min(total_error)}, worst:dm[{max_index}] = {max(total_error)}")
    

    #only return the best de-embedding protocol
    return dm[min_index]


def deembed_ABCD(s2p_files, ABCD):
# applies the selected ABCD de-embedding protocol to the data and re-writes the filename to have the de-embedded tag
    for f in s2p_files:
        f.network = rf.Network(frequency = f.network.f, a = np.matmul(ABCD,np.matmul(f.network.a,ABCD)), z0 = f.network.z0)
        f.filename = f.filename + '_deembeded_ABCD'
    return s2p_files



def keyplot(dev, cal_in = [], dev_selection = None, sub_set = None, y_range = None,
            x_range = slice(0,-1), log_x = False, plot_type = ['S_db'],m_port=[2], n_port=[1], deembed_data = True):
    # Function to plot the data for the selected devices and states
    # A number of inputs are given default values so they can be omitted from the function input if not required as they are quite standard
    # The default values also means that you can call them by name and not require the perfect ordring of the inputs
    # Plot type can be S, Z, Y, T, ABCD, or Smith from defauls - I have added linez, inputz, and power to calculate the input impedance,line impedance, and power
    #x_range: slice the data to remove the low frequency noise - default = no slicing - input form should be either '10-20ghz' or slice(1:10) for the first 10 points
    #         slightly different to y range as we are slicing the data object instead of changing the range of the plot. 
    #         necessary to generalise for the smith chart where you can't change axis limits to do this

    # y_range = [0, 1] : 2 item list giving limits for the y-axis of the plot
    # dev_selection = ['r2c1'] - specific devices to plot default is all devices
    # subset = ['formed', 'pristine'] - Select only runs that contain any of the subset strings in their (all/empty plots all states) can make this a list of strings to select multiple subsets
    # m_port/n_port: S-parameter to plot - m/n = 2/1 -> S21; can pass multiple into list so m_port=[1,2] n_port=[1] will plot S11, S21
    # plot_type: Follows rf types so can be S, Z, Y, T, ABCD, or Smith with _db/_mag/_re/_im options for all except smith (e.g. S_db, Z_re, Smith)

    figs_axes = [] #initialize an empty list to store the figure and axis objects
    plot_type = plot_type.lower() # removes case sensitivity for the plot type input (Smith/smith/SMITH... all work)
   
    # If no devices are selected then plot all devices stored in dev
    if dev_selection is None:
        dev_selection = dev.keys()

    # If subset is empty or 'all' then set the subset_check to always return True 
    if sub_set is None:
        subset_check = lambda x: True
    # Else check if the filename contains any of the sub_set strings returning true/false to plot/not plot
    else:
        subset_check = lambda x: any(m in x.filename.lower() for m in sub_set)
    
    # Loop entire plotting function over the selected plot types
    for p_type in plot_type:  
          
        # Loop over the selected devices    
        for key in dev_selection:
            if key in dev: #extract the s2p files for the selected devices
                value = dev[key] 
                fig, ax = plt.subplots()
                if deembed_data == True:
                    ax.set_title(f'{p_type}: Device_{key} - deembedding: {cal_in.name}')
                    
                else:
                    ax.set_title(f'{p_type}: Device_{key}')
                    
                num_colors = sum(1 for r in value if subset_check(r)) #count the number of files that match the subset criteria to set the number of colors            
                #print(num_colors)
                colors = plt.cm.jet(np.linspace(0,1,num_colors))
                color_count = 0
                line_styles = ['-', '--', '-.', ':'] #list of line styles to cycle through for each plot
                line_style_iterator = itertools.cycle(line_styles) #makes an iterator object that can be cyled through with next() to get the next line style

                # Loop over the s2p files for each select device plotting all that match the subset criteria
                for r in value:
                    if subset_check(r):
                        
                        if deembed_data == True:
                            data = cal_in.deembed(r.network)
                        else:
                            data = r.network
                            
                        #slice the data to plot selected frequency range (necessary for the smith chart where you can't change axis limits to do this)
                        data_sliced = data[x_range]

                        # Loop over the selected S-parameters to plot
                        for mm in m_port:
                            for nn in n_port:
                                
                                if p_type == 'smith':
                                    data_sliced.plot_s_smith(m=mm,n=nn,draw_labels=True, color=colors[color_count],
                                                                        linestyle = next(line_style_iterator), label = f'S_{mm}{nn} {r.filename}')   
                                    
                                elif p_type == 'inputz' or p_type == 'linez':
                                    med_kernel = 23
                                    Z11 = medfilt(data_sliced.z_re[:, 0, 0], kernel_size=med_kernel) + 1j *medfilt(data_sliced.z_im[:, 0, 0], kernel_size=med_kernel)
                                    Z12 = medfilt(data_sliced.z_re[:, 0, 1], kernel_size=med_kernel) + 1j *medfilt(data_sliced.z_im[:, 0, 1], kernel_size=med_kernel)
                                    Z21 = medfilt(data_sliced.z_re[:, 1, 0], kernel_size=med_kernel) + 1j *medfilt(data_sliced.z_im[:, 1, 0], kernel_size=med_kernel)
                                    Z22 = medfilt(data_sliced.z_re[:, 1, 1], kernel_size=med_kernel) + 1j *medfilt(data_sliced.z_im[:, 1, 1], kernel_size=med_kernel)
                                    Z_load = 50

                                    if p_type == 'inputz':
                                        # Calculate the input impedance
                                        z_in = Z11 - np.multiply(Z12, Z21) / (Z22 + Z_load)
                                        z_out = Z22 - np.multiply(Z12, Z21) / (Z11 + Z_load)
                                        ax.plot(data_sliced.f, abs(z_in), color=colors[color_count],
                                                linestyle = '-', label = f'Z_in_mag{p_type}_{mm}{nn}: {dev.filename}') 
                                        ax.plot(data_sliced.f, abs(z_out), color=colors[color_count],
                                                linestyle = ':', label = f'Z_out_mag{p_type}_{mm}{nn}: {dev.filename}')
                                    elif p_type == 'linez':
                                        z_line = abs(Z11-Z12) + abs(Z22-Z12)
                                        ax.plot(data_sliced.f, z_line, color=colors[color_count],
                                            linestyle = '-', label = f'Z_line_mag{p_type}_{mm}{nn}: {dev.filename}') 
                                
                                elif p_type == 'oneportz':
                                    z = 50*(1+data_sliced.s[:,1,1])/(1-data_sliced.s[:,1,1])    
                                    ax.plot(data_sliced.f, abs(z), color=colors[color_count],label = f'Z_one_port{p_type}_{2}{2}: {r.filename}')
                                    
                                elif p_type == 'power':
                                    forward_power = np.square(np.abs(data_sliced.s[:,0,0])) + np.square(np.abs(data_sliced.s[:,0,1]))
                                    reverse_power = np.square(np.abs(data_sliced.s[:,1,0])) + np.square(np.abs(data_sliced.s[:,1,1]))
                                    
                                    ax.plot(data_sliced.f, forward_power, color=colors[color_count],
                                            linestyle = '-', label = f'Forward_power_mag{p_type}_{mm}{nn}: {r.filename}') 
                                    ax.plot(data_sliced.f, reverse_power, color=colors[color_count],
                                            linestyle = ':', label = f'Reverse_power_mag{p_type}_{mm}{nn}: {r.filename}')
                                        
                                else:
                                    p_data = getattr(data_sliced, p_type)[:, mm-1, nn-1]
                                    #apply median filter to the data to smooth it
                                    p_data_smoothed = p_data#medfilt(p_data, kernel_size=23)
                                    
                                    ax.plot(data_sliced.f, p_data_smoothed, color=colors[color_count],
                                            linestyle = next(line_style_iterator), label = f'{p_type}_{mm}{nn}: {r.filename}')   
                        color_count += 1 # change color for each file (needs to be inside the subset check loop so it only changes for the files that are plotted)
    
                if log_x:
                            ax.set_xscale('log')
                if y_range is not None:
                    ax.set_ylim(y_range)          
                ax.set_xlabel('Frequency (Hz)')
                if p_type == 's_db':
                    ax.set_ylabel('Magnitude (dB)')
                else:
                    ax.set_ylabel('Magnitude')   
                ax.legend(loc='upper right',fontsize='xx-small')
        figs_axes.append((fig, ax))
    return figs_axes




def sub_plot(ax, dev_subset = [], cal_in = [], y_range = None,
            x_range = slice(0,-1), log_x = False, log_y = False, plot_type = ['S_db'],m_port=[2], n_port=[1], deembed_data = True, iterate_lines = False,
            p_legend = True, window_size = 0,R_in = [30e3],dot_line = False):
    # Plotting function that takes an input of a list of lists
    # The function then plots all the devices in each subset on the same graph giving different color maps to each subset
    # and different colors within each subset for each device
    # Plot type can be S, Z, Y, T, ABCD, or Smith from defauls - I have added linez, inputz, and power to calculate the input impedance,line impedance, and power
    # if plot_type is snorm or z norm then it will normalise the input impedance or s parameters to the first item in the dataset
    
    figs_axes = [] #initialize an empty list to store the figure and axis objects
    line_styles = ['-', '--', '-.', ':'] #list of line styles to cycle through for each plot
    line_style_iterator = itertools.cycle(line_styles) #makes an iterator object that can be cyled through with next() to get the next line style
    color_maps = ['binary','winter','autumn','Greens', 'Purples','Blues', 'Oranges',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn','Reds'] #list of sequential color maps to cycle through for each subset
    
    
    # Loop entire plotting function over the selected plot types
    for p_type in plot_type: 
        p_type = p_type.lower() # removes case sensitivity for the plot type input (Smith/smith/SMITH... all work)
        #fig, ax = plt.subplots()   
        color_map_iterator = itertools.cycle(color_maps) #makes an iterator object that can be cyled through with next() to get the next color map
        if deembed_data == True:
            ax.set_title(f'{p_type} - deembedding: {cal_in.name}')            
        else:
            ax.set_title(f'{p_type}')     
              
    # Loop over the selected devices 
        for  sub_count, subset in enumerate(dev_subset, start=0):
            if iterate_lines == True:
                line_obj = next(line_style_iterator) #get the first line style to pass as a plotting argument
            else:   
                line_obj = '-'  # Always return the first object in the line_style_iterator
                
            color_obj = next(color_map_iterator) #get the first color map to pass as a plotting argument
            
            num_colors = len(subset) # set number of colors to number of devices within the subset
            colors = plt.get_cmap(color_obj)(np.linspace(0.2,1,num_colors))
            color_count = 0 #initiate color count to cycle through the colors for each device in the subset
                
            for count_d, dev in enumerate(subset):      

                if deembed_data == True:
                    data = cal_in.deembed(dev.network)
                else:
                    data = dev.network
                    
                #slice the data to plot selected frequency range (necessary for the smith chart where you can't change axis limits to do this)
                data_sliced = data[x_range]
                freq_plot = data_sliced.f/1e9 #convert frequency to GHz for plotting

                # Loop over the selected S-parameters to plot
                for mm in m_port:
                    for nn in n_port:
                        
                        if p_type == 'smith':
                            data_sliced.plot_s_smith(m=mm,n=nn,draw_labels=True, color=colors[color_count],
                                                                linestyle = line_obj, label = f'S_{mm}{nn} {dev.filename}')   
                            
                        elif p_type == 'inputz' or p_type == 'linez' or p_type == 'znorm' or p_type == 'snorm':
                            med_kernel = 23
                            Z11 = medfilt(data_sliced.z_re[:, 0, 0], kernel_size=med_kernel) + 1j *medfilt(data_sliced.z_im[:, 0, 0], kernel_size=med_kernel)
                            Z12 = medfilt(data_sliced.z_re[:, 0, 1], kernel_size=med_kernel) + 1j *medfilt(data_sliced.z_im[:, 0, 1], kernel_size=med_kernel)
                            Z21 = medfilt(data_sliced.z_re[:, 1, 0], kernel_size=med_kernel) + 1j *medfilt(data_sliced.z_im[:, 1, 0], kernel_size=med_kernel)
                            Z22 = medfilt(data_sliced.z_re[:, 1, 1], kernel_size=med_kernel) + 1j *medfilt(data_sliced.z_im[:, 1, 1], kernel_size=med_kernel)
                            Z_load = 50

                            if p_type == 'inputz':
                                # Calculate the input impedance
                                z_in = Z11 - np.multiply(Z12, Z21) / (Z22 + Z_load)
                                z_out = Z22 - np.multiply(Z12, Z21) / (Z11 + Z_load)
                                ax.plot(freq_plot, np.real(z_in), color=colors[color_count],
                                        linestyle = '-', label = f'Real(Z_in)_mag{p_type}_{mm}{nn}: {dev.filename}') 
                                ax.plot(freq_plot, np.imag(z_in), color=colors[color_count],
                                        linestyle = ':', label = f'Imag(Z_in)_mag{p_type}_{mm}{nn}: {dev.filename}')
                            elif p_type == 'linez':
                                z_line = abs(Z11-Z12) + abs(Z22-Z12)
                                ax.plot(freq_plot, z_line, color=colors[color_count],
                                    linestyle = '-', label = f'Z_line_mag{p_type}_{mm}{nn}: {dev.filename}') 
                            elif p_type == 'znorm':
                                z_in = Z11 - np.multiply(Z12, Z21) / (Z22 + Z_load)
                                # select firs device in the subset to normalise the input impedance to
                                if count_d == 0:
                                    z_ref = np.copy(z_in)
                               
                                #normalise the input impedance to the first device in the subset                               
                                z_norm = abs(z_in)/abs(z_ref)
                                     
                                ax.plot(freq_plot, abs(z_norm), color=colors[color_count],
                                    linestyle = '-', label = f'Z_norm_mag{p_type}_{mm}{nn}: {dev.filename}')
                                
                            elif p_type == 'snorm':
                                if count_d == 0:
                                    s_ref = data_sliced.s[:, mm-1, nn-1]

                                s_norm = abs(data_sliced.s[:, mm-1, nn-1])/abs(s_ref)
                                ax.plot(freq_plot, abs(s_norm), color=colors[color_count],
                                    linestyle = '-', label = f'S_norm_mag{p_type}_{mm}{nn}: {dev.filename}')
                        
                        elif p_type == 'oneportz':
                            z = 50*(1+data_sliced.s[:,1,1])/(1-data_sliced.s[:,1,1])    
                            ax.plot(freq_plot, abs(z), color=colors[color_count],label = f'Z_one_port{p_type}_{2}{2}: {dev.filename}')
                            #ax.set_yscale('log')
                                        
                        elif p_type == 'power':
                            power_in = -12 #input power in dBm
                            power_in_mw = 10**(power_in/10) #convert input power to mW
                            if count_d == 0:
                                print('Power in mW:', power_in_mw)
                            forward_power = power_in_mw*(1 - np.square(np.abs(data_sliced.s[:,0,0])) - np.square(np.abs(data_sliced.s[:,1,0])))
                            reverse_power = power_in_mw*(1 - np.square(np.abs(data_sliced.s[:,1,1])) - np.square(np.abs(data_sliced.s[:,0,1])))
                            
                            ax.plot(freq_plot, forward_power, color=colors[color_count],
                                    linestyle = '-', label = f'Forward_power_mag{p_type}: {dev.filename}') 
                            ax.plot(freq_plot, reverse_power, color=colors[color_count],
                                    linestyle = ':', label = f'Reverse_power_mag{p_type}: {dev.filename}')
                            ax.set_ylabel('Power dissipated (mW)')
                            
                        elif p_type == 'cap':
                            z_dut = getattr(data_sliced, 'a')[:,0,1]
                            f_app = getattr(data_sliced, 'f')
                            R_p = R_in[sub_count][count_d]
                            c_f = ((1)/(1j*2*np.pi)) * ( (R_p-(z_dut-30))/((z_dut-30)*R_p) )
                            ax.plot(freq_plot, np.divide(abs(c_f),f_app), color=colors[color_count],
                                    linestyle = '-', label = f'Cap{p_type}_{mm}{nn}:res{R_p} {dev.filename}')          
                        
                        else:
                            p_data = getattr(data_sliced, p_type)[:, mm-1, nn-1]
                            # apply moving average filter to the data to smooth it
                            if window_size != 0:
                                
                                window_s = window_size
                                p_data_smoothed = np.convolve(p_data, np.ones(window_s)/window_s, mode='same')
                            else:
                                p_data_smoothed = p_data
                            # plot the data
                            if dot_line:
                                linestyle = ':'
                            else:
                                linestyle = line_obj
                            
                            ax.plot(freq_plot, p_data_smoothed, color=colors[color_count],
                                    linestyle=linestyle, label=f'{p_type}_{mm}{nn}: {dev.filename}')
                color_count += 1 # change color for each file (needs to be inside the subset check loop so it only changes for the files that are plotted)

                if log_x:
                            ax.set_xscale('log')
                if log_y:
                            ax.set_yscale('log')
                if y_range is not None:
                    ax.set_ylim(y_range)          
                ax.set_xlabel('Frequency (GHz)')
                if p_type == 's_db':
                    ax.set_ylabel('Magnitude (dB)')
                elif p_type == 's_deg' or p_type == 'z_deg' or p_type == 'y_deg' or p_type == 't_deg' or p_type == 'a_deg':
                    ax.set_ylabel('Phase (degrees)')
                elif p_type != 'power':
                    ax.set_ylabel('Magnitude')   
                if p_legend == True:
                    ax.legend(loc='lower right',fontsize='xx-small')
                    
        #figs_axes.append((fig, ax))
    #return figs_axes

def subgen(s2p_files, run_nums = [[],[],[]]):
    # Function taking all the s2p files and grouping them into a list of subsets, where each subset is itself a list of the s2p files
    # The subsets can then be plotted in different colors/linetypes on the same graph
    # Input option 1: run_nums is a list of lists, each list contains the run numbers for the subset of devices
    # Input option 2: 
    dev_subs = {}
    dev_subs = []
    for l in run_nums:
        group = []
        for n in l:
            group.append(s2p_files[n-1]) #subtract 1 from the run number due to zero indexing in python
        dev_subs.append(group)
    return dev_subs

def fourier_filter(s2p_files_copy, threshold = [1.8e-8,2.2e-8], t_window = False):
    # Function to apply a fourier filter to the data to remove noise - is applied to the list of all the files so it can be the first step in the analysis
    # The threshold is the frequency above which the noise is removed
    # The function then returns the filtered data back into the list of s2p files
    
    sum_freqs = np.zeros(len(s2p_files_copy[0].network.f)) # Initiating an empty numpy array same size as the data to store the sum of all the FFT's to find common peaks that could be systematic
    sum_freqs_filtered = np.zeros(len(s2p_files_copy[0].network.f)) # Initiating an empty numpy array same size as the data to store the filtered FFt's to compare
    for s in s2p_files_copy:
        for n in range(1,3):
            for m in range(1,3):
                
                # Extract the s-parameters
                s_params = s.network.s[:,m-1,n-1]
                
                # Interpolate the data onto a uniform grid
                f_uniform = np.linspace(s.network.f.min(), s.network.f.max(), len(s.network.f))
                interp_func = interp.interp1d(s.network.f, s_params)
                s_params_uniform = interp_func(f_uniform)
                
                # Apply Tuke window if window is True
                if t_window != False:
                    s_params_uniform *= scipy.signal.windows.tukey(len(s_params_uniform), alpha=t_window, sym=True)
                
                # Perform the inverse Fourier transform to go to time domain
                time_data = fft.ifft(s_params_uniform)
                
                # Generate the times for the inverse Fourier transform
                times = fft.fftfreq(len(s_params_uniform), d=f_uniform[1] - f_uniform[0])

                # Compute the amplitudes
                amplitudes = np.abs(time_data)
                sum_freqs += amplitudes

                # Loop over the list of thresholds applying one or multiple bandstop filters to the time data
                filtered_time_data = np.copy(time_data)
                for thr in threshold:
                    filtered_time_data[(np.abs(times) > thr[0]) & (np.abs(times) < thr[1])] = 0
                sum_freqs_filtered += np.abs(filtered_time_data)
                
                # Perform the Fourier transform to go back to frequency domain
                filtered_s_params = fft.fft(filtered_time_data)

                # Interpolate the filtered data back onto the original frequency grid
                interp_func = interp.interp1d(f_uniform, filtered_s_params)
                filtered_s_params_original = interp_func(s.network.f)

                # Write the filtered s-parameters back into the original data
                s.network.s[:,m-1,n-1] = filtered_s_params_original

        # Change the label to indicate the data has been filtered
        s.filename = s.filename[0:-1] + '_FFT_filtered'

    plt.figure(figsize=(5, 2))
    plt.plot(times,sum_freqs, color='blue',linestyle =':')
    plt.plot(times,sum_freqs_filtered, color='green')
    for thr in threshold:
        for t in thr:
            plt.axvline(x=t, color='red')
            plt.axvline(x=-t, color='red')
    plt.yscale('log')
    return s2p_files_copy




def fourier_convolve(s2p_files_copy, thru):
    # Function to apply a fourier filter to the data to remove noise - is applied to the list of all the files so it can be the first step in the analysis
    # The threshold is the frequency above which the noise is removed
    # The function then returns the filtered data back into the list of s2p files
    
    thru_average = np.zeros((len(s2p_files_copy[0].network.f),2,2), dtype=complex) #initiate object to store empty FFT average of all the thru data for each s param hence the 2,2
    for count_t, t in enumerate(thru,start=0):
        for n in range(1,3):
            for m in range(1,3):
                # Extract the s-parameters
                s_params = t.network.s[:,m-1,n-1]
                
                # Interpolate the data onto a uniform grid
                f_uniform = np.linspace(t.network.f.min(), t.network.f.max(), len(t.network.f))
                interp_func = interp.interp1d(t.network.f, s_params)
                s_params_uniform = interp_func(f_uniform)
                
                # Apply Hanning window function
                #window = np.hanning(len(s_params_uniform))
                #s_params_uniform = s_params_uniform * window
                
                ## Perform the Fourier transform
                fourier = fft.fft(s_params_uniform)
                # Generate the frequencies for the Fourier transform
                freqs = fft.fftfreq(len(s_params_uniform), d=f_uniform[1] - f_uniform[0])

                # Generate average FFT amplitude for all inputted through devices to then multiply with the device data to 'convolve' in the FD
                fourier_copy = np.copy(fourier)
                if count_t == 0:
                    thru_average[:,m-1,n-1] = fourier_copy
                else:
                    thru_average[:,m-1,n-1] = np.sum([fourier_copy, thru_average[:,m-1,n-1]], axis=0) / 2
                
    sum_freqs = np.zeros(len(s2p_files_copy[0].network.f)) # Initiating an empty numpy array same size as the data to store the sum of all the FFT's to find common peaks that could be systematic
    sum_freqs_filtered = np.zeros(len(s2p_files_copy[0].network.f)) # Initiating an empty numpy array same size as the data to store the filtered FFt's to compare
    for s in s2p_files_copy:
        for n in range(1,3):
            for m in range(1,3):
                
                # Extract the s-parameters
                s_params = s.network.s[:,m-1,n-1]
                
                # Interpolate the data onto a uniform grid
                f_uniform = np.linspace(s.network.f.min(), s.network.f.max(), len(s.network.f))
                interp_func = interp.interp1d(s.network.f, s_params)
                s_params_uniform = interp_func(f_uniform)
                
                # Apply Hanning window function
                #window = np.hanning(len(s_params_uniform))
                #s_params_uniform = s_params_uniform * window
                
                # Perform the Fourier transform
                fourier = fft.fft(s_params_uniform)
                
                # Generate the frequencies for the Fourier transform
                freqs = fft.fftfreq(len(s_params_uniform), d=f_uniform[1] - f_uniform[0])
                sum_freqs += np.abs(fourier)

                # Divide by the average of the through data to 'deconvolve' the data
                filtered_fourier = np.subtract(np.copy(fourier),thru_average[:,m-1,n-1])
                
                sum_freqs_filtered += np.abs(filtered_fourier)
                
                # plt.figure(figsize=(10, 5))
                # plt.title(f'{s.filename} - {m}{n}')
                # plt.plot(freqs,np.real(fourier), color='blue',linestyle ='-')
                # plt.plot(freqs,np.imag(fourier), color='blue',linestyle =':')
                
                # plt.plot(freqs,np.real(filtered_fourier), color='green',linestyle ='-')
                # plt.plot(freqs,np.imag(filtered_fourier), color='green',linestyle =':')
                
                # plt.plot(freqs,np.real(thru_average[:,m-1,n-1]), color = 'purple',linestyle ='-')
                # plt.plot(freqs,np.imag(thru_average[:,m-1,n-1]), color = 'purple',linestyle =':')
                #plt.yscale('log')
                
                # Perform the inverse Fourier transform
                filtered_s_params = fft.ifft(filtered_fourier)

                # Interpolate the filtered data back onto the original frequency grid
                interp_func = interp.interp1d(f_uniform, filtered_s_params)
                filtered_s_params_original = interp_func(s.network.f)

                # Write the filtered s-parameters back into the original data
                s.network.s[:,m-1,n-1] = filtered_s_params_original

        # Change the label to indicate the data has been filtered
        s.filename = s.filename[0:-4] + '_FFT_filtered'

    plt.figure(figsize=(10, 5))
    plt.plot(freqs,sum_freqs, color='blue',linestyle =':')
    plt.plot(freqs,sum_freqs_filtered, color='green')
    plt.plot(freqs,np.abs(thru_average[:,2-1,1-1]), color = 'purple')
    plt.yscale('log')
    return s2p_files_copy

def fourier_inverse(s2p_files_copy2, thru):
    # Function that carries out an inverse fourier transform on the through data to convert it to the time domain
    # Then attempts to do the same for the rest of the data and to do matrix operations between them to deconvolve the thru from the rest of the s-parameters
    # The function then returns the filtered data back into the list of s2p files

    thru_average = np.zeros((2*len(s2p_files_copy2[0].network.f),2,2), dtype=complex) #initiate object to store empty FFT average of all the thru data for each s param hence the 2,2
    for count_t, t in enumerate(thru,start=0):
        for n in range(1,3):
            for m in range(1,3):
                # Extract the s-parameters
                s_params = t.network.s[:,m-1,n-1]
                
                # Interpolate the data onto a uniform grid
                f_uniform = np.linspace(t.network.f.min(), t.network.f.max(), len(t.network.f))
                interp_func = interp.interp1d(t.network.f, s_params)
                s_params_uniform = interp_func(f_uniform)
                
                # Apply Hanning window function
                window = np.hanning(len(s_params_uniform))
                s_params_uniform = s_params_uniform * window
                
                # Create a Hermitian symmetric version of the s-parameters
                s_params_hermitian = np.concatenate((s_params_uniform, np.conj(s_params_uniform[::-1])))
                
                # Perform the inverse Fourier transform
                time_domain = fft.ifft(s_params_hermitian)
                
                # Generate the frequencies for the inverse Fourier transform
                freqs = fft.fftfreq(len(s_params_uniform), d=f_uniform[1] - f_uniform[0])

                # Generate average FFT amplitude for all inputted through devices to then multiply with the device data to 'convolve' in the FD
                time_domain_copy = np.copy(time_domain)
                if count_t == 0:
                    thru_average[:,m-1,n-1] = time_domain_copy
                else:
                    thru_average[:,m-1,n-1] = np.sum([time_domain_copy, thru_average[:,m-1,n-1]], axis=0) / 2
                    
    sum_times = np.zeros(2*len(s2p_files_copy2[0].network.f)) # Initiating an empty numpy array same size as the data to store the sum of all the time domain data to find common peaks that could be systematic
    sum_times_filtered = np.zeros(2*len(s2p_files_copy2[0].network.f)) # Initiating an empty numpy array same size as the data to store the filtered time domain data to compare
    time_domain = np.zeros((2*len(s2p_files_copy2[0].network.f),2,2), dtype=complex) #initiate object to store empty time domain data for each s param hence the 2,2
    for s in s2p_files_copy2:
        for n in range(1,3):
            for m in range(1,3):
                
                # Extract the s-parameters
                s_params = s.network.s[:,m-1,n-1]
                
                # Interpolate the data onto a uniform grid
                f_uniform = np.linspace(s.network.f.min(), s.network.f.max(), len(s.network.f))
                interp_func = interp.interp1d(s.network.f, s_params)
                s_params_uniform = interp_func(f_uniform)
                s_length = len(s_params_uniform)
                
                # Apply Hanning window function
                window = np.hanning(len(s_params_uniform))
                s_params_uniform = s_params_uniform * window
                
                # Create a Hermitian symmetric version of the s-parameters
                s_params_hermitian = np.concatenate((s_params_uniform, np.conj(s_params_uniform[::-1])))
                
                # Perform the inverse Fourier transform
                time_domain[:,m-1,n-1] = fft.ifft(s_params_hermitian)
                
        #sum_times += np.abs(time_domain)

        # data process
        #######################**************************************
        
        thru_average_sqrt = np.zeros((2*len(s2p_files_copy2[0].network.f),2,2), dtype=complex)
        thru_average_sqrt_inv = np.zeros((2*len(s2p_files_copy2[0].network.f),2,2), dtype=complex)
        time_domain_filtered = np.zeros((2*len(s2p_files_copy2[0].network.f),2,2), dtype=complex)
        
        # Compute the square root of the inverse
        for i in range(time_domain.shape[0]):
            # Compute the square root of the S-parameter matrix
            thru_average_sqrt[i,:,:] = sqrtm(time_domain[i,:,:])
            # Compute the inverse of the square root of the S-parameter matrix
            thru_average_sqrt_inv[i,:,:] = np.linalg.inv(thru_average_sqrt[i,:,:])

            # Pre and post multiply time_domain by the square root of the inverse
            time_domain_filtered[i,:,:] = np.matmul((thru_average_sqrt_inv[i,:,:]), np.matmul(time_domain[i,:,:], thru_average_sqrt_inv[i,:,:]))
        
        
        for n in range(1,3):
            for m in range(1,3):        
                # Perform the Fourier transform to convert back to s-parameters
                s_params_filtered = fft.fft(time_domain_filtered[:,m-1,n-1])
                
                # Only keep the first half of the s-parameters (discard the symmetric part)
                s_params_filtered = s_params_filtered[:s_length]
                
                # Interpolate the filtered data back onto the original frequency grid
                interp_func = interp.interp1d(f_uniform, s_params_filtered)
                filtered_s_params_original = interp_func(s.network.f)

                # Write the filtered s-parameters back into the original data
                s.network.s[:,m-1,n-1] = filtered_s_params_original

  

        # Change the label to indicate the data has been filtered
        s.filename = s.filename[0:-4] + '_FFT_filtered'


    # plt.figure(figsize=(10, 5))
    # plt.plot(sample_times, sum_times, color='blue', linestyle=':')
    # plt.plot(sample_times, sum_times_filtered, color='green')
    # plt.yscale('log')

    return s2p_files_copy2








