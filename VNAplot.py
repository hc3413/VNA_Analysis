#source VNAenv/bin/activate (launching and exiting the virtual environment containing the required modules, stored in the working directory for VNA_Analysis)
#VNAenv/bin/python your_script.py - for running a script in the virtual environment
#source deactivate

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
    files = [f for f in os.listdir(data_path) if f.endswith('.S2P')]
    
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
                    
    # Sort the list chronologically based on the datetime
    file_dates.sort(key=itemgetter(1))
    
    #for f, date in file_dates:
        #print(f, date)
        
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
        keywords = ['thrunotaper', 'opennotaper','shortnotaper', 'opensignotaper', 'openverynarrow','opennarrow', 'smallform', 'fullform', 'opensig', 'thruISS','pristine', 'formed', 'thru', 'open', 'short','set','reset']
        state = next((x for x in keywords if x in f.lower()), None) #returns the first keyword found in the state value, stops as soon as the first keyword is found
        # Extract the row, colum and wafer numbers from the filename (e.g. wafer 1 r1_c11) and store into position variable
        wafer_number = re.findall(r'Wafer(\d)', f, re.IGNORECASE)
        r_number = re.findall(r'_r(\d{1,2})_', f, re.IGNORECASE)
        c_number = re.findall(r'_c(\d{1,2})_', f, re.IGNORECASE)
        #store the network and its associated metadata in the S2PFile object grouped toegher in the s2p_files list
        if not wafer_number or not r_number or not c_number:
            s2p_files.append(S2PFile(network, f, int(0), None, None, state, run_count)) #ISS is classes as wafer zero
        else:
            s2p_files.append(S2PFile(network, f, int(wafer_number[0]), int(r_number[0]), int(c_number[0]), state.lower(),run_count))
        run_count += 1
    return s2p_files


def calibration_OS(open, short, thru, plot_cal = False):
    # Function to generate an OpenShort De-embedding calibration and apply it to the on-wafer thru measurements
    # Inputs multiple open and short measurements and compares them to see which give the most effective de-embedding
    # Outputs a list of de-embedding data and plots of the thru measurements before and after de-embedding for each protocol
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

    print(f"Best de-embedding protocol: dm[{min_index}] = {min(error)}, worst:dm[{max_index}] = {max(error)}")
    
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
    
    # Inputs multiple open and short measurements and compares them to see which give the most effective de-embedding
    # Outputs a list of de-embedding data and plots of the thru measurements before and after de-embedding for each protocol
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
    dm_list = range(len(dm))
    
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
                th.network.plot_s_mag(m=1, n=0, color=colors[color_count], linestyle='dashed',label = f'Raw: {th.label}; {sh.label}; O{count_o}S{count_sh}T{count_th}')  # Plot only s21 with colorblind colormap
                thru_TX.plot_s_mag(m=1, n=0, color=colors[color_count], label = f'OS: {th.label}; {sh.label}; O{count_o}S{count_sh}T{count_th}')  # Plot only s21 with colorblind colormap
                color_count += 1
            
            
    print(total_error)
    
    #Find the best de-embedding protocol
    min_index = total_error.index(min(total_error))
    max_index = total_error.index(max(total_error))

    print(f"Best de-embedding protocol: dm[{min_index}] = {min(total_error)}, worst:dm[{max_index}] = {max(total_error)}")
    
    #Plot the best de-embedding protocol
    if plot_cal:
        plt.figure('Best De-embedding Protocol')
        thru[0].network.plot_s_mag(m=1, n=0, color='red', linestyle='dashed',label = f'Raw')  # Plot only s21 with colorblind colormap
        dm[min_index].deembed(thru[0].network).plot_s_mag(m=1, n=0, color='green', label = f'Best OS:dm[{min_index}]')  # Plot only s21 with colorblind colormap
        dm[max_index].deembed(thru[0].network).plot_s_mag(m=1, n=0, color='blue', label = f'Worst OS:dm[{max_index}]')  # Plot only s21 with colorblind colormap
   
    #only return the best de-embedding protocol
    return dm[min_index]






def keyplot(dev, cal_in = [], dev_selection = None, sub_set = None, y_range = None,
            x_range = slice(0,-1), log_x = False, plot_type = ['S_db'],m_port=[2], n_port=[1], deembed_data = True):
    # Function to plot the data for the selected devices and states
    # A number of inputs are given default values so they can be omitted from the function input if not required as they are quite standard
    # The default values also means that you can call them by name and not require the perfect ordring of the inputs
    # Plot type can be S, Z, Y, T, ABCD, or Smith
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
                                    
                                elif p_type == 'inputz':
                                    med_kernel = 23
                                    Z11 = medfilt(data_sliced.z_re[:, 0, 0], kernel_size=med_kernel) + 1j *medfilt(data_sliced.z_im[:, 0, 0], kernel_size=med_kernel)
                                    Z12 = medfilt(data_sliced.z_re[:, 0, 1], kernel_size=med_kernel) + 1j *medfilt(data_sliced.z_im[:, 0, 1], kernel_size=med_kernel)
                                    Z21 = medfilt(data_sliced.z_re[:, 1, 0], kernel_size=med_kernel) + 1j *medfilt(data_sliced.z_im[:, 1, 0], kernel_size=med_kernel)
                                    Z22 = medfilt(data_sliced.z_re[:, 1, 1], kernel_size=med_kernel) + 1j *medfilt(data_sliced.z_im[:, 1, 1], kernel_size=med_kernel)
                                    Z_load = 50

                                    # Calculate the input impedance
                                    z_in = Z11 - np.multiply(Z12, Z21) / (Z22 + Z_load)
                                    z_out = Z22 - np.multiply(Z12, Z21) / (Z11 + Z_load)
                                    #z_in = abs(Z11-Z12)
                                    #z_in = medfilt(z_in, kernel_size=17)
                                    ax.plot(data_sliced.f, abs(z_in), color=colors[color_count],
                                            linestyle = '-', label = f'Z_in_mag{p_type}_{mm}{nn}: {r.filename}') 
                                    ax.plot(data_sliced.f, abs(z_out), color=colors[color_count],
                                            linestyle = ':', label = f'Z_out_mag{pp_type}_{mm}{nn}: {r.filename}')
                                    
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




def subplot(dev_subs = [], cal_in = [], y_range = None,
            x_range = slice(0,-1), log_x = False, plot_type = ['S_db'],m_port=[2], n_port=[1], deembed_data = True, iterate_lines = False):
    # Plotting function that takes an input of a dict of lists
    # The dict items are the subsets of devices, e.g. pristine, formed, etc
    # The lists are the devices that meet that criteria
    # The function then plots all the devices in each subset on the same graph giving different line types to each subset
    # and different colors within each subset for each device
    figs_axes = [] #initialize an empty list to store the figure and axis objects
    line_styles = ['-', '--', '-.', ':'] #list of line styles to cycle through for each plot
    line_style_iterator = itertools.cycle(line_styles) #makes an iterator object that can be cyled through with next() to get the next line style
    color_maps = ['binary','winter','autumn','Greens', 'Purples','Blues', 'Oranges',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn','Reds'] #list of sequential color maps to cycle through for each subset
    
    
    # Loop entire plotting function over the selected plot types
    for p_type in plot_type: 
        p_type = p_type.lower() # removes case sensitivity for the plot type input (Smith/smith/SMITH... all work)
        fig, ax = plt.subplots()   
        color_map_iterator = itertools.cycle(color_maps) #makes an iterator object that can be cyled through with next() to get the next color map
        if deembed_data == True:
            ax.set_title(f'{p_type} - deembedding: {cal_in.name}')            
        else:
            ax.set_title(f'{p_type}')     
              
    # Loop over the selected devices 
        for subset in dev_subs:
            if iterate_lines == True:
                line_obj = next(line_style_iterator) #get the first line style to pass as a plotting argument
            else:   
                line_obj = '-'  # Always return the first object in the line_style_iterator
                
            color_obj = next(color_map_iterator) #get the first color map to pass as a plotting argument
            
            num_colors = len(subset) # set number of colors to number of devices within the subset
            colors = plt.get_cmap(color_obj)(np.linspace(0.2,1,num_colors))
            color_count = 0 #initiate color count to cycle through the colors for each device in the subset
                
            for dev in subset:      

                if deembed_data == True:
                    data = cal_in.deembed(dev.network)
                else:
                    data = dev.network
                    
                #slice the data to plot selected frequency range (necessary for the smith chart where you can't change axis limits to do this)
                data_sliced = data[x_range]

                # Loop over the selected S-parameters to plot
                for mm in m_port:
                    for nn in n_port:
                        
                        if p_type == 'smith':
                            data_sliced.plot_s_smith(m=mm,n=nn,draw_labels=True, color=colors[color_count],
                                                                linestyle = line_obj, label = f'S_{mm}{nn} {dev.filename}')   
                            
                        elif p_type == 'inputz':
                            med_kernel = 23
                            Z11 = medfilt(data_sliced.z_re[:, 0, 0], kernel_size=med_kernel) + 1j *medfilt(data_sliced.z_im[:, 0, 0], kernel_size=med_kernel)
                            Z12 = medfilt(data_sliced.z_re[:, 0, 1], kernel_size=med_kernel) + 1j *medfilt(data_sliced.z_im[:, 0, 1], kernel_size=med_kernel)
                            Z21 = medfilt(data_sliced.z_re[:, 1, 0], kernel_size=med_kernel) + 1j *medfilt(data_sliced.z_im[:, 1, 0], kernel_size=med_kernel)
                            Z22 = medfilt(data_sliced.z_re[:, 1, 1], kernel_size=med_kernel) + 1j *medfilt(data_sliced.z_im[:, 1, 1], kernel_size=med_kernel)
                            Z_load = 50

                            # Calculate the input impedance
                            z_in = Z11 - np.multiply(Z12, Z21) / (Z22 + Z_load)
                            z_out = Z22 - np.multiply(Z12, Z21) / (Z11 + Z_load)
                            #z_in = abs(Z11-Z12)
                            #z_in = medfilt(z_in, kernel_size=17)
                            ax.plot(data_sliced.f, abs(z_in), color=colors[color_count],
                                    linestyle = '-', label = f'Z_in_mag{p_type}_{mm}{nn}: {dev.filename}') 
                            ax.plot(data_sliced.f, abs(z_out), color=colors[color_count],
                                    linestyle = ':', label = f'Z_out_mag{p_type}_{mm}{nn}: {dev.filename}')
                            
                        elif p_type == 'power':
                                    forward_power = np.square(np.abs(data_sliced.s[:,0,0])) + np.square(np.abs(data_sliced.s[:,0,1]))
                                    reverse_power = np.square(np.abs(data_sliced.s[:,1,0])) + np.square(np.abs(data_sliced.s[:,1,1]))
                                    
                                    ax.plot(data_sliced.f, forward_power, color=colors[color_count],
                                            linestyle = '-', label = f'Forward_power_mag{p_type}: {dev.filename}') 
                                    ax.plot(data_sliced.f, reverse_power, color=colors[color_count],
                                            linestyle = ':', label = f'Reverse_power_mag{p_type}: {dev.filename}')
                                
                        
                        else:
                            p_data = getattr(data_sliced, p_type)[:, mm-1, nn-1]
                            #apply median filter to the data to smooth it
                            p_data_smoothed = p_data#medfilt(p_data, kernel_size=23)
                            
                            ax.plot(data_sliced.f, p_data_smoothed, color=colors[color_count],
                                    linestyle = line_obj, label = f'{p_type}_{mm}{nn}: {dev.filename}')   
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

def fourier_filter(s2p_files_copy, threshold = [1.8e-8,2.2e-8]):
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
                
                # Perform the Fourier transform
                fourier = fft.fft(s_params_uniform)
                
                # Generate the frequencies for the Fourier transform
                freqs = fft.fftfreq(len(s_params_uniform), d=f_uniform[1] - f_uniform[0])

                # Compute the amplitudes
                amplitudes = np.abs(fourier)
                sum_freqs += amplitudes

                # Apply the bandstop filter adn plot the filtered data
                filtered_fourier = np.copy(fourier)
                filtered_fourier[(np.abs(freqs) > threshold[0]) & (np.abs(freqs) < threshold[1])] = 0
                sum_freqs_filtered += np.abs(filtered_fourier)
                
                # Perform the inverse Fourier transform
                filtered_s_params = fft.ifft(filtered_fourier)

                # Interpolate the filtered data back onto the original frequency grid
                interp_func = interp.interp1d(f_uniform, filtered_s_params)
                filtered_s_params_original = interp_func(s.network.f)

                # Write the filtered s-parameters back into the original data
                s.network.s[:,m-1,n-1] = filtered_s_params_original

    plt.figure(figsize=(10, 5))
    plt.plot(freqs,sum_freqs, color='blue',linestyle =':')
    plt.plot(freqs,sum_freqs_filtered, color='green')
    for t in threshold:
                        plt.axvline(x=t, color='red')
                        plt.axvline(x=-t, color='red')
    plt.yscale('log')
    return s2p_files_copy



# Define the path to the directory containing the VNA data
#directory = ('/Users/horatiocox/Desktop/VNA_Analysis/mag_angle_260424/')

directory = ('/Users/horatiocox/Desktop/VNA_Analysis/CPW_mem_oscillator_220524/Memristor/')
#directory = ('/Users/horatiocox/Desktop/VNA_Analysis/CPW_mem_oscillator_220524/Oscillator/')

# Import the data from the VNA files
s2p_files = import_data(directory)
# remove duplicate frequency points from all the thru data to prevent errors with skrf functions
for f in s2p_files:
    print(f.run,f.label,f.filename)
    indx = f.network.frequency.drop_non_monotonic_increasing()
    # Remove the corresponding entries from the s-parameters array
    unique_s = np.delete(f.network.s, indx, axis=0)
    unique_z0 = np.delete(f.network.z0, indx, axis=0)
    # Assign the unique s-parameters array back to the network object
    f.network.s = unique_s
    f.network.z0 = unique_z0
    



## Step 1 - fourier filter the initial files to remove noise
s2p_filt = copy.deepcopy(s2p_files)
s2p_filt = fourier_filter(s2p_filt, threshold = [1.8e-8,2.2e-8]) #apply to deepcopy to avoid modifying the original data
dev_subs = subgen(s2p_files, run_nums =[[1,2,3,4,5,6], [31,27,23,35,39,41], [47,42,50]   ] )
dev_subs_filt = subgen(s2p_filt, run_nums =[[1,2,3,4,5,6], [31,27,23,35,39,41], [47,42,50]   ] )
#dev_subs = subgen(s2p_files, run_nums =[[1,2], [23], [42]   ] )


#-------------------Grouping-------------------
# Select the On Wafer Calibration files to be used
ISS_thru = [s for s in s2p_filt if s.state == 'thru' and s.wafer_number == 0]
cal_thru = [s for s in s2p_filt if s.state == 'thru' and s.wafer_number != 0]
cal_open = [s for s in s2p_filt if s.state == 'open' or s.state == 'opensig']# or s.state == 'opennarrow' or s.state == 'openverynarrow']
cal_short = [s for s in s2p_filt if s.state == 'short']
    
# Group files by their state so I can plot and compare states
pristine = [s for s in s2p_filt if s.state == 'pristine']
formed = [s for s in s2p_filt if s.state in ['formed', 'smallform', 'fullform']]

# Group files with the same [r_number, c_number] values together so I can plot each device in all its states on one graph
# Stores a list of S2PFile objects for each device in a dictionary, "dev", with the key being the device's [r_number, c_number] values
#thus dev['11'] will have all the data for the device in row 1 column 1
dev = {}
for s in s2p_files:
    key = f"r{s.dev_row}c{s.dev_col}"
    if key in dev:
        dev[key].append(s)
    else:
        dev[key] = [s]




#-------------------De-Embedding-------------------
print('open_short_thru',len(cal_open),len(cal_short),len(cal_thru))
 #whether to plot the de-embedding results
OS = calibration_OS(cal_open, cal_short, cal_thru, plot_cal = False) #calibration object outputted from all the on wafer measurements
TX = calibration_2x(cal_thru, plot_cal = False) #calibration object outputted from all the on wafer measurements



#-------------------Plotting-------------------
#def keyplot(OS, dev, dev_selection = None, sub_set = [], y_range = None,
           # x_range = slice(0,-1), log_x = False, plot_type = 'S_db',m_port=[2], n_port=[1]):        
#'inputz' - plots the input impedance of the device

x_range_input = "0.1-20ghz"#slice(0,-1)#"0.02-0.8ghz" #
y_range_input = [0,200]#None#[0,200]
# def subplot(dev_subs = [], cal_in = [], y_range = None,
#             x_range = slice(0,-1), log_x = False, plot_type = 'S_db',m_port=[2], n_port=[1], deembed_data = True):

# def subgen(s2p_files, run_nums = [[],[],[]]):




# fig1 = subplot(dev_subs = dev_subs, cal_in = OS, plot_type = ['power', 'inputz', 'S_db'],
#                         log_x=False, m_port=[2], n_port=[1],deembed_data = True, y_range=y_range_input, x_range=x_range_input)
# fig2 = subplot(dev_subs = dev_subs, cal_in = OS, plot_type = ['power', 'inputz', 'S_db'],
#                         log_x=False, m_port=[2], n_port=[1],deembed_data = False, y_range=y_range_input)


fig3 = subplot(dev_subs = dev_subs_filt, cal_in = TX, plot_type = ['inputz'],
                        log_x=True, m_port=[2], n_port=[1],deembed_data = True, y_range=y_range_input, x_range=x_range_input)
fig3 = subplot(dev_subs = dev_subs_filt, cal_in = OS, plot_type = ['inputz'],
                        log_x=True, m_port=[2], n_port=[1],deembed_data = True, y_range=y_range_input, x_range=x_range_input)
fig4 = subplot(dev_subs = dev_subs_filt, cal_in = OS, plot_type = ['inputz'],
                        log_x=True, m_port=[2], n_port=[1],deembed_data = False, y_range=y_range_input, x_range=x_range_input)



# fig_dc, ax_dc = subplot(dev_subs = dev_subs, cal_in = OS, plot_type = 'inputz',
#                         log_x=True, m_port=[2], n_port=[1],deembed_data = True, y_range=y_range_input, x_range=x_range_input)
# fig_dc2, ax_dc2 = subplot(dev_subs = dev_subs, cal_in = OS, plot_type = 'inputz',
#                         log_x=True, m_port=[2], n_port=[1],deembed_data = False, y_range=y_range_input, x_range=x_range_input)


plt.show()



# fig_dev, ax_dev = keyplot(dev, cal_in = OS, dev_selection = ['r2c1'],sub_set = ['formed_pos','formed_0','formed_neg'], plot_type = 'inputz',
#                                     log_x=False, m_port=[2], n_port=[1],deembed_data = True, y_range=y_range_input, x_range=x_range_input)
# fig_dev, ax_dev = keyplot(dev, cal_in = TX, dev_selection = ['r2c1'],sub_set = ['formed_pos','formed_0','formed_neg'], plot_type = 'inputz',
#                                     log_x=False, m_port=[2], n_port=[1],deembed_data = True, y_range=y_range_input, x_range=x_range_input)
# fig_dev, ax_dev = keyplot(dev, cal_in = TX, dev_selection = ['r2c1'],sub_set = ['formed_pos','formed_0','formed_neg'], plot_type = 'inputz',
#                                     log_x=False, m_port=[2], n_port=[1],deembed_data = False, y_range=y_range_input, x_range=x_range_input)







#'_set','_set1', '_set2', '_set3','_set4', '_set5', '_set6','_set7', '_set8', '_set9',  formed_0dc','formed_pos0.0','c1_pristine.','pristine_pos0.2

#-------------------Network Set-------------------
#takes a dictionary or list of networks as its input and converts to a network set object that can give errors etc for repeated measurements
# Convert the dev dictionary of lists of s2p files into a dictionary of lists of networks

#need filtering, probably taken from the keyplot function to select the devices and states to include in the network set
# probably actually just want to make this another keyplot function that takes the network set as an input and then plots the data

# dev_networks = {}
# for key, value in dev.items():
#     dev_networks[key] = [s.network for s in value]

# ro_ns = NetworkSet(dev_networks, name='ro set')



#--------------------------------------------------
#--------------------------------------------------




# %%
import numpy as np
np.linspace(0,1,20)



# %%
