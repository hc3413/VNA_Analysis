#source VNAenv/bin/activate (launching and exiting the virtual environment containing the required modules, stored in the working directory for VNA_Analysis)
#VNAenv/bin/python your_script.py - for running a script in the virtual environment
#source deactivate

import numpy as np
import pandas as pd
import skrf as rf
from skrf.calibration import OpenShort, SplitTee, AdmittanceCancel
import matplotlib.pyplot as plt
import os
import datetime
from operator import itemgetter
from dataclasses import dataclass
import re
from typing import Tuple
import matplotlib.cm as cm


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


def calibration(s2p_files, open, short, thru, OS_plot):
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
            
            OS_th = [] #initiate list to store the de-embedded thru data for each open/short pair
            total_error = 0 #initialize variable to store the total error for each open/short pair
            for count_th,th in enumerate(thru, start=0):
                if OS_plot:
                    plt.figure(f'Open Short De-embedding')
                    th.network.plot_s_mag(m=1, n=0, color=colors[color_count], linestyle='dashed',label = f'Raw: {th.label}; {sh.label}; O{count_o}S{count_sh}T{count_th}')  # Plot only s21 with colorblind colormap
                    cal.deembed(th.network).plot_s_mag(m=1, n=0, color=colors[color_count], label = f'OS: {th.label}; {sh.label}; O{count_o}S{count_sh}T{count_th}')  # Plot only s21 with colorblind colormap
                #store the error for each de-embedding protocol in the respective array index
                error_value = np.abs(np.sum(cal.deembed(th.network).s[:,1,0]))
                total_error += error_value
                color_count += 1
            error.append(total_error)
            print(total_error)
    
    #Find the best de-embedding protocol
    min_index = error.index(min(error))
    max_index = error.index(max(error))

    print(f"Best de-embedding protocol: dm[{min_index}]={min(error)}, worst = {max(error)}")
    
    #Plot the best de-embedding protocol
    if OS_plot:
        plt.figure('Best De-embedding Protocol')
        thru[0].network.plot_s_mag(m=1, n=0, color='red', linestyle='dashed',label = f'Raw')  # Plot only s21 with colorblind colormap
        dm[min_index].deembed(thru[0].network).plot_s_mag(m=1, n=0, color='green', label = f'Best OS:dm[{min_index}]')  # Plot only s21 with colorblind colormap
        dm[max_index].deembed(thru[0].network).plot_s_mag(m=1, n=0, color='blue', label = f'Worst OS:dm[{max_index}]')  # Plot only s21 with colorblind colormap
   
    #only return the best de-embedding protocol
    return dm[min_index]



def keyplot(OS, dev, dev_selection_in = [], subset_in = [], x_limits = [0, 20e9], y_limits = [0, 1],
            slice_range = slice(0,-1), log_x = False, db = False, plot_type = 'S'):
    # Function to plot the data for the selected devices and states
    # A number of inputs are given default values so they can be omitted from the function input if not required as they are quite standard
    # The default values also means that you can call them by name and not require the perfect ordring of the inputs
    # Needs dev_selection_in/subset_in to be local to function hence the _in suffix
    #Plot type can be S, Z, Y, T, ABCD, or Smith
    
    plot_type = plot_type.lower() #remove case sensitivity
    
    # Check if the dev_selection_in is empty or 'all' and set the dev_selection_in to all devices if so
    if not dev_selection_in or dev_selection_in == 'all':
        dev_selection_in = dev.keys()

    # If subset_in is empty or 'all' then set the subset_check to always return True 
    if not subset_in or subset_in == 'all':
        subset_check = lambda x: True
    # Else check if the filename contains any of the subset_in strings returning true/false to plot/not plot
    else:
        subset_check = lambda x: any(m in x.filename.lower() for m in subset_in)
   
    # Loop over the selected devices    
    for key in dev_selection_in:
        if key in dev: #extract the s2p files for the selected devices
            value = dev[key] 
            
            fig, ax = plt.subplots()
            ax.set_title(f'Device_{key}')
            
            
            #ax.set_xlim(x_limits)
            #ax.set_ylim(y_limits)
            num_colors = sum(1 for r in value if subset_check(r)) #count the number of files that match the subset criteria to set the number of colors            
            colors = plt.cm.jet(np.linspace(0,1,num_colors))
            color_count = 0
            
            # Loop over the s2p files for each select device plotting all that match the subset criteria
            for r in value:
                if subset_check(r):
                    #ax.plot(r.network.s_mag[:, 1, 0], color=colors[color_count], linestyle='dashed',label = f'Raw: {r.label}')  # Plot only s21 with colorblind colormap
                    data = OS.deembed(r.network)
                    #slice the data to plot selected frequency range (necessary for the smith chart where you can't change axis limits to do this)
                    data_sliced = data[slice_range]
                    
                    ####**********I think that the data_sliced is a network object 
                    # ####and the transform maybe only works on the s parameter of the network object???????
                    #####
                    
                    if plot_type == 's':
                        ax.plot(data_sliced.f, data_sliced.s_db[:, 1, 0], color=colors[color_count], label = f'OS: {r.filename}')
                        color_count += 1
                        
                    elif plot_type == 'z':
                        #data_sliced_transformed = rf.s2z(data_sliced)
                        ax.plot(data_sliced.f, data_sliced.z_mag[:, 1, 0], color=colors[color_count], label = f'OS: {r.filename}')
                        color_count += 1
                        
                    elif plot_type == 'y':
                        data_sliced_transformed = rf.s2y(data_sliced)   
                        ax.plot(np.log10(data_sliced_transformed.f), data_sliced_transformed.s_db[:, 1, 0], color=colors[color_count], label = f'OS: {r.filename}')
                        color_count += 1
                        
                    elif plot_type == 't':
                        data_sliced_transformed = rf.s2t(data_sliced)
                        ax.plot(np.log10(data_sliced_transformed.f), data_sliced_transformed.s_db[:, 1, 0], color=colors[color_count], label = f'OS: {r.filename}')
                        color_count += 1
                        
                    elif plot_type == 'abcd':
                        data_sliced_transformed = rf.s2a(data_sliced)
                        ax.plot(np.log10(data_sliced_transformed.f), data_sliced_transformed.s_db[:, 1, 0], color=colors[color_count], label = f'OS: {r.filename}')
                        color_count += 1
                        
                    elif plot_type == 'smith':
                        data_sliced_transformed = data_sliced
                        data_sliced_transformed.plot_s_smith(m=1,n=1,draw_labels=True, color=colors[color_count], label = f'OS: {r.filename}') 
                        color_count += 1
 
            if log_x:
                        ax.set_xscale('log')
                        
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude (dB)')   
            #ax.legend(loc='upper left')
    return fig, ax






# Define the path to the directory containing the VNA data
#directory = ('/Users/horatiocox/Desktop/VNA_Analysis/mag_angle_260424/')

directory = ('/Users/horatiocox/Desktop/VNA_Analysis/CPW_mem_oscillator_220524/Memristor/')
#directory = ('/Users/horatiocox/Desktop/VNA_Analysis/CPW_mem_oscillator_220524/Oscillator/')

# Import the data from the VNA files
s2p_files = import_data(directory)
for f in s2p_files:
    print(f.filename,f.run, f.label)


#-------------------Grouping-------------------
# Select the On Wafer Calibration files to be used
ISS_thru = [s for s in s2p_files if s.state == 'thru' and s.wafer_number == 0]
cal_thru = [s for s in s2p_files if s.state == 'thru' and s.wafer_number != 0]
cal_open = [s for s in s2p_files if s.state == 'open' or s.state == 'opensig']# or s.state == 'opennarrow' or s.state == 'openverynarrow']
cal_short = [s for s in s2p_files if s.state == 'short']


# Group files by their state so I can plot and compare states
pristine = [s for s in s2p_files if s.state == 'pristine']
formed = [s for s in s2p_files if s.state in ['formed', 'smallform', 'fullform']]

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
OS_plot = False #whether to plot the de-embedding results
OS = calibration(s2p_files, cal_open, cal_short, cal_thru,OS_plot) #calibration object outputted from all the on wafer measurements

#-------------------Plotting-------------------

#########create a class which is a plotobject containing all of the below where the class puts in default values if they aren't specified then I can 
#########just set the class object for each plot and then feed the class object into keyplot to generate the plot
#plotobject = PlotObject(OS, dev, dev_selection, subset, x_limits, y_limits, slice)
#fig, ax = keyplot(plotobject)


dev_selection = ['r2c1'] #specific devices to plot (all/empty plots all devices)
subset = ['formed']; #Select only runs that contain any of the subset strings in their (all/empty plots all states)
x_limits = [0, 20e9] #limits for the x-axis of the plot
y_limits = [0, 1] #limits for the y-axis of the plot
#keyplot(OS, dev, dev_selection_in = [], subset_in = [], x_limits = [0, 20e9], y_limits = [0, 1],
            #slice = [0, -1], log_x = False, db = False, plot_type = 'S'):
#slice_range: slice the data to remove the low frequency noise - default = no slicing - input form should be either '10-20ghz' or slice(1:10)

fig_pristine, ax_pristine = keyplot(OS, dev, dev_selection_in = dev_selection,subset_in = ['pristine'], plot_type = 'S',
                                    log_x=True)
fig_formed, ax_formed = keyplot(OS, dev, dev_selection_in = dev_selection,subset_in = ['formed'], plot_type = 'z', slice_range = '0.01-20ghz')
plt.show()

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
