#source VNAenv/bin/activate (launching and exiting the virtual environment containing the required modules, stored in the working directory for VNA_Analysis)
#VNAenv/bin/python your_script.py - for running a script in the virtual environment
#source deactivate

import numpy as np
import skrf as rf
from skrf.calibration import OpenShort, SplitTee, AdmittanceCancel
import matplotlib.pyplot as plt
import os
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
    
    @property #generating a unique label for the device that can be used in plots to identify it
    def label(self) -> str:
        return f"{self.state}_{self.dev_row}_{self.dev_col}"

############ Start of Functions

def import_data(data_path: str):
    # Function to import the data from the VNA files and store it in a list of S2PFile objects
    
    # Get a list of all .s2p files in the directory
    files = [f for f in os.listdir(data_path) if f.endswith('.S2P')]
# Read each file and store it in a list of S2PFile objects
    s2p_files = []
    for f in files:
        network = rf.Network(data_path + f)
        #state keywords to look for in filename, by default thru/short etc means the tapered version and it will have "notaper" in the filename if it is the straight thru
        #note that smaller words that are substrings of larger words should be placed after the larger words in the list
        keywords = ['smallform', 'fullform', 'opensig','thrunotaper', 'opennotaper','shortnotaper', 'thruISS','pristine', 'formed', 'thru', 'open', 'short']
        state = next((x for x in keywords if x in f.lower()), None) #returns the first keyword found in the state value, stops as soon as the first keyword is found
        # Extract the row, colum and wafer numbers from the filename (e.g. wafer 1 r1_c11) and store into position variable
        wafer_number = re.findall(r'Wafer(\d)', f, re.IGNORECASE)
        r_number = re.findall(r'_r(\d{1,2})_', f, re.IGNORECASE)
        c_number = re.findall(r'_c(\d{1,2})_', f, re.IGNORECASE)
        #store the network and its associated metadata in the S2PFile object grouped toegher in the s2p_files list
        if not wafer_number or not r_number or not c_number:
            s2p_files.append(S2PFile(network, f, int(0), None, None, state)) #ISS is classes as wafer zero
        else:
            s2p_files.append(S2PFile(network, f, int(wafer_number[0]), int(r_number[0]), int(c_number[0]), state.lower()))
    return s2p_files





def calibration(s2p_files, open, short, thru):
    # Function to generate an OpenShort De-embedding calibration and apply it to the on-wafer thru measurements to check its effectiveness
    # Inputs multiple open and short measurements and compares them to see which give the most effective de-embedding
    # Outputs a list of de-embedding data and plots of the thru measurements before and after de-embedding for each protocol
    
    #**********need to edit this so that it optimises trying all combinations of open and short minimising the error in the thru!!!!****
    #**************************************
    #**************************************
    dm = [] #initialize an empty list to store the de-embedded data
    num_colors = len(open)*len(short)*len(thru)
    count = 0
    colors = plt.cm.jet(np.linspace(0,1,num_colors))
    for o in open:
        for sh in short:
            cal = OpenShort(dummy_open=o.network, dummy_short=sh.network, name='OpenShort Calibration')
            dm.append(cal)
            for th in thru:
                plt.figure(f'Open Short De-embedding')
                th.network.plot_s_mag(m=1, n=0, color=colors[count], linestyle='dashed',label = f'raw_{th.label}_{sh.label}_{count}')  # Plot only s21 with red color and label it as raw_x_x
                cal.deembed(th.network).plot_s_mag(m=1, n=0, color=colors[count], label = f'OS_{th.label}_{sh.label}_{count}')  # Plot only s21 with colorblind colormap
                
                # Calculate the error and its integral
                error = cal.deembed(th.network)
                print('error',{count},np.abs(np.sum(error.s[:11,1,0]))) #outputs the error along with the respective position in the dm array of the deembeding protocol associated with it. 
                
                
                count += 1
    plt.show()
    return dm



# def calibration(s2p_files, open, short):
# #on assumption of multiple measurements of cal_open and cal_short i will store them in dm_x and you can test the different calibrations effectiveness
# dm = [] #initialize an empty list to store the de-embedded data
# cal_thru_OS = [] #initialize an empty list to store the de-embedded data applied to the thru for reference to see how effective de-embedding is
# for x in range(len(cal_open)):
#     dm.append(OpenShort(dummy_open=cal_open[x].network, dummy_short=cal_short[x].network, name='OpenShort Calibration'))
#     cal_thru_OS.append(dm[x].deembed(cal_thru[x].network))
  
#     num_colors = len(s2p_files)
#     count = 0
#     colors = plt.cm.jet(np.linspace(0,1,num_colors))
#     for s in s2p_files:
#         if s.dev_row == 1:
#             print(s.dev_row, s.dev_col, s.state)
#             plt.figure(f'Open Short De-embedding on Device_{s.dev_row}_{s.dev_col}, cal_num = {x}')
#             s.network.plot_s_db(m=1, n=0, color=colors[count], linestyle='dashed',label = f'raw_{s.filename[-9:-4]}_{s.state}')  # Plot only s21 with red color and label it as raw_x_x
#             dm[x].deembed(s.network).plot_s_db(m=1, n=0, color=colors[count], label = f'OS_{s.filename[-9:-4]}_{s.state}')  # Plot only s21 with colorblind colormap
#         count += 1
    
#     plt.figure(f'Open Short De-embedding on the on-wafer thru measurement, cal_num = {x}')
#     cal_thru[x].network.plot_s_db(m=1, n=0, color='red', label = f'raw_{x}')  # Plot only s21 with red color and label it as raw_x_x
#     cal_thru_OS[x].plot_s_db(m=1, n=0, color='green', label = f'OS_{x}')  # Plot only s21 with green color






# Define the path to the data
path = ("/Users/horatiocox/Desktop/VNA_Analysis/mag_angle_260424/")

# Import the data from the VNA files
s2p_files = import_data(path)

#-------------------Grouping-------------------
# Select the On Wafer Calibration files to be used
ISS_thru = [s for s in s2p_files if s.state == 'thru' and s.wafer_number == 0]
cal_thru = [s for s in s2p_files if s.state == 'thru' and s.wafer_number != 0 or s.state == 'thrunotaper']
cal_open = [s for s in s2p_files if s.state == 'open' or s.state == 'opensig']
cal_short = [s for s in s2p_files if s.state == 'short']

# Group files by their state so I can plot and compare states
pristine = [s for s in s2p_files if s.state == 'pristine']
formed = [s for s in s2p_files if s.state in ['formed', 'smallform', 'fullform']]

# Group files with the same [r_number, c_number] values together so I can plot each device in all its states on one graph
# Stores a list of S2PFile objects for each device in a dictionary, "dev", with the key being the device's [r_number, c_number] values
#thus dev['11'] will have all the data for the device in row 1 column 1
dev = {}
for s in s2p_files:
    key = f"{s.dev_row}{s.dev_col}"
    if key in dev:
        dev[key].append(s)
    else:
        dev[key] = [s]

#-------------------De-Embedding-------------------
print('open_short_thru',len(cal_open),len(cal_short),len(cal_thru))
OS = calibration(s2p_files, cal_open, cal_short, cal_thru) #calibration object outputted from all the on wafer measurements

print(OS)


# Compare Memristors in different states
# print('open_',len(cal_open),'short_',len(cal_short),'thru_',len(cal_thru))
# for x in range(len(cal_open)):
#     dm.append(OpenShort(dummy_open=cal_open[x].network, dummy_short=cal_short[x].network, name='OpenShort Calibration'))
#     cal_thru_OS.append(dm[x].deembed(cal_thru[x].network))
  
#     n = len(s2p_files)
#     count = 0
#     colors = plt.cm.jet(np.linspace(0,1,n))
#     for s in s2p_files:
#         if s.dev_row == 1:
#             print(s.dev_row, s.dev_col, s.state)
#             plt.figure(f'Open Short De-embedding on {s.state} Device_{s.dev_row}_{s.dev_col}, cal_num = {x}')
#             s.network.plot_s_db(m=1, n=0, color='red', linestyle='dashed',label = f'raw_{s.filename[-9:-4]}')  # Plot only s21 with red color and label it as raw_x_x
#             dm[x].deembed(s.network).plot_s_db(m=1, n=0, color=colors[count], label = f'OS_{s.filename[-9:-4]}')  # Plot only s21 with colorblind colormap
#         count += 1
    
#     plt.figure(f'Open Short De-embedding on the on-wafer thru measurement, cal_num = {x}')
#     cal_thru[x].network.plot_s_db(m=1, n=0, color='red', label = f'raw_{x}')  # Plot only s21 with red color and label it as raw_x_x
#     cal_thru_OS[x].plot_s_db(m=1, n=0, color='green', label = f'OS_{x}')  # Plot only s21 with green color





# %%
import numpy as np
np.linspace(0,1,20)



# %%
