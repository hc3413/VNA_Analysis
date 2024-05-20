import numpy as np
import skrf as rf
from skrf.calibration import OpenShort, SplitTee, AdmittanceCancel
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
import re
from typing import Tuple

# Define a class to store the VNA data alongside its ascociated with the filename, device index, and state
@dataclass
class S2PFile:
    network: rf.Network
    filename: str
    wafer_number: int #wafer number 1 or 2
    dev_row: int #row of device on wafer
    dev_col: int #column of device on wafer
    state: str


# Define the path to the data
path = ("/Users/horatiocox/Desktop/VNA_Analysis/mag_angle_260424/")

# Get a list of all .s2p files in the directory
files = [f for f in os.listdir(path) if f.endswith('.S2P')]

# Read each file and store it in a list of S2PFile objects
s2p_files = []
for f in files:
    network = rf.Network(path + f)
    #state keywords to look for in filename, by default thru/short etc means the tapered version and it will have "notaper" in the filename if it is the straight thru
    keywords = ['pristine', 'formed', 'smallform', 'fullform', 'thru', 'opensig', 'open', 'short','thrunotaper', 'opennotaper','shortnotaper', 'thruISS']
    state = next((x for x in keywords if x in f.lower()), None) #returns the first keyword found in the state value, stops as soon as the first keyword is found
    # Extract the row, colum and wafer numbers from the filename (e.g. wafer 1 r1_c11) and store into position variable
    wafer_number = re.findall(r'Wafer(\d)', f, re.IGNORECASE)
    r_number = re.findall(r'_r(\d{1,2})_', f, re.IGNORECASE)
    c_number = re.findall(r'_c(\d{1,2})_', f, re.IGNORECASE)
    #store the network and its associated metadata in the S2PFile object grouped toegher in the s2p_files list
    if not wafer_number or not r_number or not c_number:
        s2p_files.append(S2PFile(network, f, None, None, None, state))
    else:
        s2p_files.append(S2PFile(network, f, int(wafer_number[0]), int(r_number[0]), int(c_number[0]), state.lower()))
        
    
for s in s2p_files:
    print(s.state)


# Get the on wafer calibration data in a list of S2PFile objects
cal_thru = [s for s in s2p_files if s.state == 'thru']
cal_open = [s for s in s2p_files if s.state == 'open']
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

#on assumption of multiple measurements of cal_open and cal_short i will store them in dm_x and you can test the different calibrations effectiveness
dm = [] #initialize an empty list to store the de-embedded data
cal_thru_OS = [] #initialize an empty list to store the de-embedded data applied to the thru for reference to see how effective de-embedding is
plt.figure("Open Short De-embedding on the on-wafer thru measurement")
print(len(cal_open),len(cal_short),len(cal_thru))
for x in range(len(cal_open)):
    dm[x] = OpenShort(dummy_open=cal_open[x].network, dummy_short=cal_short[x].network, name='OpenShort Calibration')
    cal_thru_OS[x] = dm[x].deembed(cal_thru[x].network)
  
    cal_thru[x].network.plot_s_db(m=1, n=0, color='red', label = f'raw_{x}')  # Plot only s21 with red color and label it as raw_x_x
    cal_thru_OS[x].plot_s_db(m=1, n=0, color='green', label = f'OS_{x}')  # Plot only s21 with green color

plt.figure("De-embedded data for the devices")
print(cal_open)

plt.show()

