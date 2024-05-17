import numpy as np
import skrf as rf
from skrf.calibration import OpenShort, SplitTee, AdmittanceCancel
import matplotlib.pyplot as plt

path = ("/Users/horatiocox/Desktop/VNA_Analysis/mag_angle_260424/")

cal_thrutaper = rf.Network(path + 'Wafer2_r10_c1_thrutaper_1.S2P')
cal_open = rf.Network(path + 'Wafer2_r10_c5_open_1.S2P')
cal_opensig = rf.Network(path + 'Wafer2_r10_c9_opensig_1.S2P')
cal_short = rf.Network(path + 'Wafer2_r10_c11_shortall_1.S2P')
cal_thrustraight = rf.Network(path + 'Wafer2_r10_c13_thru_1.S2P')



dm = OpenShort(dummy_open=cal_open, dummy_short=cal_short, name='tutorial')
OS_thru = dm.deembed(cal_thrutaper)

dmST= SplitTee(dummy_thru=cal_thrutaper, name='tutorial2')
ST_thru = dmST.deembed(cal_thrutaper)

dmAC = AdmittanceCancel(dummy_thru = cal_thrutaper, name='tutorial3')
AC_thru = dmAC.deembed(cal_thrutaper)

# Plot before and after de-embedding on the Thru device on wafer

plt.figure("Open Short De-embedding on the on-wafer thru measurement")
cal_thrutaper.plot_s_db(m=1, n=0, color='red', label = 'raw')  # Plot only s21 and s12 with red color
OS_thru.plot_s_db(m=1, n=0, color='green', label = 'OS')  # Plot only s21 and s12 with green color
ST_thru.plot_s_db(m=1, n=0, color='purple', label = 'SplitTee')  
AC_thru.plot_s_db(m=1, n=0, color='blue', label = 'AC')
#plt.figure("SplitPi De-embedding on the on-wafer thru measurement")
#cal_thrutaper.plot_s_mag(m=1, n=0, color='red', label = 'raw')  # Plot only s21 and s12 with red color






# Now looking at Memristors
pristine_mem1 = rf.Network(path + 'Wafer2_r1_c2_pristine_1.S2P')
OS_pristine_mem1 = dm.deembed(pristine_mem1)
ST_pristine_mem1 = dmST.deembed(pristine_mem1)
AC_pristine_mem1 = dmAC.deembed(pristine_mem1)


plt.figure("DB: Open Short De-embedding on pristine memristor")
pristine_mem1.plot_s_mag(m=1, n=0, color='red',label = 'raw')  # Plot only s21 and s12 with red color
OS_pristine_mem1.plot_s_mag(m=1, n=0, color='green', label = 'OS')  # Plot only s21 and s12 with green color
#ST_pristine_mem1.plot_s_mag(m=1, n=0, color='purple', label = 'SplitTee')  
AC_pristine_mem1.plot_s_mag(m=1, n=0, color='blue', label = 'AC')


#plt.figure("Impedance Params: Open Short De-embedding on pristine memristor")
#pristine_mem1.plot_z_im(m=1, n=0, color='red',label = 'raw')
#OS_pristine_mem1.plot_z_im(m=1, n=0, color='green', label = 'de-embedded')


#Comparing Memristors de-embedded with different methods
mem12_p = rf.Network(path + 'Wafer2_r1_c2_pristine_1.S2P')
mem13_p = rf.Network(path + 'Wafer2_r1_c3_pristine_1.S2P')
mem14_p = rf.Network(path + 'Wafer2_r1_c4_pristine_1.S2P')
mem15_p = rf.Network(path + 'Wafer2_r1_c5_pristine_1.S2P')
mem16_p = rf.Network(path + 'Wafer2_r1_c6_pristine_1.S2P')
mem17_p = rf.Network(path + 'Wafer2_r1_c7_pristine_1.S2P')
mem18_p = rf.Network(path + 'Wafer2_r1_c8_pristine_1.S2P')
mem19_p = rf.Network(path + 'Wafer2_r1_c9_pristine_1.S2P')

OS_mem12_p = dm.deembed(mem12_p)
OS_mem13_p = dm.deembed(mem13_p)
OS_mem14_p = dm.deembed(mem14_p)
OS_mem15_p = dm.deembed(mem15_p)
OS_mem16_p = dm.deembed(mem16_p)
OS_mem17_p = dm.deembed(mem17_p)
OS_mem18_p = dm.deembed(mem18_p)
OS_mem19_p = dm.deembed(mem19_p)

AC_mem12_p = dmAC.deembed(mem12_p)
AC_mem13_p = dmAC.deembed(mem13_p)
AC_mem14_p = dmAC.deembed(mem14_p)
AC_mem15_p = dmAC.deembed(mem15_p)
AC_mem16_p = dmAC.deembed(mem16_p)
AC_mem17_p = dmAC.deembed(mem17_p)
AC_mem18_p = dmAC.deembed(mem18_p)
AC_mem19_p = dmAC.deembed(mem19_p)

plt.figure("OS De-embedding on 8 pristine memristors")
OS_mem12_p.plot_s_db(m=1, n=0, color='red',label = 'mem12') 
OS_mem13_p.plot_s_db(m=1, n=0, color='green',label = 'mem13')
OS_mem14_p.plot_s_db(m=1, n=0, color='purple',label = 'mem14')
OS_mem15_p.plot_s_db(m=1, n=0, color='blue',label = 'mem15')
OS_mem16_p.plot_s_db(m=1, n=0, color='orange',label = 'mem16')
OS_mem17_p.plot_s_db(m=1, n=0, color='yellow',label = 'mem17')
OS_mem18_p.plot_s_db(m=1, n=0, color='black',label = 'mem18')
OS_mem19_p.plot_s_db(m=1, n=0, color='pink',label = 'mem19')

plt.figure("AC De-embedding on 8 pristine memristors")
AC_mem12_p.plot_s_db(m=1, n=0, color='red',label = 'mem12')
AC_mem13_p.plot_s_db(m=1, n=0, color='green',label = 'mem13')
AC_mem14_p.plot_s_db(m=1, n=0, color='purple',label = 'mem14')
AC_mem15_p.plot_s_db(m=1, n=0, color='blue',label = 'mem15')
AC_mem16_p.plot_s_db(m=1, n=0, color='orange',label = 'mem16')
AC_mem17_p.plot_s_db(m=1, n=0, color='yellow',label = 'mem17')
AC_mem18_p.plot_s_db(m=1, n=0, color='black',label = 'mem18')
AC_mem19_p.plot_s_db(m=1, n=0, color='pink',label = 'mem19')



plt.show()

