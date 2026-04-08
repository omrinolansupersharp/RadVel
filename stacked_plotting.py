#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from astropy.time import Time
import astropy.units as u
from astropy.timeseries import LombScargle

default_settings = {
    'font.size': 16,
    'axes.linewidth': 0.8,
    'xtick.major.size': 3.5,
    'xtick.major.width': 1,
    'ytick.major.size': 3.5,
    'ytick.major.width': 1
}


initial_settings = {
    'font.size': 22,
    'axes.linewidth': 1.25,
    'xtick.major.size': 5,
    'xtick.major.width': 1.25,
    'ytick.major.size': 5,
    'ytick.major.width': 1.25
}
plt.rcParams.update(initial_settings)


# In[ ]:


p_lines = []
spec_lines = []
#comment out the lines that we don't want to use
#pollutant lines
p_lines.append(("MgII_4481",4481.130,True)) #strong magnesium line

p_lines.append(("CaII_3933", 3933.663, False)) #only use for MIKE data
p_lines.append(("CaII_4226",4226.727, True)) #good
p_lines.append(("FeII_4923",4923.927, False)) #can't use, crosses over with interorder
p_lines.append(("FeII_5018",5018.440, False))#not there
p_lines.append(("FeII_5169",5169.033, False)) #this one gives an error when trying to fit it
p_lines.append(("SiII_5041",5041.024, True)) #don't use this
p_lines.append(("SiII_5055",5055.984, True)) #This one is good use it
p_lines.append(("SiII_5957",5957.560, True))
p_lines.append(("SiII_5978",5978.930, True))
p_lines.append(("SiII_6347",6347.100, True)) #these two are quite strong in WD1929+012
p_lines.append(("SiII_6371",6371.360, True)) #
p_lines.append(("MgI_5172",5172.683, False))
p_lines.append(("MgI_5183",5183.602, False))

p_lines.append(("MgII_4481_2",4481.327, False))
p_lines.append(("MgII_4481",4481.180,False)) #weighted combination of mg_4481 lines

p_lines.append(("MgII_7877",7877.054, True)) 
p_lines.append(("MgII_7896",7896.366, True)) #definitely not present
p_lines.append(("OI_7771",7771.944, True)) #definitely not present
 #hydrogen lines
p_lines.append(("H_4860",4860.680, False))
p_lines.append(("H_4860_2",4860.968, True))#This one gives better values
p_lines.append(("H_4860_2",4860.968, False))#Weighted combo
p_lines.append(("H_4340",4340.472,True)) #present
p_lines.append(("H_6563",6562.79 ,True)) #not present in the 2020 spectra?
#pick the spectral lines present in this white dwarf
"""


# #stacked plot creation
# #data = pd.read_csv(r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/resultfiles/WD1929/Voigt_fitting/only_mg_line/snr_cutoff_16.05612842459828/2021-04-21T03:09:52.703/4481.185.txt", sep='\t')
# #data = pd.DataFrame({"Wavelength":(gwav),"Normalized Data": (gdata/p_result), "Voigt fit": (result.best_fit/p_result), "Time": t, "SNR": snr, "Depth": depth, "RV": rv, "Error": err}) #[t],[snr], [depth], [rv], [err]])
# 
# for i in p_lines:
#     line = i[1]
#     salt_root_dir = '/data/wdplanetary/omri/Output/resultfiles/WD1929/Voigt_fitting/red_lines/corrected_for_pixel/'  # Replace this with the path to your root directory
#     # Initialize an empty list to store file paths
#     plot_files = []
#     # Walk through all directories and subdirectories
#     for root, dirs, files in os.walk(salt_root_dir):
#         # Iterate over each file in the current directory
#         for file in files:
#             # Check if the file ends with '4481.185.txt'
#             if file.endswith(f'{line}.txt'):
#                 # If it does, append the file path to the list
#                 plot_files.append(os.path.join(root, file))
#     if len(plot_files) == 0:
#         continue
#     def extract_time(file_path):
#         data = pd.read_csv(file_path, sep='\t')
#         return data['Time'].values[0]
# 
#     # Sort the files based on the time
#     sorted_files = sorted(plot_files, key=extract_time)
#     print(sorted_files)
#     n = 0
#     fig,ax = plt.subplots(figsize=(6, 30),dpi = 200)
#     for file in sorted_files:
# 
#         data = pd.read_csv((file), sep='\t')
#     #fig,ax = plt.subplots(figsize=(6, 4),dpi = 100)
#     #data = pd.read_csv(r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/resultfiles/WD1929/Voigt_fitting/red_lines/corrected_for_pixel/2017-07-1123:29:21.148000/5978.93.txt", sep='\t')
#         time = data['Time'].values[0]
#         wavelength = data['Wavelength'].values
#         normalized_data = data['Normalized Data'].values
#         voigt_fit = data['Voigt fit'].values
# 
#         snr = data['SNR'].values[0]
#         depth = data['Depth'].values[0]
#         rv = data['RV'].values[0]
#         error = data['Error'].values[0]
# 
# 
# 
#         ax.plot(wavelength, normalized_data  + n/3 , color='black', linewidth=0.5)
#         ax.plot(wavelength, voigt_fit  + n/3, color='red')
#         ax.set_xlim(line-10,line+10)
#         ax.set_xlabel("Wavelength (Å)")
#         ax.set_ylabel("Normalized and offset flux")
#         x_text = wavelength[-1] + 0.2  # Add an offset for spacing
#         ax.text(line+6.5, (1.12 + n/3), f"SNR:{snr:.3g}" , verticalalignment='center', fontsize = 12)
#         ax.text(line+1.9, (1.12+ n/3), f"{time[:10]}" , verticalalignment='center', fontsize = 12)
# 
# 
#         n = n+1
#     plt.show()

# In[ ]:


#radial velocity calculations

times = []
rvs = []
errors = []
snrs = []
depths = []
lines = []

for i in p_lines:
    line = i[1]
    salt_root_dir = '/data/wdplanetary/omri/Output/resultfiles/WD1929/Voigt_fitting/red_lines/corrected_for_pixel/'  # Replace this with the path to your root directory
    # Initialize an empty list to store file paths
    plot_files = []
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(salt_root_dir):
        # Iterate over each file in the current directory
        for file in files:
            # Check if the file ends with '4481.185.txt'
            if file.endswith(f'{line}.txt'):
                # If it does, append the file path to the list
                plot_files.append(os.path.join(root, file))
    if len(plot_files) == 0:
        continue
    def extract_time(file_path):
        data = pd.read_csv(file_path, sep='\t')
        return data['Time'].values[0]

    # Sort the files based on the time
    sorted_files = sorted(plot_files, key=extract_time)
    print(sorted_files)
    n = 0
    for file in sorted_files:
        data = pd.read_csv((file), sep='\t')
        time = data['Time'].values[0]
        wavelength = data['Wavelength'].values
        normalized_data = data['Normalized Data'].values
        voigt_fit = data['Voigt fit'].values

        snr = data['SNR'].values[0]
        depth = data['Depth'].values[0]
        rv = data['RV'].values[0]
        error = data['Error'].values[0]

        rvs.append(rv)
        errors.append(error)
        times.append(time)
        snrs.append(snr)
        depths.append(depth)
        lines.append(line)



# In[ ]:


#for i,j,k,l,m,n in zip(times,rvs,errors,snrs,depths,lines):
#    print(i,j,k,l,m,n)
times_dt = []
for i in times:
    time_object = Time(i, format='iso', scale='utc')

    times_dt.append(time_object.datetime)


df = pd.DataFrame({
    'RVS': rvs,
    'ERRS': errors,
    'TIMES': times_dt,
    'SNRS': snrs,
    'LINES': lines
})

# Group by 'Timestamp' and aggregate values and errors
grouped = df.groupby('TIMES').agg({'RVS': 'sum', 'ERRS': lambda x: np.sqrt((x**2).sum())})

grouped = df.groupby('TIMES').apply(lambda x: np.average(x['RVS'], weights=1/x['ERRS']))
grouped_errors = df.groupby('TIMES').apply(lambda x: np.sqrt(np.sum(x['ERRS']**2)))

# Create a new DataFrame with combined RV and error values
result_df = pd.DataFrame({'TIMES': grouped.index, 'RVS': grouped.values, 'ERRS': grouped_errors.values})





condition = (result_df['RVS'] < mean - 2) | (result_df['RVS'] > mean + 2)  # Example condition: remove rows where Age is less than 30
result_df = result_df[~condition]  # Use ~ to negate the condition
result_df['ERRS'] = np.where(result_df['ERRS'] > 7, 7, result_df['ERRS'])

result_df["RVS"]= result_df["RVS"].abs()

mean = np.mean(result_df["RVS"])
stdev = np.std(result_df["RVS"])
print(mean,stdev)
result_df.to_csv('/data/wdplanetary/omri/Output/resultfiles/SALT_Voigt/reruns/mg4481_for_table.txt', sep='\t', index=False)
print(result_df)
#columns = ["Times","RVS","ERRORS","SNRS","DEPTHS","LINES"]
#all_data = pd.DataFrame([times,rvs,errors,snrs,depths,lines],columns = columns)
#print(all_data)

#mean = (np.sum((RV[~np.isnan(RV)] * RV_weight[~np.isnan(RV_weight)])) / (np.sum(RV_weight[~np.isnan(RV_weight)])))
#mean_error = np.sqrt(np.sum(RV_weight[~np.isnan(RV_weight)] * RV_err[~np.isnan(RV_err)]**2) / np.sum(RV_weight[~np.isnan(RV_weight)])+ v_precision**2) 


# In[ ]:





#frequencies = np.linspace(1.95,2.05,3000)
t = result_df["TIMES"]
t_days = np.array([(time - t[1]).total_seconds() / (24 * 3600) for time in t])

mean = np.mean(result_df["RVS"])
stdev = np.std(result_df["RVS"]) 
v = result_df["RVS"] - mean
errs = result_df["ERRS"]
errs[np.abs(errs) > 5] = 5
print(mean)

frequencies = np.linspace(0.001,5,30000)
powers = LombScargle(t_days, v).power(frequencies)

 
fig, (ax_t, ax_w) = plt.subplots(2, 1, facecolor="white", figsize=(10,12), constrained_layout=True, dpi = 400)

ax_t.errorbar(t_days, v, yerr=errs,fmt = '.k',capsize=5,markersize=10,lw = 1)
ax_t.fill_between(t_days, -stdev, stdev, color='gray',label = "1 sigma", alpha=0.4)
ax_t.fill_between(t_days, -3* stdev, 3* stdev, color='gray',label = "3 sigma", alpha=0.2)
ax_t.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax_t.legend()
#ax_t.text(np.max(t)/3, 2.5* stdev, f"RV mean = {mean:.3g} km/s")
ax_t.text(np.max(t_days)/3, 2* stdev, f'sigma = {stdev:.3g} km/s')
ax_t.set_xlabel("T - Days")
ax_t.set_ylabel("ΔRV - km/s")
""" 
#insrerting a planet
periodtime = np.linspace(np.min(t), np.max(t), 10000) 
ax_t.plot(periodtime, np.abs(v).max() * np.sin(2*np.pi* 2*periodtime))
ax_t.set_xlim(10,50)
 """
#plt.title("WD1929+012 Radial Velocity variations using Voigt fits - SALT data")
#plt.xticks(rotation=45)
#plt.tight_layout()
#plt.ylim(-20,20)
#plt.xlim(0,112)

normalized_powers = powers/(np.max(np.abs(powers)))
ax_w.plot(frequencies, normalized_powers )
ax_w.set_xlabel('Angular frequency [Periods/days]')
ax_w.set_ylabel('Normalized Power')
ax_w.tick_params(axis='x')
ax_w.tick_params(axis='y')

# Identify significant peaks (you can set your own threshold here)
threshold = 0.9  # Adjust as needed
significant_peak_indices = np.where(normalized_powers >= threshold)[0]
print(significant_peak_indices)
print(frequencies[1])
significant_peaks = np.array(frequencies[significant_peak_indices])

# Estimate false alarm rate for each peak
false_alarm_rates = []
for peak_freq in significant_peaks:
    # Use your preferred method to estimate false alarm rate here
    # Example: Monte Carlo simulations
    num_simulations = 100  # Adjust as needed
    peak_heights_simulated = []
    for _ in range(num_simulations):
        simulated_data = np.random.normal(0, 1, len(t_days))  # Generate random noise
        simulated_power = LombScargle(t, simulated_data, errs).power([peak_freq])
        peak_heights_simulated.append(simulated_power[0])
    false_alarm_rate = np.sum(peak_heights_simulated >= normalized_powers[np.where(frequencies == peak_freq)]) / num_simulations
    false_alarm_rates.append(false_alarm_rate)

# Print significant peaks and their corresponding false alarm rates
print("Significant Peaks:")
for i, freq in enumerate(significant_peaks):
    print(f"Frequency: {freq}, False Alarm Rate: {false_alarm_rates[i]}")
 
os.makedirs(r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/DeltaRV_files/SALT/Self_crosscorr/", exist_ok=True)
#plt.savefig(r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/DeltaRV_files/SALT/Self_crosscorr/firstrun.pdf")
plt.show()


# In[ ]:


#mike stacked plot
line = 3933.663


mike_root_dir = '/data/wdplanetary/omri/Output/resultfiles/WD1929/MIKE_Voigt_fitting/all_lines/'
# Initialize an empty list to store file paths
plot_files = []
# Walk through all directories and subdirectories
for root, dirs, files in os.walk(mike_root_dir):
    # Iterate over each file in the current directory
    for file in files:
        # Check if the file ends with '4481.185.txt'
        if file.endswith(f'{line}.txt'):
            # If it does, append the file path to the list
            plot_files.append(os.path.join(root, file))

""" def extract_time(file_path):
    data = pd.read_csv(file_path, sep='\t')
    return data['Time'].values[0]
# Sort the files based on the time
sorted_files = sorted(plot_files, key=extract_time)
 """

plot_files 
n = 0
fig,ax = plt.subplots(figsize=(6, 15),dpi = 200)
for file in plot_files[1:]:
    data = pd.read_csv((file), sep='\t')
    #print(data.to_string(index=False))
    time = data['Time'].values
    wavelength = data['Wavelength'].values
    normalized_data = data['Normalized Data'].values
    voigt_fit = data['Voigt fit'].values
    
    snr = data['SNR'].values[0]
    depth = data['Depth'].values[0]
    rv = data['RV'].values[0]
    error = data['Error'].values[0]
    ax.plot(wavelength, normalized_data  + n/2 , color='black', linewidth=0.5)
    ax.plot(wavelength, voigt_fit  + n/2, color='red')
    ax.set_xlim(line-10,line+10)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Normalized and offset flux")
    x_text = wavelength[-1] + 0.2  # Add an offset for spacing
    ax.text(line+6.5, (1.2 + n/2), f"SNR:{snr:.3g}" , verticalalignment='center', fontsize = 12)
    ax.text(line+1.9, (1.1+ n/2), f"{time[1]}" , verticalalignment='center', fontsize = 12)
    n = n+1
plt.show()

