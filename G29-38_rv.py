#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# code to do the radial velocity for the other white dwarfs

# mike folder for G29-38 /data/wdplanetary/laura/MIKE/Data/G29-38/blue/


# In[ ]:


# importing
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import pandas as pd
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from scipy.ndimage import uniform_filter1d
from scipy.stats import norm
from scipy.special import voigt_profile
from scipy import signal
from scipy.signal import correlate
from PyAstronomy import pyasl
from scipy.optimize import curve_fit
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.constants import c
import astropy.io.fits as fits
from astropy.units import dimensionless_unscaled
from lmfit.models import LinearModel, GaussianModel, VoigtModel, PolynomialModel
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import glob
import fnmatch
import os
from astropy.timeseries import LombScargle
from plotnine import *
import traceback
import sys
import linecache
import textwrap
import re
import emcee
import corner
import tqdm

""" 
plt.rcParams['axes.linewidth'] = 1.25
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1.25
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 1.25
 """
default_settings = {
    "font.size": 16,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3.5,
    "xtick.major.width": 1,
    "ytick.major.size": 3.5,
    "ytick.major.width": 1,
}


initial_settings = {
    "font.size": 22,
    "axes.linewidth": 1.25,
    "xtick.major.size": 5,
    "xtick.major.width": 1.25,
    "ytick.major.size": 5,
    "ytick.major.width": 1.25,
}
plt.rcParams.update(initial_settings)


# In[ ]:


# importing Mike data function


def read_mike_spec(filename):

    hdulist = fits.open(filename)
    obj_fits = hdulist[0]
    header = obj_fits.header
    date_obs = header.get("UT-DATE", None)
    time_obs = header.get("UT-START", None)

    OBJECT = header.get("OBJECT", None)

    sky_spec = obj_fits.data[0, :, :]
    obj_spec = obj_fits.data[1, :, :]
    noi_spec = obj_fits.data[2, :, :]
    snr_spec = obj_fits.data[3, :, :]
    lamp_spec = obj_fits.data[4, :, :]
    flat_spec = obj_fits.data[5, :, :]
    nobj_spec = obj_fits.data[6, :, :]
    hdulist.close()

    # Ca H and K are found in orders 18-22 which are indexed 17-21. Over these
    # 4 order, the extracted Ca H and K can be combined to further reduce noise

    i = 0
    found = -1

    while (found == -1) and (i < len(header)):
        found = str(header[i]).find("spec1")
        i += 1

    all_order_wav = str(header[(i - 1) : :])
    all_order_wav = "'" + all_order_wav

    # The following loop can be used to remove all the header keys from the string
    i = 1
    match = True
    while match:
        string = "'WAT2_%03d= '" % i
        wat = re.compile(string)
        match = wat.search(all_order_wav)
        all_order_wav = wat.sub("", all_order_wav)
        i += 1

    wavelength_soln_data = []

    # This loop parses the wavelength solns by the spec keyword and stores the entire
    # solution as a list of lists, each entry containing one place for each solved
    # parameter.
    i = 1
    match = True
    while match:
        string = "spec%d" % i
        spec_key = re.compile(string)
        match = spec_key.search(all_order_wav)
        if match:
            wavelength_soln_data.append(
                (all_order_wav[(match.start() + 10) : match.start() + 91]).split()
            )
        i += 1

    # This loop generates an array of arrays with each array containing the wavelength
    # soln.
    wav_solns = []
    for i in range(len(wavelength_soln_data)):
        wav_start = float(wavelength_soln_data[i][3])
        wav_delta = float(wavelength_soln_data[i][4])
        wav_end = wav_start + wav_delta * float(wavelength_soln_data[i][5])
        wav_solns.append(np.arange(wav_start, wav_end, wav_delta))
        i += 1
    #
    #    plt.figure(1)
    #    plt.clf()
    #    for i in range(len(wav_solns)):
    #        plt.plot(wav_solns[i],nobj_spec[i,:])

    all_data = np.zeros([len(wav_solns), 9], dtype=object)
    for i in range(len(wav_solns)):
        all_data[i, 0] = i + 1
        all_data[i, 1] = wav_solns[i]
        all_data[i, 2] = sky_spec[i, :]
        all_data[i, 3] = obj_spec[i, :]
        all_data[i, 4] = noi_spec[i, :]
        all_data[i, 5] = snr_spec[i, :]
        all_data[i, 6] = lamp_spec[i, :]
        all_data[i, 7] = flat_spec[i, :]
        all_data[i, 8] = nobj_spec[i, :]
    keywords = [
        "Order",
        "Wavelength Soln",
        "Sky Spectrum",
        "Object Spectrum",
        "Noise Spectrum",
        "SNR Spectrum",
        "Lamp Spectrum",
        "Flat Spectrum",
        "Normalized Object Spectrum",
    ]

    # First, calculating the heliocentric correction

    site_lat = header.get("SITELAT", None)
    site_long = header.get("SITELONG", None)
    site_alt = header.get("SITEALT", None)
    ra = header.get("RA-D", None)  # in degrees
    dec = header.get("DEC-D", None)  # in degrees
    epoch = str(header.get("EPOCH", None))  # e.g., 'J2000'
    time = Time(str(date_obs) + "T" + str(time_obs), format="fits", scale="utc")

    # print(time)
    # Create EarthLocation object
    location = EarthLocation.from_geodetic(
        lat=site_lat * u.deg, lon=site_long * u.deg, height=site_alt * u.m
    )
    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    heliocorr = sc.radial_velocity_correction(
        "heliocentric", obstime=time, location=location
    )
    # print(heliocorr.to(u.km/u.s))
    heliocorr = heliocorr.to(u.m / u.s) / c

    # Now making easy arrays to use:
    wav = []
    order = []
    flux = []
    snr = []
    for i in range(len(all_data)):
        wav.append(all_data[i, 1])
        order.append(all_data[i, 0])
        flux.append(all_data[i, 8])
        snr.append(all_data[i, 5])

    # Now apply the heliocentric correction to the wavelength data:
    for i in range(len(wav)):
        for jk in range(len(wav[i])):
            wav[i][jk] *= 1 + heliocorr

        # heliocentric correction
    #   wavelength[i] = (CRVAL1 + CDELT1*i) + (HEL_COR/299792 )*(CRVAL1 + CDELT1*i)
    # wavelength[i] = (CRVAL1 + CDELT1*i)
    print(OBJECT)
    return order, wav, flux, snr, OBJECT, time


# filename = "/data/wdplanetary/laura/MIKE/Data/WD1929+011/blue/galex1931_blue_2011-06-09.fits"
""" order,wav,flux,snr,OBJECT,time = read_mike_spec(filename)
long_string = str(header)
wrapped_text = textwrap.wrap(long_string, width=80)
for line in wrapped_text:
    print(line)
plt.figure()
plt.plot(wav[8],flux[8])
plt.xlim(4475,4485)
plt.show()

plt.figure()
plt.plot(wav[8],snr[8])
plt.xlim(4475,4485)
plt.show() """


# In[ ]:


# line dictionaries: Name, position(A), present?, log gf
p_lines = []
spec_lines = []

# pollutant lines
p_lines.append(("CaII_3933", 3933.663, True))  # only use for MIKE data
p_lines.append(("CaII_4226", 4226.727, False))  # good
p_lines.append(
    ("FeII_4923", 4923.927, False)
)  # can't use, crosses over with interorder
p_lines.append(("FeII_5018", 5018.440, False))  # not there
p_lines.append(
    ("FeII_5169", 5169.033, False)
)  # this one gives an error when trying to fit it
p_lines.append(("SiII_5041", 5041.024, False))  # don't use this
p_lines.append(("SiII_5055", 5055.984, False))  # This one is good use it
p_lines.append(("SiII_5957", 5957.560, False))
p_lines.append(("SiII_5978", 5978.930, False))
p_lines.append(
    ("SiII_6347", 6347.100, False)
)  # these two are quite strong in WD1929+012
p_lines.append(("SiII_6371", 6371.360, False))  #
p_lines.append(("MgI_5172", 5172.683, False))
p_lines.append(("MgI_5183", 5183.602, False))
p_lines.append(("MgII_4481", 4481.130, False))  # strong magnesium line
p_lines.append(("MgII_4481_2", 4481.327, False))
p_lines.append(("MgII_4481", 4481.180, False))  # weighted combination of mg_4481 lines

p_lines.append(("MgII_7877", 7877.054, False))
p_lines.append(("MgII_7896", 7896.366, False))  # definitely not present
p_lines.append(("OI_7771", 7771.944, False))  # definitely not present
# hydrogen lines
p_lines.append(("H_4860", 4860.680, False))
p_lines.append(("H_4860_2", 4860.968, False))  # This one gives better values
p_lines.append(("H_4860_2", 4860.968, False))  # Weighted combo
p_lines.append(("H_4340", 4340.472, False))  # present
p_lines.append(("H_6563", 6562.79, False))
# pick the spectral lines present in this white dwarf

for i in p_lines:
    if i[2] == True:
        spec_lines.append(i)

b_lines = []
r_lines = []
for i in spec_lines:
    if i[1] <= 5550:
        # 370 - 555 nm
        b_lines.append(i)
    else:
        # 555 - 890 nm
        r_lines.append(i)

# Now define the sky lines that we will use to find the stability corrections from the instrument variability

sky_lines = []
sky_lines.append(("OI_5577", 5577.340))
sky_lines.append(("OI_6300", 6300.304))
sky_lines.append(("OI_6364", 6363.776))
# sky_lines.append(("H2O_7392"))
# sky_lines.append(("H2O_8365"))


# In[ ]:


def process_data_mike_gaussian(wav, flux, c_lines):
    # make a new wavelength filter that is accurate to 3dp
    min = wav[np.argmin(wav)]
    max = wav[np.argmax(wav)]
    xwav = np.arange(min, max, 0.001)

    minlen = np.minimum(len(wav), len(flux))

    # Adjust the size of the flux data to fit the mask
    interp_func = interp1d(
        wav[:minlen], flux[:minlen], bounds_error=False, fill_value=np.nan
    )
    padded_flux = interp_func(xwav)

    # movingavg
    window_size = 3000
    n_moving_avg = uniform_filter1d(padded_flux, size=window_size)
    return xwav, padded_flux, n_moving_avg


# In[ ]:


def calculate_error(data):
    window_size = int(500)
    # Use a rolling window approach for efficient calculation of standard deviation
    # least mean squared method
    rolling_std = np.sqrt(
        np.convolve(data**2, np.ones(window_size) / window_size, mode="valid")
        - np.convolve(data, np.ones(window_size) / window_size, mode="valid") ** 2
    )

    # Extend the result to match the original length of the data
    pad_width = window_size // 2
    errors = np.pad(rolling_std, (pad_width, pad_width), mode="edge")
    # print(len(errors),len(data))
    errorsv = np.nan_to_num(errors)
    # print(np.mean(errorsv))
    return errorsv[: len(data)]


# In[ ]:


def poly_fit(gwav, gdata):
    poly_model = PolynomialModel(degree=7)
    params = poly_model.guess(gdata, x=gwav)
    poly_result = poly_model.fit(gdata, params, x=gwav)
    print(
        poly_result.params["c0"],
        poly_result.params["c1"],
        poly_result.params["c2"],
        poly_result.params["c3"],
        poly_result.params["c4"],
    )

    # find the individual polynomial best fit
    c0 = poly_result.params["c0"].value
    c1 = poly_result.params["c1"].value
    c2 = poly_result.params["c2"].value
    c3 = poly_result.params["c3"].value
    c4 = poly_result.params["c4"].value
    c5 = poly_result.params["c5"].value
    c6 = poly_result.params["c6"].value
    c7 = poly_result.params["c7"].value

    p_result = (
        c0
        + c1 * gwav
        + c2 * gwav**2
        + c3 * gwav**3
        + c4 * gwav**4
        + c5 * gwav**5
        + c6 * gwav**6
        + c7 * gwav**7
    )

    plt.figure()
    plt.plot(gwav, p_result, color="green", linewidth=1)
    plt.plot(gwav, gdata, color="blue", linewidth=0.5)

    plt.show()
    return p_result


# In[ ]:


# function to pick the files that we want


def pick_files_by_patterns(folder_path, start_patterns, end_patterns):
    """
    Pick files in a folder based on given start and end patterns in their filename.

    Parameters:
    - folder_path (str): Path to the folder containing files.
    - start_patterns (list): List of patterns to match at the start of filenames.
    - end_patterns (list): List of patterns to match at the end of filenames.

    Returns:
    - List of filenames matching the specified start and end patterns.
    """
    matching_files = []

    # Ensure the folder path is valid
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return matching_files

    for folder_name in os.listdir(folder_path):
        fp = os.path.join(folder_path, folder_name, "product/")

        # List all files in the folder
        try:
            all_files = os.listdir(fp)
        except FileNotFoundError:
            continue
        except NotADirectoryError:
            continue

        # Filter files based on start patterns
        # for start_pattern in start_patterns:
        #   matching_files.extend(fnmatch.filter(all_files, start_pattern + '*'))

        # Filter files based on end patterns
        # for end_pattern in end_patterns:
        #   matching_files.extend(fnmatch.filter(all_files, '*' + end_pattern))
        for file in all_files:
            if file.startswith(start_patterns) and file.endswith(end_patterns):
                matching_files.append(os.path.join(fp, file))

    return matching_files


def mike_pick_files_by_patterns(folder_path, start_patterns, end_patterns):
    matching_files = []
    # Ensure the folder path is valid
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return matching_files
    try:
        for file_name in os.listdir(folder_path):
            matching_files.append(os.path.join(folder_path, file_name))
    except FileNotFoundError:
        # Handle the case where the folder doesn't exist
        print(f"The folder '{folder_path}' does not exist.")
        return None

    # Return the list of matching files
    return matching_files


# Example usage:

# This is where we pick the folder path and the name of the file that we want
# Need to adjust this so that it is more clear
star = "G29-38"
MIKE_blue_folder_path = "/data/wdplanetary/laura/MIKE/Data/G29-38/blue/"
MIKE_red_folder_path = "/data/wdplanetary/laura/MIKE/Data/G29-38/red/"

mike_start = "gal"
mike_end = ".fits"

# Creating file directory for the blue and red channels separately

mike_b_files = mike_pick_files_by_patterns(MIKE_blue_folder_path, mike_start, mike_end)
mike_r_files = mike_pick_files_by_patterns(MIKE_red_folder_path, mike_start, mike_end)

mike_b_files.sort()
mike_r_files.sort()

print(mike_b_files)


# In[ ]:


# Voigt fitting for the MIKE Spectrum


def Mike_Gaussian(
    wav_list,
    flux_list,
    moving_avg_list,
    errors_list,
    snr_list,
    c_lines,
    time,
    n,
    plot_dir,
    OBJECT,
):

    v_precision = 0.0  # kneeed to find the stability for MIKE
    w_size = 30
    snr_cutoff = 13
    line_depth_cutoff = 3
    # initialise result arrays
    RV = np.array([])
    RV_weight = np.array([])
    RV_err = np.array([])

    # initialise the arrays for plotting
    D = np.array([])
    SNR = np.array([])

    for i in c_lines:
        print(i)
        try:
            for j, wav_value in enumerate(wav_list):
                """
                plt.figure()
                plt.plot(wav_value,flux_list[j])
                plt.show()
                """
                if float(np.min(wav_value)) <= float(i[1]) <= float(np.max(wav_value)):
                    print(f" For order {j} the line is in the order")
                    xwav = wav_value
                    xflux = flux_list[j]
                    xmoving_avg = moving_avg_list[j]
                    xerrors = errors_list[j]
                    # xsnr = snr_list[j]

                    # change plot directory
                    wd = os.path.join(plot_dir, str(i[0]) + "/")
                    # Create the directory if it doesn't exist
                    if not os.path.exists(wd):
                        os.makedirs(wd)

                    # gwav,gdata,gerrors,gweights,D,snr = make_window(wav,flux,moving_avg,i,w_size):

                    # find the upper bound of the window
                    u_loc = np.searchsorted(xwav, (i[1] + (w_size / 2)))
                    closest_value = xwav[max(0, u_loc - 1)]
                    u_bound = np.where(xwav == closest_value)
                    u_bound = int(u_bound[0])
                    # find the lower bound of the window
                    l_loc = np.searchsorted(xwav, (i[1] - (w_size / 2)))
                    closest_value = xwav[max(0, l_loc - 1)]
                    l_bound = np.where(xwav == closest_value)
                    l_bound = int(l_bound[0])
                    # make small datasets around the line
                    # if l_bound >= 0 and u_bound <= len(flux):
                    # Slice the array using integer indices

                    gdata = xflux[l_bound:u_bound]
                    gwav = xwav[l_bound:u_bound]
                    gavg = xmoving_avg[l_bound:u_bound]
                    gavg = np.where(gavg != 0, gavg, 1)
                    # gsnr = xsnr[l_bound:u_bound]
                    """ else:
                        raise Exception("The window falls outside the limit of the order") """
                    # make a new window for errors which is just outside the window - basically the next window down instead
                    # Instead we want to make a small window around the absorption line
                    # and then take the errors from the points around it
                    # and then take the average value of that
                    # and set the errors of the line to it
                    el_loc = np.searchsorted(xwav, (i[1] - (1 / 4) * (w_size)))
                    closest_value = xwav[max(0, el_loc - 1)]
                    el_bound = np.where(xwav == closest_value)
                    el_bound = int(el_bound[0])
                    eh_loc = np.searchsorted(xwav, (i[1] + (1 / 4) * (w_size)))
                    closest_value = xwav[max(0, eh_loc - 1)]
                    eh_bound = np.where(xwav == closest_value)
                    eh_bound = int(eh_bound[0])

                    firsterrorbox = xerrors[l_bound:el_bound]
                    seconderrorbox = xerrors[eh_bound:u_bound]
                    err_avg = (np.mean(firsterrorbox) + np.mean(seconderrorbox)) / 2
                    lineerroravg = [] * (eh_bound - el_bound)
                    lineerroravg[el_bound:eh_bound] = [err_avg] * (eh_bound - el_bound)

                    gerrors = np.concatenate(
                        (firsterrorbox, lineerroravg, seconderrorbox)
                    )
                    # gerrors[np.isnan(gerrors)] = 0.001
                    gerrors[: len(gdata)]

                    gweights = np.where(gerrors != 0, 1 / gerrors, 0.001)

                    # Calculate the mean signal
                    signal_mean = (
                        (
                            np.mean(xflux[l_bound:el_bound])
                            + np.mean(xflux[eh_bound:u_bound])
                        )
                        / 2
                    ) - np.min(xflux[el_bound:eh_bound])
                    # Calculate the signal-to-noise ratio (SNR)
                    depth = np.abs(signal_mean / err_avg)
                    D = np.append(D, depth)

                    # now calculate snr
                    snr = np.median(xflux[eh_bound:u_bound]) / (
                        np.std(xflux[eh_bound:u_bound])
                    )

                    # instead do this with the snr dataset - take mean of snr in bounds
                    # snr = np.mean(gsnr)

                    SNR = np.append(SNR, snr)

                    # set the initial guess of the mean value of the gaussian to the wavelength in air
                    line = i[1]

                    # p_result = poly_fit(xwav,gdata)
                    poly_model = PolynomialModel(degree=5)
                    params = poly_model.guess(gdata, x=gwav)
                    poly_result = poly_model.fit(
                        gdata, params, x=gwav, weights=gweights
                    )
                    print(
                        poly_result.params["c0"],
                        poly_result.params["c1"],
                        poly_result.params["c2"],
                        poly_result.params["c3"],
                        poly_result.params["c4"],
                    )

                    # find the individual polynomial best fit
                    c0 = poly_result.params["c0"].value
                    c1 = poly_result.params["c1"].value
                    c2 = poly_result.params["c2"].value
                    c3 = poly_result.params["c3"].value
                    c4 = poly_result.params["c4"].value
                    c5 = poly_result.params["c5"].value
                    p_result = (
                        c0
                        + c1 * gwav
                        + c2 * gwav**2
                        + c3 * gwav**3
                        + c4 * gwav**4
                        + c5 * gwav**5
                    )

                    """     plt.figure()
                    plt.plot(gwav, p_result, color='green', linewidth=1)
                    plt.plot(gwav, gdata, color='blue', linewidth=0.5)
                    
                    #plt.show()
                    plt.close()
                    """

                    n_flux = gdata / p_result
                    n_errors = gerrors / p_result
                    n_weights = np.where(n_errors != 0, 1 / n_errors, 0.001)
                    # CHANGE THE SNR VALUE TO FILTER DATASETS THAT ARE NOT FITTING WELL
                    # We want this code to try the next line in the dataset
                    if snr <= snr_cutoff:
                        raise Exception(f"SNR too low: SNR {snr} <{snr_cutoff}")
                    if depth <= line_depth_cutoff:
                        raise Exception(
                            f"Line not significant: Line depth {depth} < {line_depth_cutoff}"
                        )
                    ####

                    print(f"SNR is {snr:.3g}")

                    voigt_model = VoigtModel(prefix="voigt_")
                    linear_model = LinearModel(prefix="linear_")
                    composite_model = voigt_model + linear_model
                    # When these parameters fit right for the Ca 3xxx fit the amp is -200, the sigma and gamma are 0.03
                    params = voigt_model.make_params(
                        voigt_amplitude=-0.6,
                        voigt_center=(line + 0.6),
                        voigt_sigma=0.03,
                        voight_gamma=0.03,
                    )
                    params += linear_model.make_params(
                        slope=0, intercept=np.median(n_flux)
                    )
                    params["voigt_amplitude"].min = -1
                    params["voigt_amplitude"].max = -0.02
                    params["voigt_center"].min = line - 2
                    params["voigt_center"].max = line + 2
                    # params['voigt_sigma'].max = 0.05
                    print("absorption params made")

                    result = composite_model.fit(
                        n_flux, params, x=gwav, weights=n_weights
                    )  # ,sigma = wav_errors)
                    print(
                        result.values["voigt_center"],
                        result.params["voigt_center"],
                        result.params["voigt_amplitude"],
                    )
                    # print("The results of the fitting report are: ")
                    # print(result.fit_report())

                    t = time.datetime
                    # plt.show()
                    # print(result.values['voigt_center'])
                    # print(result.fit_report())
                    Line_Offset = result.values["voigt_center"] - line
                    rv = (Line_Offset / line) * 299792
                    # print(f"The voigt sigma is {(result.params['voigt_sigma'])} and the center error is{ result.params['voigt_center'].stderr}")
                    if result.params["voigt_center"].stderr is not None:
                        err = (
                            (result.params["voigt_center"].stderr * 3)
                            / result.values["voigt_center"]
                        ) * 299792

                    else:
                        err = 0.2
                    # err = (((result.params['voigt_sigma'].value)/result.values['voigt_center']) * 299792)

                    # Append values to RV
                    # if 35 < rv < 40:
                    RV = np.append(RV, rv)
                    RV_err = np.append(RV_err, (err))  # +v_precision
                    RV_weight = np.append(RV_weight, (1 / (err)))
                    # else:
                    #    raise Exception("The rv shift is not in the bounds")

                    rv_significant_figures = max(3, -int(np.floor(np.log10(err))) + 2)

                    # Format the radial velocity and its error using the determined significant figures
                    rv_formatted = f"{rv:.{4}g}"
                    err_formatted = f"{err:.{2}g}"

                    # Plotting results
                    if c_lines is not sky_lines:

                        # Add a simple plot to the stacked plot
                        """
                        ax.plot(gwav, gdata/p_result  + n/3 , color='black', linewidth=0.5)
                        ax.plot(gwav, result.best_fit/p_result  + n/3, color='red')
                        ax.set_xlim(line-10,line+10)
                        x_text = gwav[-1] + 0.2  # Add an offset for spacing
                        ax.text(line+11, (1 + n/3), f"SNR:{snr:.3g}" , verticalalignment='center')
                        n = n+1
                        """
                        print("stacked plot made")

                        # We can add a label next to the line by using

                        # Now save the line plot with the best fit and the shaded errors
                        plt.figure(facecolor="white", figsize=(10, 8))
                        # add spectrum
                        plt.plot(gwav, n_flux, color="black", linewidth=0.5)
                        # add best fit line
                        plt.plot(gwav, result.best_fit, color="red")
                        # shade standard deviation of data
                        plt.fill_between(
                            gwav,
                            (result.best_fit - n_errors),
                            (result.best_fit + n_errors),
                            color="gray",
                            alpha=0.3,
                        )

                        plt.title(
                            f"{t.year}/{t.month}/{t.day}, Order: {j}, Object: {OBJECT}"
                        )
                        plt.text(
                            0.95,
                            0.10,
                            f"SNR = {snr:.3g}",
                            horizontalalignment="right",
                            verticalalignment="top",
                            transform=plt.gca().transAxes,
                        )
                        plt.text(
                            0.95,
                            0.05,
                            f" RV = {rv:.{4}g} ± {err:.{2}g} km/s",
                            horizontalalignment="right",
                            verticalalignment="top",
                            transform=plt.gca().transAxes,
                        )
                        plt.ylim(0.5, 1.3)
                        plt.xlim(line - 6, line + 6)
                        plt.xlabel("Wavelength (Å)")
                        plt.ylabel("Normalized Flux")
                        title = str(i[0]) + "_" + str(t) + ".pdf"
                        title_without_spaces = title.replace(" ", "")
                        plt.savefig(os.path.join(wd, title_without_spaces))
                        plt.show()
                        plt.close()

                        # save the datafiles instead of stacking the plot here
                        data = pd.DataFrame(
                            {
                                "Wavelength": (gwav),
                                "Normalized Data": (n_flux),
                                "Voigt fit": (result.best_fit),
                                "Time": t,
                                "SNR": snr,
                                "Depth": depth,
                                "RV": rv,
                                "Error": err,
                                "Order": j,
                            }
                        )  # [t],[snr], [depth], [rv], [err]])
                        dir_name = fr"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/resultfiles/G29-38/MIKE_Voigt_fitting/all_lines/{t}/"
                        dir_name_without_spaces = dir_name.replace(" ", "")
                        os.makedirs(dir_name_without_spaces, exist_ok=True)
                        file_end = f"order{j}_{str(line)}.txt"
                        file_name = os.path.join(dir_name_without_spaces, file_end)
                        file_name_without_spaces = file_name.replace(" ", "")
                        # np.savetxt(file_name_without_spaces, data, delimiter=',', fmt = '%f') #"fmt=['%f', '%f', '%f', '%s', '%f', '%f', '%f', '%f'])
                        data.to_csv(file_name_without_spaces, sep="\t", index=False)

                    else:
                        raise Exception("Line not in this order")

        except Exception as e:
            print(f"Error occurred: {e}")

            RV = np.append(RV, np.nan)
            RV_err = np.append(RV_err, np.nan)
            RV_weight = np.append(RV_weight, np.nan)
            continue

    # weighted mean and standard deviation error calculations
    if np.any(~np.isnan(RV)):
        # Weighted mean and standard deviation error calculations
        mean = np.sum((RV[~np.isnan(RV)] * RV_weight[~np.isnan(RV_weight)])) / (
            np.sum(RV_weight[~np.isnan(RV_weight)])
        )
        mean_error = np.sqrt(
            np.sum(RV_weight[~np.isnan(RV_weight)] * RV_err[~np.isnan(RV_err)] ** 2)
            / np.sum(RV_weight[~np.isnan(RV_weight)])
            + v_precision**2
        )

    else:
        mean = np.nan
        mean_error = np.nan

    print("The rv mean and error is", mean, mean_error)

    return mean, mean_error, n


# In[ ]:


# Voigt fitting runfile for MIKE
rvs = []
time_strings = []
rverrs = []

n = 0
star = ""

mike_blue_plot_dir = r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/G29-38/MIKE/Voigtfits"
# fig, ax = plt.subplots(figsize=(6, 20))


for j in mike_b_files:
    order, wav, flux, snr, OBJECT, time = read_mike_spec(j)
    print("------------NEW--FILE----------", str(time))
    xwav_list = np.empty(len(order), dtype=object)
    padded_flux_list = np.empty(len(order), dtype=object)
    moving_avg_list = np.empty(len(order), dtype=object)
    errors_list = np.empty(len(order), dtype=object)
    snr_list = np.empty(len(order), dtype=object)

    print("imported files")
    # if OBJECT.startswith(star):
    # then call data processing
    for o in order:
        xwav, padded_flux, moving_avg = process_data_mike_gaussian(
            wav[o - 1], flux[o - 1], b_lines
        )
        errors = calculate_error(padded_flux)

        xwav_list[o - 1] = xwav
        padded_flux_list[o - 1] = padded_flux
        moving_avg_list[o - 1] = moving_avg
        errors_list[o - 1] = errors
        snr_list[o - 1] = snr[o - 1]

    # print(xwav_list)
    # print(snr)

    print("Smoothed flux and calculated errors")
    # print(errors,padded_flux,moving_avg)
    # then do Gaussian
    rv, rv_err, n = Mike_Gaussian(
        xwav_list,
        padded_flux_list,
        moving_avg_list,
        errors_list,
        snr_list,
        b_lines,
        time,
        n,
        mike_blue_plot_dir,
        OBJECT,
    )
    # now need to add to directories of wavelengths, flux_results, best_fits,snrs, line depths

    # Then find the atmospheric correction- don't need to do this as we have the stabilities
    # rv_corr,corr_err,ax = Gaussian(xskywav,sky_padded_flux,sky_moving_avg,sky_errors,sky_lines,skytime,n,ax)
    if np.isnan(rv):  # or np.isnan(rv_corr):
        print("Fit could not be calculated")
        continue

    else:
        # rv_corrs.append((rv_corr))
        rvs.append((rv))  # -rv_corr
        rverrs.append((rv_err))
        time_strings.append((time))
        print("The fit has been calculated successfully")
    # else:
    #    print("This is a file for " + str(OBJECT) + " instead of " + str(star))
    #    continue

print(np.nanmean(rvs))


# In[ ]:


# Combining data points that are the same times
t_list = Time(time_strings, scale="utc")
times = t_list.datetime
t0 = times[0]
tdays = [(dt - t0).days for dt in times]
mean = np.nanmean(rvs)
print(mean)


tarray = np.array(tdays)
rvarray = np.array(rvs)
rverrsarray = np.array(rverrs)
# rvcorrsarray = np.array(rv_corrs)
# rverrsarray = np.full(len(tarray),2)

filtered_indices = np.where((rverrsarray < 20) & (rvarray > 30) & (rvarray < 45))[
    0
]  # & (rvarray > 10)
filtered_rvarray = rvarray[filtered_indices]
filtered_tarray = tarray[filtered_indices]
filtered_rverrsarray = rverrsarray[filtered_indices]
# filtered_rvcorrsarray = rvcorrsarray[filtered_indices]

filtered_mean = np.nanmean(filtered_rvarray)
print(filtered_mean)
unique_times = np.unique(filtered_tarray)
averages_unfiltered = np.array([tarray, rvarray, rverrsarray])

averages = np.array(
    [
        [
            t,
            np.nanmean(filtered_rvarray[filtered_tarray == t]),
            np.nanmean(filtered_rverrsarray[filtered_tarray == t]),
        ]
        for t in unique_times
    ]
)
# averages = averages[np.isnan(averages[:,1])]

print(averages)  #

np.savetxt(
    r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/resultfiles/G29-38/MIKE_Voigt_Results/ca3933.txt",
    averages,
)


# In[ ]:


file_name = (
    r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/resultfiles/G29-38/MIKE_Voigt_Results/ca3933.txt"
)
times = []
delta_rvs = []
errors = []
frequencies = []
powers = []

data = np.loadtxt(file_name, delimiter=" ")
print(data)


t = data[:, 0]  # First column
mean = np.nanmean(data[:, 1])
v = data[:, 1] - mean  # Second column
errs = data[:, 2]  # Third column
variance = np.var(v)  # + np.mean(errs)**2
stdev = np.sqrt(variance)
# frequencies = np.linspace(1.95,2.05,3000)
frequencies = np.linspace(0.001, 1, 1000)
powers = LombScargle(t, v, errs).power(frequencies)


fig, (ax_t, ax_w) = plt.subplots(
    2, 1, facecolor="white", figsize=(12, 14), constrained_layout=True, dpi=200
)

ax_t.errorbar(t, v, yerr=errs, fmt=".k", capsize=5, lw=1.5)
ax_t.fill_between(t, -stdev, stdev, color="gray", label="1 sigma", alpha=0.4)
ax_t.fill_between(t, -3 * stdev, 3 * stdev, color="gray", label="3 sigma", alpha=0.2)
ax_t.axhline(y=0, color="black", linestyle="--", linewidth=1)
ax_t.legend()
ax_t.text(np.max(t) / 3, 2.5 * stdev, f"RV mean = {mean:.3g} km/s")
ax_t.text(np.max(t) / 3, 2 * stdev, f"sigma = {stdev:.3g} km/s")
ax_t.set_xlabel("T - Days")
ax_t.set_ylabel("ΔRV - km/s")
""" 
#insrerting a planet
periodtime = np.linspace(np.min(t), np.max(t), 10000) 
ax_t.plot(periodtime, np.abs(v).max() * np.sin(2*np.pi* 2*periodtime))
ax_t.set_xlim(10,50)
 """
# plt.title("WD1929+012 Radial Velocity variations using Voigt fits - SALT data")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.ylim(-20,20)
# plt.xlim(0,112)

normalized_powers = powers / (np.max(np.abs(powers)))
ax_w.plot(frequencies, normalized_powers)
ax_w.set_xlabel("Angular frequency [Periods/days]")
ax_w.set_ylabel("Normalized Power")
ax_w.tick_params(axis="x")
ax_w.tick_params(axis="y")


# Identify significant peaks (you can set your own threshold here)
threshold = 0.6  # Adjust as needed
significant_peak_indices = np.where(normalized_powers >= threshold)[0]
print(significant_peak_indices)
print(frequencies[1])
significant_peaks = np.array(frequencies[significant_peak_indices])

# Estimate false alarm rate for each peak
false_alarm_rates = []
for peak_freq in significant_peaks:
    # Use your preferred method to estimate false alarm rate here
    # Example: Monte Carlo simulations
    num_simulations = 500  # Adjust as needed
    peak_heights_simulated = []
    for _ in range(num_simulations):
        simulated_data = np.random.normal(0, 1, len(t))  # Generate random noise
        simulated_power = LombScargle(t, simulated_data, errs).power([peak_freq])
        peak_heights_simulated.append(simulated_power[0])
    false_alarm_rate = (
        np.sum(
            peak_heights_simulated
            >= normalized_powers[np.where(frequencies == peak_freq)]
        )
        / num_simulations
    )
    false_alarm_rates.append(false_alarm_rate)

# Print significant peaks and their corresponding false alarm rates
print("Significant Peaks:")
for i, freq in enumerate(significant_peaks):
    print(f"Frequency: {freq}, False Alarm Rate: {false_alarm_rates[i]}")

os.makedirs(
    r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/DeltaRV_files/SALT/Self_crosscorr/", exist_ok=True
)
# plt.savefig(r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/DeltaRV_files/SALT/Self_crosscorr/firstrun.pdf")
plt.show()


# In[ ]:


rvshift = (0.001 / 3933) * 299792000
print(f"rv = {rvshift} (m/s)")


# In[ ]:
