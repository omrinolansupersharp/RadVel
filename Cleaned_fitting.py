salt_direct = False
mike_direct = False
salt_selfcc = False
mike_selfcc = True
salt_modelcc = False #ready
mike_modelcc = False




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
from pathlib import Path



# salt read in data
def Get_Wavelength_Flux_File(filename):
    # Get the wavelength and flux from a MIDAS pipeline file
    hdulist = fits.open(filename)
    header = hdulist[0].header
    date_obs = header.get("DATE-OBS", None)
    time_obs = header.get("TIME-OBS", None)
    time = Time(str(date_obs) + "T" + str(time_obs), format="fits")
    # time = Time(date_obs,format = "fits")

    flux = hdulist[0].data  # flux in counts

    CRVAL1 = hdulist[0].header["CRVAL1"]  # Coordinate at reference pixel
    CRPIX1 = hdulist[0].header["CRPIX1"]  # Reference pixel
    CDELT1 = hdulist[0].header["CDELT1"]  # Coordinate increase per pixel
    HEL_COR = hdulist[0].header["HEL_COR"]  # Heliocentric correction
    OBJECT = str(header.get("OBJECT", None))

    # Need to use the info above to make the wavelength axis
    wavelength = np.zeros(len(flux))
    #print(f"The reference pixel in timefile: {time} is {CRPIX1}")
    for i in range(len(wavelength)):
        # heliocentric correction
        wavelength[i] = (
            (CRVAL1 + CDELT1 * i)
            + (HEL_COR / 299792) * (CRVAL1 + CDELT1 * i)
            - (CRPIX1 * CDELT1)
        )
        # wavelength[i] = (CRVAL1 + CDELT1*i)

    hdulist.close()

    return wavelength, flux, time, OBJECT

#mike read in file
def read_mike_spec(filename):
    # importing Mike data function
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

    return order, wav, flux, snr, OBJECT, time


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

            
def sanitize_component(value) -> str:
    s = str(value)
    s = s.replace('T', '_').replace(':', '-').replace(' ', '_')
    return s.rstrip(' .')
    
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


def process_data_gaussian(wav, flux):

    # make a new wavelength filter that is accurate to 3dp
    min = wav[np.argmin(wav)]
    max = wav[np.argmax(wav)]
    xwav = np.arange(min, max, 0.001)

    # Adjust the size of the flux data to fit the mask
    interp_func = interp1d(wav, flux, bounds_error=False, fill_value=np.nan)
    padded_flux = interp_func(xwav)

    # movingavg
    window_size = 3000
    n_moving_avg = uniform_filter1d(padded_flux, size=window_size)

    return xwav, padded_flux, n_moving_avg


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


def process_data_cc(wav, flux):
    s_flux = savgol_filter(flux, window_length=501, polyorder=3)
    flux = flux / s_flux
    return wav, flux



def process_data_model(wav, flux, c_lines):
    u_loc = np.searchsorted(wav, (5200))
    closest_value = wav[max(0, u_loc - 1)]
    u_bound = np.where(wav == closest_value)
    u_bound = int(u_bound[0].item())
    # find the lower bound of the window
    l_loc = np.searchsorted(wav, (4100))
    closest_value = wav[max(0, l_loc - 1)]
    l_bound = np.where(wav == closest_value)
    l_bound = int(l_bound[0].item())
    # make small datasets around the window
    flux = flux[l_bound:u_bound]
    wav = wav[l_bound:u_bound]

    # interpolate the spectrum to make it more detailed
    minimum = wav[np.argmin(wav)]
    maximum = wav[np.argmax(wav)]
    xwav = np.arange(minimum, maximum, 0.01)

    # Adjust the size of the flux data to fit the mask
    interp_func = interp1d(wav, flux, bounds_error=False, fill_value=np.nan)
    padded_flux = interp_func(xwav)

    # normalise the continuum
    s_flux = savgol_filter(padded_flux, window_length=501, polyorder=4)

    n_flux = padded_flux / s_flux

    # n_flux = flux

    # now add voight profiles to the continuum where the lines should be:
    for i in c_lines:
        # Mg4481 line
        center = float(i[1])
        amplitude = -0.05
        sigma = 0.3
        gamma = 0.1
        model = VoigtModel()
        # print((model.func(wav, amplitude, center, sigma, gamma)))
        n_flux -= 4 * np.abs(model.func(xwav, amplitude, center, sigma, gamma))
    return xwav, n_flux



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


















#Direct fitting voigt functions:


def Gaussian(wavelength, flux, moving_avg, errors, c_lines, time, n, plot_dir):
    # Laura's Gaussian fitting
    # Function that takes a SALT spectrogram file for a specific date spectrograph reading, fits voigt functions to each of the spectral line dips individually,
    # and outputs the mean shift from the theoretical lab values in terms of the radial velocity of the star this corresponds to, with a corresponding error
    # Inputs:
    # wavelength

    print("------------NEW DATE FILE -------------")
    v_precision = 0.2  # kniazev stability analysis for SALT HRS LRS
    w_size = 30
    snr_cutoff = 10  # 13
    line_depth_cutoff = 0  # 3
    sky_line_depth_cutoff = 3
    # initialise result arrays
    RV = np.array([])
    RV_weight = np.array([])
    RV_err = np.array([])

    # initialise the arrays for plotting
    D = np.array([])
    SNR = np.array([])
    wavelengths = []
    best_fits = np.array([])
    flux_results = np.array([])

    for i in c_lines:

        # change plot directory
        wd = os.path.join(plot_dir, str(i[0]) + "/")
        # Create the directory if it doesn't exist
        if not os.path.exists(wd):
            os.makedirs(wd)

        # gwav,gdata,gerrors,gweights,D,snr = make_window(wav,flux,moving_avg,i,w_size):

        # find the upper bound of the window
        u_loc = np.searchsorted(xwav, (i[1] + (w_size / 2)))
        closest_value = xwav[max(0, u_loc - 1)]
        upper_bound = np.where(xwav == closest_value)
        u_bound = int(upper_bound[0].item())
        # find the lower bound of the window
        l_loc = np.searchsorted(xwav, (i[1] - (w_size / 2)))
        closest_value = xwav[max(0, l_loc - 1)]
        lower_bound = np.where(xwav == closest_value)
        l_bound = int(lower_bound[0].item())
        # make small datasets around the line
        gdata = flux[l_bound:u_bound]
        gwav = wavelength[l_bound:u_bound]
        gavg = moving_avg[l_bound:u_bound]
        gavg = np.where(gavg != 0, gavg, 1)
        # make a new window for errors which is just outside the window - basically the next window down instead
        # Instead we want to make a small window around the absorption line
        # and then take the errors from the points around it
        # and then take the average value of that
        # and set the errors of the line to it
        el_loc = np.searchsorted(xwav, (i[1] - (1 / 10) * (w_size)))
        closest_value = xwav[max(0, el_loc - 1)]
        elower_bound = np.where(xwav == closest_value)
        el_bound = int(elower_bound[0].item())
        eh_loc = np.searchsorted(xwav, (i[1] + (1 / 10) * (w_size)))
        closest_value = xwav[max(0, eh_loc - 1)]
        ehigher_bound = np.where(xwav == closest_value)
        eh_bound = int(ehigher_bound[0].item())

        firsterrorbox = errors[l_bound:el_bound]
        seconderrorbox = errors[eh_bound:u_bound]
        err_avg = (np.mean(firsterrorbox) + np.mean(seconderrorbox)) / 2
        lineerroravg = [] * (eh_bound - el_bound)
        lineerroravg[el_bound:eh_bound] = [err_avg] * (eh_bound - el_bound)

        gerrors = np.concatenate((firsterrorbox, lineerroravg, seconderrorbox))
        # gerrors[np.isnan(gerrors)] = 0.001
        gerrors[: len(gdata)]

        gweights = np.where(gerrors != 0, 1 / gerrors, 0.001)

        # Calculate the mean signal
        signal_mean = (
            (np.mean(flux[l_bound:el_bound]) + np.mean(flux[eh_bound:u_bound])) / 2
        ) - np.min(flux[el_bound:eh_bound])
        # Calculate the line depth
        depth = np.abs(signal_mean / err_avg)
        D = np.append(D, depth)

        # now calculate snr
        snr = np.median(flux[l_bound:el_bound]) / (np.std(flux[l_bound:el_bound]))
        SNR = np.append(SNR, snr)

        # set the initial guess of the mean value of the gaussian to the wavelength in air
        line = i[1]
        try:

            voigt_model = VoigtModel(prefix="voigt_")
            linear_model = LinearModel(prefix="linear_")
            poly_model = PolynomialModel(degree=3)
            # combine models - can swap in or out the polynomial or linear model as we wish
            composite_model = voigt_model + poly_model
            # if line_type is a sky line then it is an emission line with a positive peak, if not, then it is an absorption line with negative peak
            
            # CHANGE THE SNR VALUE TO FILTER DATASETS THAT ARE NOT FITTING WELL
            # We want this code to try the next line in the dataset
            if snr <= snr_cutoff:
                raise Exception(f"SNR too low: {snr} <{snr_cutoff}")
            if depth <= line_depth_cutoff:
                raise Exception(
                    f"Line not significant: Line depth {depth} < {line_depth_cutoff}"
                )
            ####

            print(f"SNR is {snr:.3g}")
            params = voigt_model.make_params(
                voigt_amplitude=-0.01,
                voigt_center=line,
                voigt_sigma=0.3,
                voight_gamma=1,
            )
            params["voigt_amplitude"].min = -2
            params["voigt_amplitude"].max = 0.0
            params["voigt_center"].min = line - 2
            params["voigt_center"].max = line + 2
            #print("absorption params made")
            # params += linear_model.make_params(slope=0, intercept=np.median(gdata))
            params += poly_model.guess(gdata, x=gwav)

            result = composite_model.fit(
                gdata, params, x=gwav, weights=gweights
            )  # ,sigma = wav_errors)
            # print(result.values['voigt_center'],result.params['voigt_center'])
            #print("The results of the fitting report are: ")
            #print(result.fit_report())
            # find the individual polynomial best fit
            c0 = result.params["c0"].value
            c1 = result.params["c1"].value
            c2 = result.params["c2"].value
            c3 = result.params["c3"].value
            p_result = c0 + c1 * gwav + c2 * gwav**2 + c3 * gwav**3

            t = time.datetime
            # plt.show()
            print(result.values['voigt_center'])
            # print(result.fit_report())
            Line_Offset = result.values["voigt_center"] - line
            rv = (Line_Offset / line) * 299792
            # print(f"The voigt sigma is {(result.params['voigt_sigma'])} and the center error is{ result.params['voigt_center'].stderr}")
            if result.params["voigt_center"].stderr is not None:
                err = (
                    (result.params["voigt_center"].stderr * 2.35)
                    / result.values["voigt_center"]
                ) * 299792

            else:
                err = 0.2
            # err = (((result.params['voigt_sigma'].value)/result.values['voigt_center']) * 299792)

            # Append values to RV
            RV = np.append(RV, rv)
            RV_err = np.append(RV_err, (err))  # +v_precision
            RV_weight = np.append(RV_weight, (1 / (err)))

            # save the datafiles instead of stacking the plot here
            data = pd.DataFrame(
                {
                    "Wavelength": (gwav),
                    "Normalized Data": (gdata / p_result),
                    "Voigt fit": (result.best_fit / p_result),
                    "Time": t,
                    "SNR": snr,
                    "Depth": depth,
                    "RV": rv,
                    "Error": err,
                }
            )  # [t],[snr], [depth], [rv], [err]])
            safe_t = sanitize_component(t)
            base = Path("C:/Users/OmriNolan/OneDrive - SUPER-SHARP Space Systems Limited/Documents")
            dir_path = base / "Paper_project" / "Results" / "resultfiles" / "WD1929" / "Voigt_fitting" / "red_lines" / "corrected_for_pixel" / safe_t
            dir_path.mkdir(parents=True, exist_ok=True)
            file_name = dir_path / f"{line}.txt"


            # np.savetxt(file_name_without_spaces, data, delimiter=',', fmt = '%f') #"fmt=['%f', '%f', '%f', '%s', '%f', '%f', '%f', '%f'])
            # data.to_csv(file_name_without_spaces, sep='\t', index=False)
            # how to read the data back in
            #
            # Now save the line plot with the best fit and the shaded errors
            plt.figure(facecolor="white", figsize=(8, 6))
            # add spectrum
            plt.plot(gwav, gdata / p_result, color="black", linewidth=0.5)
            # add best fit line
            plt.plot(gwav, result.best_fit / p_result, color="red")
            # shade standard deviation of data
            plt.fill_between(
                gwav,
                (result.best_fit - gerrors) / p_result,
                (result.best_fit + gerrors) / p_result,
                color="gray",
                alpha=0.3,
            )

            plt.title(
                f"{t.year}/{t.month}/{t.day}  Line Centre:{result.values['voigt_center']:.7g} Å"
            )
            # plt.text(0.95, 0.10, f"SNR = {snr:.3g}", horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
            # plt.text(0.95, 0.05, f" RV = {rv:.{4}g} ± {err:.{2}g} km/s", horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
            plt.ylim(0.5, 1.3)
            plt.xlabel("Wavelength (Å)")
            plt.ylabel("Normalized Flux")
            title = str(i[0]) + "_" + str(t) + ".png"
            title_without_spaces = title.replace(" ", "")
            # plt.savefig(os.path.join(wd, title_without_spaces))
            # plt.show()
            plt.close()

            print("line depth: ", depth, "SNR: ", snr)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_num = exc_tb.tb_lineno
            line_str = linecache.getline(file_name, line_num).strip()

            # Print out the error message along with the line of code
            print(f"Error occurred at line {line_num}: {line_str}")
            print(f"Error occurred: {e}")
            RV = np.append(RV, np.nan)
            RV_err = np.append(RV_err, np.nan)
            RV_weight = np.append(RV_weight, np.nan)
            continue

    # weighted mean and standard deviation error calculations
    #if np.any(~np.isnan(RV)):
    # Weighted mean and standard deviation error calculations
    mean = np.sum((RV[~np.isnan(RV)] * RV_weight[~np.isnan(RV_weight)])) / (
        np.sum(RV_weight[~np.isnan(RV_weight)])
    )
    mean_error = np.sqrt(
        np.sum(RV_weight[~np.isnan(RV_weight)] * RV_err[~np.isnan(RV_err)] ** 2)
        / np.sum(RV_weight[~np.isnan(RV_weight)])
        + v_precision**2
    )

    print("The rv mean and error is", mean, mean_error)

    return mean, mean_error, n


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
):
    # Voigt fitting for the MIKE Spectrum
    print("------------NEW--FILE----------", str(time))
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
                plt.figure()
                plt.plot(wav_value, flux_list[j])
                plt.show()
                if float(np.min(wav_value)) <= float(i[1]) <= float(np.max(wav_value)):
                    print(f" For order {j} the line is in the order")

                    xwav = wav_value
                    xflux = flux_list[j]
                    xmoving_avg = moving_avg_list[j]
                    xerrors = errors_list[j]
                    xsnr = snr_list[j]

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
                    gsnr = xsnr[l_bound:u_bound]
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

                    plt.figure()
                    plt.plot(gwav, p_result, color="green", linewidth=1)
                    plt.plot(gwav, gdata, color="blue", linewidth=0.5)

                    # plt.show()
                    plt.close()

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
                    #print("The results of the fitting report are: ")
                    #print(result.fit_report())

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
                            f"{t.year}/{t.month}/{t.day}, Order: {j},  Line Centre:{result.values['voigt_center']}"
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
                        dir_name = fr"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/resultfiles/WD1929/MIKE_Voigt_fitting/all_lines/{t}/"
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
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_num = exc_tb.tb_lineno
            line_str = linecache.getline(file_name, line_num).strip()

            # Print out the error message along with the line of code
            print(f"Error occurred at line {line_num}: {line_str}")
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


#Salt cross-correlation

def RVCC(
    wavelength_reference,
    flux_reference,
    wavelength_observed,
    flux_observed,
    window_size,
):
    # Working cross- correlation pyastronomy crosscorrRV with bimodal gaussian fits to the central cc peak

    tw, tf = process_data_cc(wavelength_reference, flux_reference)
    dw, df = process_data_cc(wavelength_observed, flux_observed)

    # find the upper bound of the window
    u_loc = np.searchsorted(tw, 5200)
    closest_value = tw[max(0, u_loc - 1)]
    u_bound = np.where(tw == closest_value)
    u_bound = int(u_bound[0])
    # find the lower bound of the window
    l_loc = np.searchsorted(tw, 4100)
    closest_value = tw[max(0, l_loc - 1)]
    l_bound = np.where(tw == closest_value)
    l_bound = int(l_bound[0])
    # make new datasets to exclude the edge of the files
    tf = tf[l_bound:u_bound]
    tw = tw[l_bound:u_bound]

    # make a new dataset for the observed wav
    u_loc = np.searchsorted(dw, 5200)  # 5200
    closest_value = dw[max(0, u_loc - 1)]
    u_bound = np.where(dw == closest_value)
    u_bound = int(u_bound[0])
    # find the lower bound of the window
    l_loc = np.searchsorted(dw, 4100)  # 4100
    closest_value = dw[max(0, l_loc - 1)]
    l_bound = np.where(dw == closest_value)
    l_bound = int(l_bound[0])
    df = df[l_bound:u_bound]
    dw = dw[l_bound:u_bound]

    df = np.clip(df, 0, 2)
    tf = np.clip(tf, 0, 2)

    # Carry out the cross-correlation.
    # The first and last x points of the data are skipped.
    rv, cc = pyasl.crosscorrRV(
        dw, df, tw, tf, -window_size, window_size, 0.1, skipedge=50
    )
    n_cc = cc / np.abs(max(cc))

    gaussian1 = GaussianModel(prefix="gaussian1_")
    voigt = VoigtModel(prefix="voigt_")
    gaussian2 = GaussianModel(prefix="gaussian2_")
    linear_model = LinearModel(prefix="linear_")
    composite_model = gaussian1 + gaussian2 + linear_model
    params = linear_model.make_params(slope=0, intercept=np.min(n_cc))
    """ 
    params += voigt.make_params(voigt_amplitude=0.1, voigt_center=0, voigt_sigma=1.,voigt_gamma=1)
    params['voigt_amplitude'].min = 0
    params['voigt_amplitude'].max = 1
    params['voigt_center'].min = (-10)
    params['voigt_center'].max = (10)
    params['voigt_sigma'].min = (0)
    params['voigt_sigma'].max = (15)
    """
    params += gaussian1.make_params(
        gaussian1_amplitude=0.1, gaussian1_center=0, gaussian1_sigma=1.0
    )
    params["gaussian1_amplitude"].min = 0
    params["gaussian1_amplitude"].max = 1
    params["gaussian1_center"].min = -30
    params["gaussian1_center"].max = 30
    params["gaussian1_sigma"].min = 0
    params["gaussian1_sigma"].max = 15

    params += gaussian2.make_params(
        gaussian2_amplitude=0.01, gaussian2_center=0, gaussian2_sigma=50
    )
    params["gaussian2_amplitude"].min = 0
    params["gaussian2_amplitude"].max = 1
    params["gaussian2_center"].min = -30
    params["gaussian2_center"].max = 30
    params["gaussian2_sigma"].min = 10
    params["gaussian2_sigma"].max = 1000

    result = composite_model.fit(n_cc, params, x=rv)  # ,sigma = wav_errors)
    print(result.values["gaussian1_center"], result.params["gaussian1_center"])
    max_rv = result.values["gaussian1_center"]
    sigma = result.params["gaussian1_sigma"].value

    # shifted wavelength data
    rv_dw = dw * (1 + max_rv / 299792)

    plt.figure(figsize=(10, 10), dpi=200)
    # Plot template and data
    plt.subplot(2, 1, 1)
    # plt.title("Spectrum cross-correlation against reference spectrum")
    plt.plot(tw, tf, color="blue", linewidth=0.5, label="Reference Template")
    # plt.text(4483,1.3, f"RV shift = {max_rv:.3f} ± {sigma:.2f}")
    plt.errorbar(
        rv_dw, df, color="green", linewidth=0.5, label="Shifted Observed Spectrum"
    )  # need to shift this back by the rv shift
    plt.xlim(4470, 4490)
    plt.ylim(0.5, 1.6)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Normalised Flux")
    plt.legend()

    # plot the cross-correlation function and fit
    plt.subplot(2, 1, 2)
    plt.errorbar(rv, n_cc, fmt=".", label="Data", zorder=0)  # ccfs_err
    plt.plot(rv, result.best_fit, "r-", label="Fit", zorder=1)
    plt.xlabel("Delta RV (km/s)")
    plt.ylabel("CCF")
    plt.legend()
    # plt.show()
    plt.close()
    print(max_rv)
    print(f"Optimized mu: {max_rv:.2f} km/s")

    return max_rv, sigma



def RVCC_MIKE(tw, tf, dw, df, line, obs_time):
    window_size = 100

    # Create the new dataset excluding the edge points
    exclude_points = int(0.06 * len(tf))
    tf = tf[exclude_points:-exclude_points]
    tw = tw[exclude_points:-exclude_points]
    df = df[exclude_points:-exclude_points]
    dw = dw[exclude_points:-exclude_points]

    # normalize the dataset
    ref_p_result = poly_fit(tw, tf)
    obs_p_result = poly_fit(dw, df)
    ntf = tf / ref_p_result
    ndf = df / obs_p_result

    print(tw[0], dw[0])
    rv, cc = pyasl.crosscorrRV(
        dw, ndf, tw, ntf, -window_size, window_size, 0.1, skipedge=50
    )

    n_cc = cc / np.abs(max(cc))
    init = rv[np.argmax(cc)]
    print(f"The max radial velocity from the cc argmax = {init}")

    # ccf fitting
    gaussian1 = GaussianModel(prefix="gaussian1_")
    # voigt = VoigtModel(prefix = "voigt_")
    gaussian2 = GaussianModel(prefix="gaussian2_")
    linear_model = LinearModel(prefix="linear_")
    composite_model = gaussian1 + gaussian2 + linear_model
    params = linear_model.make_params(slope=0, intercept=np.min(n_cc))
    """
    params += voigt.make_params(voigt_amplitude=0.1, voigt_center=0, voigt_sigma=1.,voigt_gamma=1)
    params['voigt_amplitude'].min = 0
    params['voigt_amplitude'].max = 1
    params['voigt_center'].min = (25)
    params['voigt_center'].max = (55)
    params['voigt_sigma'].min = (0)
    params['voigt_sigma'].max = (15)
    """
    params += gaussian1.make_params(
        gaussian1_amplitude=0.01, gaussian1_center=init, gaussian1_sigma=1.0
    )
    params["gaussian1_amplitude"].min = 0
    params["gaussian1_amplitude"].max = 1
    params["gaussian1_center"].min = -1
    params["gaussian1_center"].max = 1
    params["gaussian1_sigma"].min = 0
    params["gaussian1_sigma"].max = 5

    params += gaussian2.make_params(
        gaussian2_amplitude=0.1, gaussian2_center=init, gaussian2_sigma=50
    )
    params["gaussian2_amplitude"].min = 0
    params["gaussian2_amplitude"].max = 1
    # params['gaussian2_center'].min = (-)
    # params['gaussian2_center'].max = (60)
    params["gaussian2_sigma"].min = 0
    params["gaussian2_sigma"].max = 1000

    result = composite_model.fit(n_cc, params, x=rv)  # ,sigma = wav_errors)
    # print(result.values['voigt_center'],result.params['voigt_center'])

    # results from fit
    max_rv = result.values["gaussian1_center"]
    sigma = result.params["gaussian1_sigma"].value

    # max_rv = init
    # sigma = 0.2
    # shifted wavelength data
    rv_dw = dw * (1 - max_rv / 299792)

    # plotting for the spectrum and ccf
    t = obs_time.datetime
    plt.figure(figsize=(10, 10))
    # Plot template and data
    plt.subplot(2, 1, 1)
    plt.title("Spectrum cross-correlation against reference spectrum")
    plt.plot(tw, ntf, color="blue", linewidth=0.5, label="Reference Spectrum")
    plt.text(0.7 * np.max(tw), 1.25, f"RV shift = {max_rv:.3f} ± {sigma:.2f}")
    plt.errorbar(
        rv_dw,
        ndf,
        color="green",
        linewidth=0.5,
        label=f"Shifted spectrum: {t.year}/{t.month}/{t.day}",
    )
    # plt.xlim(line -10,line +10)
    # plt.ylim(0.4,1.5)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Normalised Flux")
    plt.legend()

    # plot the cross-correlation function and fit
    plt.subplot(2, 1, 2)
    plt.errorbar(rv, n_cc, fmt=".", label="Data", zorder=0)  # ccfs_err
    plt.plot(rv, result.best_fit, "r-", label="Fit", zorder=1)
    plt.xlabel("V (km/s)")
    plt.ylabel("CCF")
    plt.legend()
    plt.show()
    plt.close()
    print(max_rv)
    print(f"Optimized mu: {max_rv:.2f} km/s")

    # save the datafiles instead of stacking the plot here
    
    # data = pd.DataFrame(
    #     {
    #         "template wavelength": tw,
    #         "template flux": ntf,
    #         "Shifted observed wavelength": rv_dw,
    #         "observed flux": ndf,
    #         "RV": (rv),
    #         "CC": (n_cc),
    #         "Bimodal fit": (result.best_fit),
    #         "Time": t,
    #         "RV": max_rv,
    #         "Error": sigma,
    #     }
    # )  # [t],[snr], [depth], [rv], [err]])
    # dir_name = (
    #     fr"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/resultfiles/WD1929/Reference_CCF/wholespectrum/"
    # )
    # dir_name_without_spaces = dir_name.replace(" ", "")
    # os.makedirs(dir_name_without_spaces, exist_ok=True)
    # file_end = f"{t}.txt"
    # file_name = os.path.join(dir_name_without_spaces, file_end)
    # file_name_without_spaces = file_name.replace(" ", "")
    # # np.savetxt(file_name_without_spaces, data, delimiter=',', fmt = '%f') #"fmt=['%f', '%f', '%f', '%s', '%f', '%f', '%f', '%f'])
    # data.to_csv(file_name_without_spaces, sep="\t", index=False)
    # # how to read the data back in
    # # data = pd.read_csv(r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/resultfiles/WD1929/Voigt_fitting/only_mg_line/snr_cutoff_16.05612842459828/2021-04-21T03:09:52.703/4481.185.txt", sep='\t')
    
    return max_rv, sigma




def RVCC_model(
    model_wav,
    model_flux,
    wavelength_observed,
    flux_observed,
    window_size,
    b_lines,
    time,
):
    # Working cross- correlation pyastronomy crosscorrRV with bimodal gaussian fits to the central cc peak

    t = time
    tw, tf = process_data_model(model_wav, model_flux, b_lines)
    dw, df = process_data_cc(wavelength_observed, flux_observed)

    """   #find the upper bound of the window
    u_loc = np.searchsorted(dw, 5200)
    closest_value = tw[max(0, u_loc-1)]
    u_bound = np.where(tw == closest_value)
    u_bound = int(u_bound[0])
    #find the lower bound of the window
    l_loc = np.searchsorted(tw, 4100)
    closest_value = tw[max(0, l_loc-1)]
    l_bound = np.where(tw == closest_value)
    l_bound = int(l_bound[0])
    #make new datasets to exclude the edge of the files
    tf = tf[l_bound:u_bound]
    tw = tw[l_bound:u_bound] """

    # make a new dataset for the observed wav
    u_loc = np.searchsorted(dw, 5200)  # 5200
    closest_value = dw[max(0, u_loc - 1)]
    u_bound = np.where(dw == closest_value)
    u_bound = int(u_bound[0].item())
    # find the lower bound of the window
    l_loc = np.searchsorted(dw, 4100)  # 4100
    closest_value = dw[max(0, l_loc - 1)]
    l_bound = np.where(dw == closest_value)
    l_bound = int(l_bound[0].item())
    df = df[l_bound:u_bound]
    dw = dw[l_bound:u_bound]

    df = np.clip(df, 0, 2)
    # tf = np.clip(tf, 0, 2)
    """ 
    plt.figure()
    plt.plot(tw,tf,color = "green",label = "Model spectrum")
    plt.legend()
    plt.show()
     """
    # Carry out the cross-correlation.
    # The first and last x points of the data are skipped.
    rv, cc = pyasl.crosscorrRV(
        dw, df, tw, tf, -window_size + 35, window_size + 35, 0.1, skipedge=500
    )
    n_cc = cc / np.abs(max(cc))

    init = rv[np.argmax(cc)]
    print(f"The max radial velocity from the cc argmax = {init}")

    gaussian1 = GaussianModel(prefix="gaussian1_")
    voigt = VoigtModel(prefix="voigt_")
    gaussian2 = GaussianModel(prefix="gaussian2_")
    linear_model = LinearModel(prefix="linear_")
    composite_model = gaussian1 + gaussian2 + linear_model
    params = linear_model.make_params(slope=0, intercept=np.min(n_cc))
    """
    params += voigt.make_params(voigt_amplitude=0.1, voigt_center=0, voigt_sigma=1.,voigt_gamma=1)
    params['voigt_amplitude'].min = 0
    params['voigt_amplitude'].max = 1
    params['voigt_center'].min = (25)
    params['voigt_center'].max = (55)
    params['voigt_sigma'].min = (0)
    params['voigt_sigma'].max = (15)
    """
    params += gaussian1.make_params(
        gaussian1_amplitude=0.01, gaussian1_center=init, gaussian1_sigma=1.0
    )
    params["gaussian1_amplitude"].min = 0
    params["gaussian1_amplitude"].max = 1
    params["gaussian1_center"].min = 30
    params["gaussian1_center"].max = 50
    params["gaussian1_sigma"].min = 0
    params["gaussian1_sigma"].max = 20

    params += gaussian2.make_params(
        gaussian2_amplitude=0.1, gaussian2_center=init, gaussian2_sigma=50
    )
    params["gaussian2_amplitude"].min = 0
    params["gaussian2_amplitude"].max = 1
    params["gaussian2_center"].min = 0
    params["gaussian2_center"].max = 70
    params["gaussian2_sigma"].min = 5
    params["gaussian2_sigma"].max = 1000

    result = composite_model.fit(n_cc, params, x=rv)  # ,sigma = wav_errors)
    # print(result.values['voigt_center'],result.params['voigt_center'])

    # results from fit
    max_rv = result.values["gaussian1_center"]
    sigma = result.params["gaussian1_sigma"].value

    # max_rv = init
    # sigma = 0.2
    # shifted wavelength data
    rv_dw = dw * (1 - max_rv / 299792)

    plt.figure(figsize=(10, 10))
    # Plot template and data
    plt.subplot(2, 1, 1)
    plt.plot(tw, tf, color="blue", linewidth=0.5, label="Koester Model Spectrum")
    # plt.text(4485,1.25, f"RV shift = {max_rv:.3f} ± {sigma:.2f}")
    plt.errorbar(
        rv_dw, df, color="green", linewidth=0.5, label="Shifted observed spectrum"
    )  # need to shift this back by the rv shift
    plt.xlim(4471, 4491)
    plt.ylim(0.4, 1.5)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Normalised Flux")
    plt.legend()

    # plot the cross-correlation function and fit
    plt.subplot(2, 1, 2)
    plt.errorbar(rv, n_cc, fmt=".", label="Data", zorder=0)  # ccfs_err
    plt.plot(rv, result.best_fit, "r-", label="Fit", zorder=1)
    plt.xlabel("Delta V (km/s)")
    plt.ylabel("CCF")
    plt.legend()
    # plt.show()
    plt.close()
    print(max_rv)
    print(f"Optimized mu: {max_rv:.2f} km/s")

    # save the datafiles instead of stacking the plot here
    
    return max_rv, sigma, init




io_noise = []
# dictionary for all the inter-order noise areas for the SALT blue spectrum
# manually removing them
io_noise.append((3890, 3910))
io_noise.append((3890, 3910))
io_noise.append((3934, 3945))
io_noise.append((3960, 3972))
io_noise.append((3995, 4010))
io_noise.append((4030, 4040))
io_noise.append((4063, 4075))
io_noise.append((4103, 4115))
io_noise.append((4140, 4150))
io_noise.append((4175, 4185))
io_noise.append((4175, 4185))
io_noise.append((4215, 4225))
io_noise.append((4175, 4185))
io_noise.append((4255, 4265))
io_noise.append((4290, 4305))
io_noise.append((4330, 4345))
io_noise.append((4370, 4382))
io_noise.append((4415, 4428))
io_noise.append((4455, 4470))
io_noise.append((4495, 4510))
io_noise.append((4545, 4555))
io_noise.append((4455, 4470))
io_noise.append((4585, 4600))
io_noise.append((4633, 4650))
io_noise.append((4680, 4690))
io_noise.append((4728, 4740))
io_noise.append((4777, 4790))
io_noise.append((4826, 4835))
io_noise.append((4878, 4888))
io_noise.append((4928, 4938))
io_noise.append((4980, 4988))
io_noise.append((5032, 5050))
io_noise.append((5080, 5100))
io_noise.append((5148, 5158))
io_noise.append((5208, 5215))
io_noise.append((5265, 5275))
io_noise.append((5325, 5335))
io_noise.append((5325, 5335))
io_noise.append((5385, 5395))
io_noise.append((5400, 5600))  # big cut off for second channel (red)
io_noise.append((5625, 5635))
io_noise.append((5695, 5705))
io_noise.append((5765, 5779))


# line dictionaries: Name, position(A), present?, log gf
p_lines = []
spec_lines = []

# pollutant lines
p_lines.append(("CaII_3933", 3933.663, False))  # only use for MIKE data
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
p_lines.append(("MgII_4481", 4481.180, True))  # weighted combination of mg_4481 lines

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


star = "WD1929+012"
SALT_folder_path = r"C:\\Users\\OmriNolan\\OneDrive - SUPER-SHARP Space Systems Limited\Documents\\Paper_project\Data\SALT\WD1929+012"
MIKE_blue_folder_path = r"C:\\Users\\OmriNolan\\OneDrive - SUPER-SHARP Space Systems Limited\Documents\\Paper_project\Data\\MIKE\WD1929+011\\blue"
MIKE_red_folder_path = r"C:\\Users\\OmriNolan\\OneDrive - SUPER-SHARP Space Systems Limited\Documents\\Paper_project\Data\\MIKE\WD1929+011\\red"



blue_start = "mbgphH"
red_start = "mbgphR"
end_patterns = "u2wm.fits"  # End pattern for the object fiber
sky_end_patterns = "u1wm.fits"  # End pattern for the sky fiber
merged_end_patterns = "uwm.fits"  # end pattern for the reduced fiber
mike_start = "gal"
mike_end = ".fits"

# Creating file directory for the blue and red channels separately
b_files = pick_files_by_patterns(SALT_folder_path, blue_start, end_patterns)
merged_b_files = pick_files_by_patterns(
    SALT_folder_path, blue_start, merged_end_patterns
)
r_files = pick_files_by_patterns(SALT_folder_path, red_start, end_patterns)
sky_r_files = pick_files_by_patterns(SALT_folder_path, red_start, sky_end_patterns)
mike_b_files = mike_pick_files_by_patterns(MIKE_blue_folder_path, mike_start, mike_end)
mike_r_files = mike_pick_files_by_patterns(MIKE_red_folder_path, mike_start, mike_end)

b_files.sort()
r_files.sort()
sky_r_files.sort()
mike_b_files.sort()
mike_r_files.sort()
bad_date = [
    "/data/wdplanetary/omri/Data/WD1929+012/2019-1-SCI-008.20190513",
    "/data/wdplanetary/omri/Data/WD1929+012/2020-1-SCI-043.20200531",
    "/data/wdplanetary/omri/Data/WD1929+012/2018-1-SCI-043.20180605",
    "/data/wdplanetary/omri/Data/WD1929+012/2019-2-SCI-049.20200824",
    "/data/wdplanetary/omri/Data/WD1929+012/2017-1-SCI-031.20170813/product/mbgphH20170813003",
]

b_files = [f for f in b_files if not any(f.startswith(start) for start in bad_date)]
r_files = [f for f in r_files if not any(f.startswith(start) for start in bad_date)]
sky_r_files = [
    f for f in sky_r_files if not any(f.startswith(start) for start in bad_date)
]
merged_b_files = [
    f for f in merged_b_files if not any(f.startswith(start) for start in bad_date)
]


















#################  RUNFILES ################

#####SALT Direct fitting running and plotting

if salt_direct:

    # Gaussian fits runfile for both channels
    # using code from above and iterating over the filenames
    # making arrays to plot values after
    rvs = []
    time_strings = []
    rverrs = []
    rv_corrs = []
    rv_corrs_errs = []
    star = "WD1929"
    n = 0
    blue_plot_dir = r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/Gaussianline/blue/"
    red_plot_dir = r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/Gaussianline/red/"
    sky_plot_dir = r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/Gaussianline/sky/"
    # stacked_plot_dir =  r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/stackedplots/"
    # fig, ax = plt.subplots(figsize=(6, 20))
    for j, k, s in zip(b_files, r_files, sky_r_files):
        wav, flux, time, bOBJECT = Get_Wavelength_Flux_File(j)
        skywav, skyflux, skytime, SKY = Get_Wavelength_Flux_File(s)
        rwav, rflux, rtime, rOBJECT = Get_Wavelength_Flux_File(k)
        if bOBJECT.startswith(star):
            # then call data processing
            xwav, padded_flux, moving_avg = process_data_gaussian(wav, flux)
            errors = calculate_error(padded_flux)
            # print(errors,padded_flux,moving_avg)
            # then do Gaussian
            rv, rv_err, n = Gaussian(
                xwav, padded_flux, moving_avg, errors, b_lines, time, n, blue_plot_dir
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
        else:
            print("This is a file for " + str(bOBJECT) + " instead of " + str(star))
            continue
        # Now do the same for the red wav and the sky spectra - use the sky spectra to correct the red spectra
        if rOBJECT.startswith(star):
            # then call data processing
            xwav, padded_flux, moving_avg = process_data_gaussian(rwav, rflux)
            errors = calculate_error(padded_flux)
            # print(errors,padded_flux,moving_avg)
            # then do Gaussian
            rv, rv_err, n = Gaussian(
                xwav, padded_flux, moving_avg, errors, r_lines, time, n, red_plot_dir
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
        else:
            print("This is a file for " + str(bOBJECT) + " instead of " + str(star))
            continue
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
    print(f"Averages are: {averages}")  #
    print("Time          ,RVs          ,Errors        ") 
    print(f"{rvs}")
    # Build the target directory and file path robustly
    base = Path("C:/Users/OmriNolan/OneDrive - SUPER-SHARP Space Systems Limited/Documents")
    save_dir = base / "Paper_project" / "Results" / "resultfiles" / "SALT_Voigt"
    save_dir.mkdir(parents=True, exist_ok=True)  # <-- ensure directory exists
    file_path = save_dir / "all_lines_pixel_corrected.txt"
    # Now save (np.savetxt accepts Path in recent NumPy; str(...) is universally safe)
    np.savetxt(str(file_path), averages, fmt="%.6f")  # adjust fmt & array as needed
    # rv variations with time plotting
    #file_path = save_dir / "voigt_fitting.txt"

    times = []
    delta_rvs = []
    errors = []
    frequencies = []
    powers = []
    # normal textfiles:
    #data = np.loadtxt(file_path, delimiter=",")
    data = averages
    print(data)
    # data = np.delete((data), [2],axis = 0)
    salt_directline_t = data[:, 0]  # First column
    salt_directline_mean = np.nanmean(data[:, 1])
    salt_directline_v = data[:, 1] - salt_directline_mean  # Second column
    salt_directline_errs = data[:, 2]  # Third column
    salt_directline_variance = np.var(salt_directline_v)  # + np.mean(errs)**2
    salt_directline_stdev = np.sqrt(salt_directline_variance)
    # frequencies = np.linspace(1.95,2.05,3000)
    salt_directline_frequencies = np.linspace(0.001, 5, 3000)
    salt_directline_powers = LombScargle(salt_directline_t, salt_directline_v, salt_directline_errs).power(salt_directline_frequencies)







####MIKE Direct fitting running and plotting:

if mike_direct:
    # Voigt fitting runfile for MIKE
    rvs = []
    time_strings = []
    rverrs = []

    n = 0
    mike_blue_plot_dir = r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/mike/Gaussianline/blue/"
    stacked_plot_dir = r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/stackedplots/"

    star = "G"

    stacked_plot_dir = r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/stackedplots/"
    # fig, ax = plt.subplots(figsize=(6, 20))

    for files_list in (mike_r_files, mike_b_files):
        for j in files_list:
            order, wav, flux, snr, OBJECT, time = read_mike_spec(j)

            xwav_list = np.empty(len(order), dtype=object)
            padded_flux_list = np.empty(len(order), dtype=object)
            moving_avg_list = np.empty(len(order), dtype=object)
            errors_list = np.empty(len(order), dtype=object)
            snr_list = np.empty(len(order), dtype=object)

            print("imported files")
            if OBJECT.startswith(star):
                # then call data processing
                for o in order:
                    xwav, padded_flux, moving_avg = process_data_mike_gaussian(
                        wav[o - 1], flux[o - 1], r_lines
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
                    r_lines,
                    time,
                    n,
                    mike_blue_plot_dir,
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
            else:
                print("This is a file for " + str(OBJECT) + " instead of " + str(star))
                continue
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
    print(f"Averages are: {averages}")  #
    print("Time          ,RVs          ,Errors        ") 
    print(f"{rvs}")
    # Build the target directory and file path robustly
    base = Path("C:/Users/OmriNolan/OneDrive - SUPER-SHARP Space Systems Limited/Documents")
    save_dir = base / "Paper_project" / "Results" / "resultfiles" / "MIKE_Voigt"
    save_dir.mkdir(parents=True, exist_ok=True)  # <-- ensure directory exists
    file_path = save_dir / "all_lines_pixel_corrected.txt"
    # Now save (np.savetxt accepts Path in recent NumPy; str(...) is universally safe)
    np.savetxt(str(file_path), averages, fmt="%.6f")  # adjust fmt & array as needed
    # rv variations with time plotting
    #file_path = save_dir / "voigt_fitting.txt"

    times = []
    delta_rvs = []
    errors = []
    frequencies = []
    powers = []
    # normal textfiles:
    #data = np.loadtxt(file_path, delimiter=",")
    data = averages
    print(data)
    # data = np.delete((data), [2],axis = 0)
    mike_directline_t = data[:, 0]  # First column
    mike_directline_mean = np.nanmean(data[:, 1])
    mike_directline_v = data[:, 1] - mike_directline_mean  # Second column
    mike_directline_errs = data[:, 2]  # Third column
    mike_directline_variance = np.var(mike_directline_v)  # + np.mean(errs)**2
    mike_directline_stdev = np.sqrt(mike_directline_variance)
    # frequencies = np.linspace(1.95,2.05,3000)
    mike_directline_frequencies = np.linspace(0.001, 5, 3000)
    mike_directline_powers = LombScargle(mike_directline_t, mike_directline_v, mike_directline_errs).power(mike_directline_frequencies)


# Now need to plot and all for MIKE





# SALT self-correlation running and plotting
if salt_selfcc:
    window_size = 100.0
    wav_reference, flux_reference, ref_time, ref_object = Get_Wavelength_Flux_File(
        r"Users\\OmriNolan\\OneDrive - SUPER-SHARP Space Systems Limited\Documents\\Paper_project\Data\SALT\WD1929+012\\2017-1-SCI-031.20170706\\product\\._mbgphR201707060020_uwm.fits"
        )
    # could also use this as the best: /data/wdplanetary/omri/Data/WD1929+012/2017-1-SCI-031.20170730/product/mbgphH201707300036_u2wm.fits
    time_strings = [(ref_time)]
    vels = [(0)]
    rverrs = [(0)]
    snr_cutoff = 8
    for i in b_files:
        wav_observed, flux_observed, obs_time, obs_object = Get_Wavelength_Flux_File(i)
        if obs_object.startswith(str(ref_object)):
            snr = np.median(flux_observed) / (np.std(flux_observed))
            # print(snr)
            rv_max, sigma = RVCC(
                wav_reference, flux_reference, wav_observed, flux_observed, window_size
            )
            # bvels.append((rv[np.argmax(cc)]))
            # rv_max,sigma = CCF_processing_gaussian(rv,cc)
            # popt,pcov = CCF_processing(rv,cc)
            # print(popt,pcov)
            vels.append(rv_max)
            time_strings.append(obs_time)
            rverrs.append(sigma)
        else:
            print("This is a file for" + str(obs_object))
            continue
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
    print(f"Averages are: {averages}")  #
    print("Time          ,RVs          ,Errors        ") 
    print(f"{rvs}")
    # Build the target directory and file path robustly
    base = Path("C:/Users/OmriNolan/OneDrive - SUPER-SHARP Space Systems Limited/Documents")
    save_dir = base / "Paper_project" / "Results" / "resultfiles" / "SALT_SelfCCF"
    save_dir.mkdir(parents=True, exist_ok=True)  # <-- ensure directory exists
    file_path = save_dir / "bluespec.txt"
    # Now save (np.savetxt accepts Path in recent NumPy; str(...) is universally safe)
    np.savetxt(str(file_path), averages, fmt="%.6f")  # adjust fmt & array as needed
    # rv variations with time plotting
    #file_path = save_dir / "voigt_fitting.txt"

    times = []
    delta_rvs = []
    errors = []
    frequencies = []
    powers = []
    # normal textfiles:
    #data = np.loadtxt(file_path, delimiter=",")
    data = averages
    print(data)
    # data = np.delete((data), [2],axis = 0)
    salt_selfccf_t = data[:, 0]  # First column
    salt_selfccf_mean = np.nanmean(data[:, 1])
    salt_selfccf_v = data[:, 1] - salt_selfccf_mean  # Second column
    salt_selfccf_errs = data[:, 2]  # Third column
    salt_selfccf_variance = np.var(salt_selfccf_v)  # + np.mean(errs)**2
    salt_selfccf_stdev = np.sqrt(salt_selfccf_variance)
    # frequencies = np.linspace(1.95,2.05,3000)
    salt_selfccf_frequencies = np.linspace(0.001, 5, 3000)
    salt_selfccf_powers = LombScargle(salt_selfccf_t, salt_selfccf_v, salt_selfccf_errs).power(salt_selfccf_frequencies)




##### MIKE Self CCF Running and plotting 

if mike_selfcc:
    # CCF self-correlation for MIKE
    rvs = []
    time_strings = []
    rverrs = []

    n = 0
    mike_blue_plot_dir = r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/mike/referenceCCF/"
    stacked_plot_dir = r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/stackedplots/"

    ref_order, ref_wav, ref_flux, ref_snr, refOBJECT, ref_time = read_mike_spec(
        r"C:\\Users\\OmriNolan\\OneDrive - SUPER-SHARP Space Systems Limited\\Documents\\Paper_project\\Data\\MIKE\\WD1929+011\\blue\\galex1931_blue_2010-06-16.fits"
    )
    rvs.append(0)
    time_strings.append(ref_time)
    rverrs.append(0)

    star = "G"

    # fig, ax = plt.subplots(figsize=(6, 20))

    for j in mike_b_files:
        order, wav, flux, snr, obs_OBJECT, obs_time = read_mike_spec(j)

        xwav_list = np.empty(len(order), dtype=object)
        padded_flux_list = np.empty(len(order), dtype=object)
        moving_avg_list = np.empty(len(order), dtype=object)
        errors_list = np.empty(len(order), dtype=object)
        snr_list = np.empty(len(order), dtype=object)

        print("imported files")
        if obs_OBJECT.startswith(star):
            # then call data processing
            for o in order:
                # xwav,padded_flux,moving_avg = process_data_mike_gaussian(wav[o-1],flux[o-1],b_lines)
                obs_wav, obs_flux = wav[o - 1], flux[o - 1]
                r_wav, r_flux = ref_wav[o - 1], ref_flux[o - 1]

                for i in b_lines:
                    line = float(i[1])
                    if np.min(obs_wav) <= line <= np.max(obs_wav):

                        rv, err = RVCC_MIKE(
                            r_wav, r_flux, obs_wav, obs_flux, line, obs_time
                        )
                        rvs.append(rv)
                        time_strings.append(obs_time)
                        rverrs.append(err)
                        print("The fit has been calculated successfully")
        else:
            print("This is a file for " + str(obs_OBJECT) + " instead of " + str(star))
            continue

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
    print(f"Averages are: {averages}")  #
    print("Time          ,RVs          ,Errors        ") 
    print(f"{rvs}")
    # Build the target directory and file path robustly
    base = Path("C:/Users/OmriNolan/OneDrive - SUPER-SHARP Space Systems Limited/Documents")
    save_dir = base / "Paper_project" / "Results" / "resultfiles" / "MIKE_SelfCCF"
    save_dir.mkdir(parents=True, exist_ok=True)  # <-- ensure directory exists
    file_path = save_dir / "bluespec.txt"
    # Now save (np.savetxt accepts Path in recent NumPy; str(...) is universally safe)
    np.savetxt(str(file_path), averages, fmt="%.6f")  # adjust fmt & array as needed
    # rv variations with time plotting
    #file_path = save_dir / "voigt_fitting.txt"

    times = []
    delta_rvs = []
    errors = []
    frequencies = []
    powers = []
    # normal textfiles:
    #data = np.loadtxt(file_path, delimiter=",")
    data = averages
    print(data)
    # data = np.delete((data), [2],axis = 0)
    mike_selfccf_t = data[:, 0]  # First column
    mike_selfccf_mean = np.nanmean(data[:, 1])
    mike_selfccf_v = data[:, 1] - mike_selfccf_mean  # Second column
    mike_selfccf_errs = data[:, 2]  # Third column
    mike_selfccf_variance = np.var(mike_selfccf_v)  # + np.mean(errs)**2
    mike_selfccf_stdev = np.sqrt(mike_selfccf_variance)
    # frequencies = np.linspace(1.95,2.05,3000)
    mike_selfccf_frequencies = np.linspace(0.001, 5, 3000)
    mike_selfccf_powers = LombScargle(mike_selfccf_t, mike_selfccf_v, mike_selfccf_errs).power(mike_selfccf_frequencies)






####SALT Model CCF Running and plotting
if salt_modelcc:
    # Runfile for the cross-correlation with the model spectrum
    star = "WD1929+012"
    # import the Koester model
    data_path = r"C:\\Users\\OmriNolan\\OneDrive - SUPER-SHARP Space Systems Limited\\Documents\\Paper_project\\Code\\RadVel\\da21000_800.dk.dat.txt"
    wavelengths = []
    fluxes = []
    with open(data_path, "r") as file:
        lines = file.readlines()
    for line in lines[7:]:
        # Split the line into columns (assuming whitespace as delimiter)
        columns = line.split()
        
        if len(columns) < 2:
            continue   # skip empty or malformed line
        # Convert wavelength and flux to float and append to respective lists
        wavelengths.append(columns[0])
        fluxes.append(columns[1])
    # Convert lists to NumPy arrays
    m_wav = np.array(wavelengths).astype(float)
    m_flux = np.array(fluxes).astype(float)
    model_wav, model_flux = process_data_model(m_wav, m_flux, b_lines)

    window_size = 100.0
    time_strings = []
    rvs = []
    vels_from_argmax = []
    rverrs = []
    snr_cutoff = 10
    # running part
    for i in b_files:
        wav_observed, flux_observed, obs_time, obs_object = Get_Wavelength_Flux_File(i)
        if obs_object.startswith(star):
            # snr = np.median(flux_observed)/(np.std(flux_observed))
            # print(snr)
            rv_max, sigma, rv_max_from_argmax = RVCC_model(
                model_wav,
                model_flux,
                wav_observed,
                flux_observed,
                window_size,
                b_lines,
                obs_time,
            )
            # bvels.append((rv[np.argmax(cc)]))
            # rv_max,sigma = CCF_processing_gaussian(rv,cc)
            # popt,pcov = CCF_processing(rv,cc)
            # print(popt,pcov)
            rvs.append(rv_max)
            time_strings.append(obs_time)
            rverrs.append(sigma)
            vels_from_argmax.append(rv_max_from_argmax)
        else:
            print("This is a file for" + str(obs_object))
            continue

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
    ]  
    print(f"filtered indicies are: {filtered_indices}")# & (rvarray > 10)
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
    print(f"Averages are: {averages}")  #
    print("Time          ,RVs          ,Errors        ") 
    print(f"{rvs}")
    # Build the target directory and file path robustly
    base = Path("C:/Users/OmriNolan/OneDrive - SUPER-SHARP Space Systems Limited/Documents")
    save_dir = base / "Paper_project" / "Results" / "resultfiles" / "SALT_SelfCCF"
    save_dir.mkdir(parents=True, exist_ok=True)  # <-- ensure directory exists
    file_path = save_dir / "bluespec.txt"
    # Now save (np.savetxt accepts Path in recent NumPy; str(...) is universally safe)
    np.savetxt(str(file_path), averages, fmt="%.6f")  # adjust fmt & array as needed
    # rv variations with time plotting
    #file_path = save_dir / "voigt_fitting.txt"

    times = []
    delta_rvs = []
    errors = []
    frequencies = []
    powers = []
    # normal textfiles:
    #data = np.loadtxt(file_path, delimiter=",")
    data = averages
    
    # data = np.delete((data), [2],axis = 0)
    salt_modelccf_t = data[:, 0]  # First column
    salt_modelccf_mean = np.nanmean(data[:, 1])
    salt_modelccf_v = data[:, 1] # Second column
    salt_modelccf_errs = data[:, 2]  # Third column
    salt_modelccf_variance = np.var(salt_modelccf_v)  # + np.mean(errs)**2
    salt_modelccf_stdev = np.sqrt(salt_modelccf_variance)
    # frequencies = np.linspace(1.95,2.05,3000)
    salt_modelccf_frequencies = np.linspace(0.001, 5, 3000)
    salt_modelccf_powers = LombScargle(salt_modelccf_t, salt_modelccf_v, salt_modelccf_errs).power(salt_modelccf_frequencies)
    print("*****************")
    print(salt_modelccf_t)
    
    # plotting for the model_ cross_correlation
    tarray = np.array(salt_modelccf_t)
    rvarray = np.array(salt_modelccf_v)
    rverrsarray = np.array(salt_modelccf_errs)

    # rverrsarray = np.full(len(tarray),2)

    """ filtered_indices = np.where((rverrsarray < 100) & (rvarray > 0))[0] #& (rvarray > 10)
    filtered_rvarray = rvarray[filtered_indices]
    filtered_tarray = tarray[filtered_indices]
    filtered_rverrsarray = rverrsarray[filtered_indices]
    filtered_rvcorrsarray = rvcorrsarray[filtered_indices] """

    
    unique_times = np.unique(tarray)

    averages = np.array(
        [
            [t, np.nanmean(rvarray[tarray == t]), np.nanmean(rverrsarray[tarray == t])]
            for t in unique_times
        ]
    )
    # averages = averages[np.isnan(averages[:,1])]

    # print(averages)
    file_name = r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/resultfiles/crosscorrRV_model_mg4481_bothlines_results.txt"
    np.savetxt(file_name, averages, delimiter=",", fmt="%f")

    # Radial Velocity variations with time plot

    t = averages[:, 0]
    v = averages[:, 1]
    mean = np.mean(v)
    v = v - np.mean(v)
    # errs = np.full(len(v),0.2)
    errs = averages[:, 2] + 0.2
    stdev = np.std(v)

    """ fig, (ax_t, ax_w) = plt.subplots(2, 1,figsize = (10,6), constrained_layout=True)
    ax_t.errorbar(t, y, yerr=dy, fmt='b+', label='Data with Errors')
    ax_t.set_xlabel('Time [days]')
    ax_t.legend()

    ax_w.plot(frequency, power)
    ax_w.set_xlabel('Angular frequency [rad/days]')
    ax_w.set_ylabel('Normalized amplitude')
    plt.show() """

    plt.figure(facecolor="white", figsize=(10, 6))
    plt.errorbar(t, v, yerr=errs, fmt=".k", lw=0.4)
    plt.fill_between(t, -stdev, stdev, color="gray", label="1 sigma", alpha=0.4)
    plt.fill_between(t, -3 * stdev, 3 * stdev, color="gray", label="3 sigma", alpha=0.2)
    plt.legend()
    legend_pos = plt.gca().get_legend().get_texts()[0].get_position()
    plt.text(1200, 3, f"RV mean = {mean:.3g}")
    plt.text(1200, 2.5, f"sigma = {stdev:.3g}")


    plt.xlabel("Time - Days")
    plt.ylabel("Change in Radial Velocity - km/s")
    # plt.title("WD1929+012 Radial Velocity variations using Voigt fits - SALT data")
    plt.text
    plt.gcf().autofmt_xdate()
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.ylim(-20,20)
    # plt.xlim(0,112)
    plt.savefig(r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/DeltaRV_CrossCorr_model_4481_bothlines.pdf")
    plt.show()





###################### proper clean plot


# rv variations with time plotting
times = []
delta_rvs = []
errors = []
frequencies = []
powers = []
# normal textfiles:
salt_direct_file_name = r"C:\\Users\\OmriNolan\\OneDrive - SUPER-SHARP Space Systems Limited\Documents\\Paper_project\\Results\\resultfiles\\SALT_Voigt\\.txt"
#mike_direct_file_name = "a"
salt_directline_data = np.loadtxt(file_name, delimiter=" ")
print(data)
# data = np.delete((data), [2],axis = 0)
t = data[:, 0]  # First column
mean = np.nanmean(data[:, 1])
v = data[:, 1] - mean  # Second column
errs = data[:, 2]  # Third column
variance = np.var(v)  # + np.mean(errs)**2
stdev = np.sqrt(variance)
# frequencies = np.linspace(1.95,2.05,3000)
frequencies = np.linspace(0.001, 5, 3000)
powers = LombScargle(t, v, errs).power(frequencies)
fig, (ax_t, ax_w) = plt.subplots(
    2, 1, facecolor="white", figsize=(12, 14), constrained_layout=True, dpi=400
)

ax_t.errorbar(t, v, yerr=errs, fmt=".k", capsize=5, lw=1.5)
ax_t.fill_between(t, -stdev, stdev, color="gray", label="1 sigma", alpha=0.4)
ax_t.fill_between(t, -3 * stdev, 3 * stdev, color="gray", label="3 sigma", alpha=0.2)
ax_t.axhline(y=0, color="black", linestyle="--", linewidth=1)
ax_t.legend()
# ax_t.text(np.max(t)/3, 2.5* stdev, f"RV mean = {mean:.3g} km/s")
ax_t.text(np.max(t) / 3, 2 * stdev, f"sigma = {stdev:.3g} km/s")
ax_t.set_xlabel("T - Days")
ax_t.set_ylabel("ΔRV - km/s")

normalized_powers = powers / (np.max(np.abs(powers)))
ax_w.plot(frequencies, normalized_powers)
ax_w.set_xlabel("Angular frequency [Periods/days]")
ax_w.set_ylabel("Normalized Power")
ax_w.tick_params(axis="x")
ax_w.tick_params(axis="y")
# Identify significant peaks (you can set your own threshold here)
threshold = 0.3  # Adjust as needed
significant_peak_indices = np.where(normalized_powers >= threshold)[0]
print(significant_peak_indices)
print(frequencies[1])
significant_peaks = np.array(frequencies[significant_peak_indices])

# Estimate false alarm rate for each peak
false_alarm_rates = []
for peak_freq in significant_peaks:
    # Use your preferred method to estimate false alarm rate here
    # Example: Monte Carlo simulations
    num_simulations = 400  # Adjust as needed
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










#########Violin plots and fit comparisons


mg4481 = np.loadtxt(
    r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/resultfiles/voigt_fitting_mg4481.txt", delimiter=","
)
crosscorr_bluespec = np.loadtxt(
    r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/resultfiles/crosscorrRV_bluespec_results.txt",
    delimiter=",",
)
crosscorr_model = np.loadtxt(
    r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/resultfiles/crosscorrRV_model_mg4481_results.txt",
    delimiter=",",
)
data = np.array(
    (
        mg4481[:, 1] - np.mean(mg4481[:, 1]),
        crosscorr_bluespec[:, 1],
        crosscorr_model[:, 1] - np.mean(crosscorr_model[:, 1]),
    )
)
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_ylabel("Radial Velocity variation (km/s)")


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6), sharey=True)

# ax.set_title('Violin plot')
parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

for pc in parts["bodies"]:
    pc.set_facecolor("#D43F3A")
    pc.set_edgecolor("black")
    pc.set_alpha(1)

# Initialize lists to store quartiles and adjacent values for each column
all_quartile1 = []
all_medians = []
all_quartile3 = []
all_whiskers_min = []
all_whiskers_max = []

# Iterate over each column and calculate quartiles and adjacent values
for column in data:
    q1, median, q3 = np.percentile(column, [25, 50, 75])
    all_quartile1.append(q1)
    all_medians.append(median)
    all_quartile3.append(q3)
    whiskers = adjacent_values(np.sort(column), q1, q3)
    all_whiskers_min.append(whiskers[0])
    all_whiskers_max.append(whiskers[1])

inds = np.arange(1, len(all_medians) + 1)
ax.scatter(inds, all_medians, marker="o", color="white", s=30, zorder=3)
ax.vlines(inds, all_quartile1, all_quartile3, color="k", linestyle="-", lw=5)
ax.vlines(inds, all_whiskers_min, all_whiskers_max, color="k", linestyle="-", lw=1)

# set style for the axes
labels = [
    "Voigt fit on Mg 4481 line",
    "Self Cross-Correlation",
    "Model Cross-Correlation",
]
set_axis_style(ax, labels)

plt.savefig(r"C:\Users\OmriNolan\OneDrive - SUPER-SHARP Space Systems Limited\Documents\Paper_project\Results/violinplots/voigt_vs_crosscorr.pdf")
# plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()
