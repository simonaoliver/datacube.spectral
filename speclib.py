'''
Functions to support HSI image visualisation and spectral analysis

License: The code in this notebook is licensed under the Apache License,
Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).

Last modified: December 2023
'''

# Import required packages
import os, sys
import numpy as np
import xarray 
import rioxarray
import holoviews as hv
from holoviews import opts
import geoviews as gv
import datashader as ds
import cartopy.crs as ccrs
import hvplot
from holoviews.operation.datashader import regrid, shade
from bokeh.tile_providers import STAMEN_TONER
import rasterio
from osgeo import gdal
from pathlib import Path
#import panel
import hvplot.xarray
from ipywidgets import interact, Dropdown, FloatSlider, IntSlider, SelectMultiple, Text
import matplotlib.pyplot as pyplot
import xmltodict
import zipfile
import tarfile
from skimage import exposure, img_as_float
import pandas
import cv2

import math
import requests
import xml.etree.ElementTree as ET
import io
from scipy.interpolate import UnivariateSpline
from scipy import signal
from scipy import ndimage
from scipy import stats
from scipy.stats import norm
from scipy.stats import wasserstein_distance
import pandas
from pathlib import Path
import plotly
plotly.offline.init_notebook_mode()
import zipfile
import netCDF4 
import csv
import matplotlib.pyplot as pyplot
import matplotlib
import warnings
cm = pyplot.cm

import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from bisect import bisect_left
from scipy import stats
import re

warnings.simplefilter("ignore")

# Define custom functions
# From LPDAAC EMIT Tools
def emit_xarray(filepath, ortho=True, qmask=None, unpacked_bmask=None): 
    """
        This function utilizes other functions in this module to streamline opening an EMIT dataset as an xarray.Dataset.
        
        Parameters:
        filepath: a filepath to an EMIT netCDF file
        ortho: True or False, whether to orthorectify the dataset or leave in crosstrack/downtrack coordinates.
        qmask: a numpy array output from the quality_mask function used to mask pixels based on quality flags selected in that function. Any non-orthorectified array with the proper crosstrack and downtrack dimensions can also be used.
        unpacked_bmask: a numpy array from  the band_mask function that can be used to mask band-specific pixels that have been interpolated.
                        
        Returns:
        out_xr: an xarray.Dataset constructed based on the parameters provided.

        """
    # Read in Data as Xarray Datasets
    ds = xarray.open_dataset(filepath,engine = 'h5netcdf')
    loc = xarray.open_dataset(filepath, engine = 'h5netcdf', group='location')
    wvl = xarray.open_dataset(filepath, engine = 'h5netcdf', group='sensor_band_parameters') 
    
    # Building Flat Dataset from Components
    data_vars = {**ds.variables} 
    coords = {'downtrack':(['downtrack'], ds.downtrack.data),'crosstrack':(['crosstrack'],ds.crosstrack.data), **loc.variables, **wvl.variables}
    out_xr = xarray.Dataset(data_vars=data_vars, coords = coords, attrs= ds.attrs)
    out_xr.attrs['granule_id'] = os.path.splitext(os.path.basename(filepath))[0]
    
    # Apply Quality and Band Masks
    if qmask is not None:
        out_xr[list(out_xr.data_vars)[0]].values[qmask == 1] = np.nan
    if unpacked_bmask is not None:
        out_xr[list(out_xr.data_vars)[0]].values[unpacked_bmask == 1] = np.nan               
    
    if ortho is True:
       out_xr = ortho_xr(out_xr)
       out_xr.attrs['Orthorectified'] = 'True'
              
    return out_xr
def ortho_xr(ds, GLT_NODATA_VALUE=0, fill_value = -9999):
    """
    This function applies the GLT to variables within an EMIT dataset that has been read into the format provided in by the emit_xarray function.

    Parameters:
    ds: an xarray dataset produced by emit_xarray
    GLT_NODATA_VALUE: no data value for the GLT tables, 0 by default
    fill_value: the fill value for EMIT datasets, -9999 by default

    Returns:
    ortho_ds: an orthocorrected xarray dataset.  
    
    """
    # Build glt_ds

    glt_ds = np.nan_to_num(np.stack([ds['glt_x'].data,ds['glt_y'].data],axis=-1),nan=GLT_NODATA_VALUE).astype(int)  
    

    # List Variables
    var_list = list(ds.data_vars)
    
    # Create empty dictionary for orthocorrected data vars
    data_vars = {}   

    # Extract Rawspace Dataset Variable Values (Typically Reflectance)    
    for var in var_list:
        raw_ds = ds[var].data
        var_dims = ds[var].dims
        # Apply GLT to dataset
        out_ds = apply_glt(raw_ds,glt_ds, GLT_NODATA_VALUE=GLT_NODATA_VALUE)

        del raw_ds
        #Update variables
        data_vars[var] = (['latitude','longitude', var_dims[-1]], out_ds)
    
    # Calculate Lat and Lon Vectors
    lon, lat = coord_vects(ds) # Reorder this function to make sense in case of multiple variables

    # Apply GLT to elevation
    #elev_ds = apply_glt(ds['elev'].data[:,:,np.newaxis],glt_ds)
    
    # Delete glt_ds - no longer needed
    del glt_ds
    
    # Create Coordinate Dictionary
    coords = {'latitude':(['latitude'],lat), 'longitude':(['longitude'],lon), **ds.coords}# unpack to add appropriate coordinates 
    
    # Remove Unnecessary Coords
    for key in ['downtrack','crosstrack','lat','lon','glt_x','glt_y','elev']:
        del coords[key]
    
    # Add Orthocorrected Elevation
    #coords['elev'] = (['latitude','longitude'], np.squeeze(elev_ds))

    # Build Output xarray Dataset and assign data_vars array attributes
    out_xr = xarray.Dataset(data_vars=data_vars, coords=coords, attrs=ds.attrs)
    
    del out_ds
    # Assign Attributes from Original Datasets
    out_xr[var].attrs = ds[var].attrs
    out_xr.coords['latitude'].attrs = ds['lat'].attrs
    out_xr.coords['longitude'].attrs = ds['lon'].attrs
    
    # Add Spatial Reference in recognizable format
    out_xr.rio.write_crs(ds.spatial_ref,inplace=True)

    # Mask Fill Values
    out_xr[var].data[out_xr[var].data == fill_value] = np.nan
    
    return out_xr

# Function to Apply the GLT to an array
def apply_glt(ds_array,glt_array,fill_value=-9999,GLT_NODATA_VALUE=0):
    """
    This function applies a numpy array of the EMIT glt to a numpy array of the desired dataset (i.e. reflectance, radiance, etc. 
    
    Parameters:
    ds_array: numpy array of the desired variable
    glt_array: a GLT array constructed from EMIT GLT data
    
    Returns: 
    out_ds: a numpy array of orthorectified data.
    """

    # Build Output Dataset
    out_ds = np.full((glt_array.shape[0], glt_array.shape[1], ds_array.shape[-1]), fill_value, dtype=np.float32)
    valid_glt = np.all(glt_array != GLT_NODATA_VALUE, axis=-1)
    
    # Adjust for One based Index
    glt_array[valid_glt] -= 1 
    out_ds[valid_glt, :] = ds_array[glt_array[valid_glt, 1], glt_array[valid_glt, 0], :]
    return out_ds

# Function to Calculate the Lat and Lon Vectors/Coordinate Grid
def coord_vects(ds):
    """
    This function calculates the Lat and Lon Coordinate Vectors using the GLT and Metadata from an EMIT dataset read into xarray.
    
    Parameters:
    ds: an xarray.Dataset containing the root variable and metadata of an EMIT dataset
    loc: an xarray.Dataset containing the 'location' group of an EMIT dataset

    Returns:
    lon, lat (numpy.array): longitute and latitude array grid for the dataset

    """
    # Retrieve Geotransform from Metadata
    GT = ds.geotransform
    # Create Array for Lat and Lon and fill
    dim_x = ds.glt_x.shape[1]
    dim_y = ds.glt_x.shape[0]
    lon = np.zeros(dim_x)
    lat = np.zeros(dim_y)
    # Note: no rotation for EMIT Data
    for x in np.arange(dim_x):
        x_geo = (GT[0]+0.5*GT[1]) + x * GT[1] # Adjust coordinates to pixel-center
        lon[x] = x_geo
    for y in np.arange(dim_y):
        y_geo = (GT[3]+0.5*GT[5]) + y * GT[5]
        lat[y] = y_geo
    return lon,lat
    
def updatexarray(ds, banddim, xdim, ydim, wavelengthlist, fwhmlist): 
    if xdim !='x' and ydim != 'y':
        ds = ds.rename_dims({'band':banddim, 'x': xdim,'y':ydim})
    ds = ds.assign_coords({'wavelengths':wavelengthlist, 'fwhm': fwhmlist})
    ds = ds.rename({'band_data':'reflectance'})
    ds = ds.drop_indexes(['wavelengths', 'fwhm'])
    return(ds)
    
def set_band_descriptions(filepath, bands, null):
    """
    Take a filepath to a raster, opens the file and updates descriptions as defined in bands

    Parameters
    ----------
    filepath : path/virtual path/uri to raster
    bands : tuples representing the reference band number and the description of to be added 
        ((band, description), (band, description),...)
    """
    ds = gdal.Open(filepath, gdal.GA_Update)
    for band, desc, metadata in bands:
        rb = ds.GetRasterBand(band)
        rb.SetNoDataValue(null)
        rb.SetDescription(desc)
        rb.SetMetadata(metadata)
    del ds

def rgbhsi(xarraydataarray, rband, gband, bband, lowerq, upperq, dim):
    r = xarraydataarray[[rband]]
    g = xarraydataarray[[gband]]
    b = xarraydataarray[[bband]]
    
    rmin, rmax = xarraydataarray[[rband]].quantile([lowerq,upperq]).values
    gmin, gmax = xarraydataarray[[gband]].quantile([lowerq,upperq]).values
    bmin, bmax = xarraydataarray[[bband]].quantile([lowerq,upperq]).values
    
    r_rescale = exposure.rescale_intensity(r, in_range=(rmin, rmax))
    g_rescale = exposure.rescale_intensity(g, in_range=(gmin, gmax))
    b_rescale = exposure.rescale_intensity(b, in_range=(bmin, bmax))
    
    return(xarray.concat((r_rescale,g_rescale,b_rescale), dim=dim))

def reshapeprisma(ndarray):
    '''Consume the PRISMA 3D nparray as read by h5py and reshape to be more consistent with the y,x,band arrangement'''
    reshaped = []
    count = 0
    while count < ndarray.shape[1]:
        if count == 0:
            reshaped = ndarray[:,count,:]
            reshaped = reshaped[..., np.newaxis]
        else:
            newnp = ndarray[:,count,:]
            newnp = newnp[..., np.newaxis]   
            reshaped = np.append(reshaped, newnp, axis=2) 
        count = count+1

    return(reshaped)
def CreateGeoTiff(outRaster, data, projection, geo_transform):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols, no_bands = data.shape
    DataSet = driver.Create(outRaster, cols, rows, no_bands, gdal.GDT_Int16)
    DataSet.SetGeoTransform(geo_transform)
    DataSet.SetProjection(projection)

    #data = np.moveaxis(data, -1, 0)
    data = np.moveaxis(data, 2, 0)
    count = 1
    for i, image in reversed(list(enumerate(data, 1))):
        
        DataSet.GetRasterBand(count).WriteArray(image)
        count = count + 1
    DataSet = None

def CreateENVI(outRaster, data, projection, geo_transform):
    driver = gdal.GetDriverByName('ENVI')
    rows, cols, no_bands = data.shape
    DataSet = driver.Create(outRaster, cols, rows, no_bands, gdal.GDT_Int16)
    DataSet.SetGeoTransform(geo_transform)
    DataSet.SetProjection(projection)

    #data = np.moveaxis(data, -1, 0)
    data = np.moveaxis(data, 2, 0)
    count = 1
    for i, image in reversed(list(enumerate(data, 1))):
        
        DataSet.GetRasterBand(count).WriteArray(image)
        count = count + 1
    DataSet = None

def getbands(nparray, prefix, cwl):
    tmpbands = list(range(1,nparray.shape[2]+1))
    bands = []
    count = 0
    cwl = list(reversed(cwl.split()))
    for band in tmpbands:
        bands.append((band, prefix+str(band)+" "+str(int(float(cwl[count])))))
        count = count+1
    return(bands)
    
def renamevars(rioxr, prefix):
    count= 0
    renamedict = {}
    while count < len(rioxr.data_vars):
        #name = (prefix+str(count+1)+_+rioxr.attrs['long_name'][count])
        name = rioxr.attrs['long_name'][count]
        renamedict[count+1] = name
        count=count+1
    return(rioxr.rename(renamedict))

def normalise(rsr):
    return(rsr/rsr.max())

# Plot every sensor band for a given sensor
def plotallbands(sensordict, sensor):
    plotlydatalist = []
    
    rows = len(sensordict[sensor].keys())
    cmap = cm.get_cmap('gist_rainbow_r', rows)
    count = 0                   
    for key in sensordict[sensor].keys():
        
        plotlydatalist.append({"x": sensordict[sensor][(key)].wavelength,\
                                "y": normalise(sensordict[sensor][(key)].rsr),\
                                "name": (key),\
                                "line": {'color': matplotlib.colors.rgb2hex(cmap(count)), 'width': 2}})
        count = count+1
    plotly.offline.iplot({ "data": plotlydatalist,"layout": {"title": sensor}})

def plotallbandslist(sensordict, sensorlist, plotname):

    
    plotlydatalist = []
    red = 255
    green = 255
    blue = 255
    greenorange = 165
    for sensor in sensorlist:
        red = red - 10
        green = green - 10
        blue = blue -10 
        greenorange = greenorange - 2

        
        #rows = len(sensordict[sensor].keys())
        #cmap = cm.get_cmap('gist_rainbow_r', rows)
        
        for i, val in enumerate(set(sensordict[sensor].keys())):
            
            if i == 0: 
                color = 'rgb(255,'+str(greenorange)+', 0)'
            if i == 1:
                color = 'rgb(0,0,'+str(blue)+')'
            if i == 2:
                color = 'rgb(0,'+str(green)+', 0)'
            if i == 3:
                color = 'rgb('+str(red)+', 0, 0)'
            
            plotlydatalist.append({"x": sensordict[sensor][list(sensordict[sensor])[i]].wavelength,\
                                    "y": normalise(sensordict[sensor][list(sensordict[sensor])[i]].rsr),
                                    "name": (sensor+' '+str(val)),\
                                    "line": {'color': color, 'width': 0.5}})
                                    #"line": {'color': matplotlib.colors.rgb2hex(cmap(count)), 'width': 2}})
    plotly.offline.iplot({ "data": plotlydatalist,"layout": {"title": plotname}})

def reshape_interpolate(start, stop, samples, npdatatype, input1dwavelength,input1drsr,wlscalefactor):
    wavelength = np.linspace(start,stop,samples, dtype=float)
    rsr = np.nan_to_num(np.interp(wavelength,input1dwavelength*wlscalefactor, input1drsr))

    return wavelength, rsr

# Plot every sample for a given chapter (theme)
def plotallsamples(sampledict,chapter):
    plotlydatalist = []
    for key in sampledict[chapter].keys():
        plotlydatalist.append({"x": sampledict[chapter][(key)].wavelength,\
                                "y": normalise(sampledict[chapter][(key)].rsr),\
                                "name": (key)})

    plotly.offline.iplot({ "data": plotlydatalist,"layout": {"title": chapter}})

# Trim records with rsr close to equal to zero from pandas dataframe
def trim_zero_records(dataframe):
    dataframe = dataframe[dataframe['rsr'] != 0]
    dataframe = dataframe[dataframe['rsr'] != 0.0]
    return(dataframe)

def fill_nans(dataframe):
    dataframe = dataframe.fillna(0.0)
    return(dataframe)

def updatecolumns(df):
    for key in df.keys():
        if len(df[key].columns) == 3:
            df[key].columns = ['Wavelength', 'RSR', 'std']   
        if len(df[key].columns) == 2:
            df[key].columns = ['Wavelength', 'RSR']
    return(df)

# Type 2 - single dict i.e. multiple bands per sheet, one wavelength column, multiple rsr columns 
    

def multiplebandsonewavelength(sensordict, inputdict, sensor, wvscale):
    sensordict[sensor] = {}
    for key in inputdict.keys():

        if (type(inputdict) is dict) or (str(type(inputdict)) == "<class 'collections.OrderedDict'>"):
            for column in inputdict[(key)].columns:
                
                if not (column == 'Wavelength'):

                    sensorrsr = pandas.Series(inputdict[(key)][column])
                    
                    sensorwavelength = pandas.Series(inputdict[(key)].Wavelength*wvscale )
                    combinedsensors = pandas.DataFrame({'wavelength': sensorwavelength,'rsr': sensorrsr})    
                    sensordict[sensor][(column)] = combinedsensors
        else:

            for column in inputdict.columns:
                if not (column == 'Wavelength'):

                    sensorrsr = pandas.Series(inputdict[column])

                    sensorwavelength = pandas.Series(inputdict.Wavelength*wvscale )
                    combinedsensors = pandas.DataFrame({'wavelength': sensorwavelength,'rsr': sensorrsr})

                    sensordict[sensor][(column)] = combinedsensors
    return(sensordict)

# Synthetic band from centre wavelength and spline roots for fwhm
def synthetic_rsr(samples, bandcentrewavelength,fwhm):
    #returns 1d normalized relative spectral response assuming normal distibution
    sigma = fwhm / 2.35
    normdist = stats.norm.pdf(samples, loc=bandcentrewavelength, scale=sigma)
    response = (normdist - normdist.min())/(normdist.max() - normdist.min())
    return(response)


# Plot every sensor band for a given sensor and overlay a spectral sample
def plotallbandsplussample(sensordict, sampledict, sensor, category, sample):
    plotlydatalist = []
    for key in sensordict[sensor].keys():
        plotlydatalist.append({"x": sensordict[sensor][(key)].wavelength,\
                                "y": normalise(sensordict[sensor][(key)].rsr),\
                                "name": (key)})
    plotlydatalist.append({"x": sampledict[category][sample].wavelength,\
                           "y": normalise(sampledict[category][sample].rsr),\
                           "name": (sample), "line":dict(color='#0061ff', width=2, dash='dot')})    
    plotly.offline.iplot({ "data": plotlydatalist,"layout": {"title": sensor}})
    
    

# SPOT needs reorganisation

def spotdataframe(sensordict, sensor,inputframe):
    sensordict[sensor] = {}
    inputdict = inputframe
    for column in inputdict.columns:
        if not (column == 'Spectral band'):
            start = inputdict[column][0]
            step = inputdict[column][1]
            offset=2
            numsteps = inputdict[column].count()-offset
            wavelength = np.linspace(start,start+((numsteps-1)*step),numsteps)
            
            rsr = inputdict[column][2:len(wavelength)+2]
            
            sensordict[sensor][column] = pandas.DataFrame({'wavelength': wavelength, 'rsr': rsr})#.reset_index(inplace=True)
    return(sensordict)

def fwhm_gaussian(sensordict, sensor, band):
    bounds = []
    bounds.append(sensordict[sensor][(band)].wavelength\
            [sensordict[sensor][(band)].rsr.replace(0., np.nan).first_valid_index()].astype(int))
    bounds.append(sensordict[sensor][(band)].wavelength\
            [sensordict[sensor][(band)].rsr.replace(0., np.nan).last_valid_index()].astype(int))
    
    sensorwl, sensorrsr = reshape_interpolate(min(bounds),max(bounds),max(bounds)-min(bounds)+1,'float', (sensordict[sensor][(band)].wavelength).astype(int),\
                            sensordict[sensor][(band)].rsr.replace(0., np.nan), 1)
    
    A = ndimage.filters.gaussian_filter(sensorrsr, 2) #10
    spline = UnivariateSpline(sensorwl, A-A.max()/2, s=0)

    return(spline.roots())

def equivalence(sensordict, reference, target, pscore):
    # lists to hold inputs to dataframe once we're done

    sensor1list = []
    sensor1keys = []
    sensor2list = []
    sensor2keys = []
    pcorrelation = []
    emdistance = []
    weightedcentredelta = []
    areadelta = []
    fwhmdelta = []
    ks_pvalue= []
    ks_statistic = []

    # TODO update the interpolation range to fit the min and max wavelength range for the input pairwise comparison
    for sensor1 in reference:
        for key1 in sensordict[sensor1].keys():
            for sensor2 in target:

                for key2 in sensordict[sensor2].keys():
                    sensor1list.append(sensor1)
                    sensor1keys.append(key1)
                    sensor2list.append(sensor2)
                    sensor2keys.append(key2)
                    # Find the wavelength range of the rsr values and interpolate within it

                    bounds = []
                    bounds.append(sensordict[sensor1][(key1)].wavelength\
                                  [sensordict[sensor1][(key1)].rsr.replace(0., np.nan).first_valid_index()].astype(int))
                    bounds.append(sensordict[sensor1][(key1)].wavelength\
                                  [sensordict[sensor1][(key1)].rsr.replace(0., np.nan).last_valid_index()].astype(int))
                    bounds.append(sensordict[sensor2][(key2)].wavelength\
                                  [sensordict[sensor2][(key2)].rsr.replace(0., np.nan).first_valid_index()].astype(int))
                    bounds.append(sensordict[sensor2][(key2)].wavelength\
                                  [sensordict[sensor2][(key2)].rsr.replace(0., np.nan).last_valid_index()].astype(int))

                    # Interpolate rsr 
                    sensor1wl, sensor1rsr = \
                    reshape_interpolate(min(bounds),max(bounds),max(bounds)-min(bounds)+1,'float', (sensordict[sensor1][(key1)].wavelength).astype(int),\
                                        sensordict[sensor1][(key1)].rsr.replace(0., np.nan), 1)
                    sensor2wl, sensor2rsr = \
                    reshape_interpolate(min(bounds),max(bounds),max(bounds)-min(bounds)+1,'float', (sensordict[sensor2][(key2)].wavelength).astype(int),\
                                        sensordict[sensor2][(key2)].rsr.replace(0., np.nan), 1)

                    # A smoothed distrubution seems important for Earth Mover Distance
                    A = ndimage.filters.gaussian_filter(sensor1rsr, 10)
                    B = ndimage.filters.gaussian_filter(sensor2rsr, 10)

                    print ("Calculating equivalence metrics for :", sensor1, key1, "with", sensor2, key2)
                    # Earth Mover Distance 
                    try:
                        # normalise - confirm with someone who has maths skills that doing this makes sense - seems to be required for EMD
                        A = (A - A.min())/(A.max() - A.min())
                        B = (B - B.min())/(B.max() - B.min())
                        EMD = wasserstein_distance(signal.resample(A,500), signal.resample(B,500))
                    except:
                        EMD = 0
                        pass
                    EMD = wasserstein_distance(signal.resample(A,500), signal.resample(B,500))
                    emdistance.append(EMD)
                    # Pearson correlation coefficient
                    pearson = stats.pearsonr(sensor1rsr, sensor2rsr)
                    kstest = stats.ks_2samp(sensor1rsr, sensor2rsr)
                    ks_pvalue.append(kstest.pvalue)
                    ks_statistic.append(kstest.statistic)
                    pcorrelation.append(pearson[0])

                    # "Area" under each curve
                    sensor1trapz = np.trapz(sensor1rsr, sensor1wl)
                    sensor2trapz = np.trapz(sensor2rsr, sensor2wl)
                    areadelta.append(abs(sensor1trapz - sensor2trapz))
                    sensor1mean = np.average(sensor1wl, weights=sensor1rsr)
                    sensor2mean = np.average(sensor2wl, weights=sensor2rsr)
                    weightedcentredelta.append(abs(sensor1mean - sensor2mean))

                    #FWHM as spline roots
                    spline1 = UnivariateSpline(sensor1wl, A-A.max()/2, s=0)
                    spline2 = UnivariateSpline(sensor2wl, B-B.max()/2, s=0)
                    try:
                        sensor1r1, sensor1r2 = spline1.roots()
                        sensor2r1, sensor2r2 = spline2.roots()
                    except:
                        sensor1r1 = 100.
                        sensor1r2 = 100. 
                        sensor2r1 = 100.
                        sensor2r2 = 100. 
                        pass
                    fwhmdelta.append(abs((sensor1r2-sensor1r1)-(sensor2r2-sensor2r1)))

                    # Reduce the number of plots output, use a correlation threshold to determine whether to display

                    if (pearson[0] > pscore):
                        plotly.offline.iplot({
                            "data": [{"x": sensor1wl,"y": sensor1rsr, "name": sensor1+"-"+key1, "line": dict(color = ('rgb(255, 1, 1)'))},\
                                     {"x": pandas.Series([sensor1mean, sensor1mean]),"y": pandas.Series([0,1]), "name": 'mean wavelength',\
                                      "line": dict(color = ('rgb(255, 1, 1)'), width = 1, dash = 'dash')},\
                                     {"x": sensor1wl,"y": A, "name": 'A',\
                                      "name": 'gaussian', "line": dict(color = ('rgb(255, 1, 1)'), width = 1, dash = 'dot')},\
                                     {"x": pandas.Series([sensor1r1, sensor1r1]),"y": pandas.Series([0,1]), "name": 'fwhm root1',\
                                      "line": dict(color = ('rgb(255, 1, 1)'), width = 1, dash = 'dashdot')},
                                     {"x": pandas.Series([sensor1r2, sensor1r2]),"y": pandas.Series([0,1]), "name": 'fwhm root2',\
                                      "line": dict(color = ('rgb(255, 1, 1)'), width = 1, dash = 'dashdot')},


                                     {"x": sensor2wl,"y": sensor2rsr, "name": sensor2+"-"+key2, "line": dict(color = ('rgb(1, 1, 255)'))},\
                                     {"x": pandas.Series([sensor2mean, sensor2mean]),"y": pandas.Series([0,1]) , "name": 'mean wavelength',\
                                      "line": dict(color = ('rgb(1, 1, 255)'), width = 1, dash = 'dash')},
                                     {"x": sensor2wl,"y": B, "name": 'B',\
                                      "name": 'gaussian', "line": dict(color = ('rgb(1, 1, 255)'), width = 1, dash = 'dot')},\
                                     {"x": pandas.Series([sensor2r1, sensor2r1]),"y": pandas.Series([0,1]), "name": 'fwhm root1',\
                                      "line": dict(color = ('rgb(1, 1, 255)'), width = 1, dash = 'dashdot')},
                                     {"x": pandas.Series([sensor2r2, sensor2r2]),"y": pandas.Series([0,1]), "name": 'fwhm root2',\
                                      "line": dict(color = ('rgb(1, 1, 255)'), width = 1, dash = 'dashdot')},\

                                    ],

                            "layout": {"title": 'Equivalence match: '+sensor1+' '+key1+' and '+sensor2+' '+key2}})

                    del bounds, EMD, sensor1trapz, sensor2trapz, pearson, A, B, sensor1wl, sensor1rsr,sensor2wl, sensor2rsr,\
                    sensor1r1, sensor1r2, sensor2r1, sensor2r2
        global spectrumcomparison
        spectrumcomparison = pandas.DataFrame({'sensor1' : sensor1list, 'sensor1keys' : sensor1keys, 'sensor2': sensor2list,\
                                               'sensor2keys': sensor2keys, 'pcorrelation' : pcorrelation, 'ks_pvalue': ks_pvalue,\
                                               'ks_statistic': ks_statistic, 'distance': emdistance,\
                                               'areadelta': areadelta, 'weightedcentredelta': weightedcentredelta, 'fwhmdelta': fwhmdelta})

    return(spectrumcomparison)

# https://landsat.gsfc.nasa.gov/article/the-intervening-atmosphere-tracing-the-provenance-of-a-favorite-landsat-infographic
# Adapted from https://stackoverflow.com/questions/63843796/advanced-horizontal-bar-chart-with-python
# Split axis method inspired by https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
def hor_gradient_image(ax, extent, darkest, **kwargs):
    '''
    puts a horizontal gradient in the rectangle defined by extent (x0, x1, y0, y1)
    darkest is a number between 0 (left) and 1 (right) setting the spot where the gradient will be darkest
    '''
    ax = ax or pyplot.gca().margins(x=0)
    
    img = np.interp(np.linspace(0, 1, 100), [0, darkest, 1], [0, 1, 0]).reshape(1, -1)
    return ax.imshow(img, extent=extent, interpolation='bilinear', vmin=0, vmax=1, **kwargs)

def gradient_hbar(sensor_print_dict, sensor, band, y, x0, x1, color, ax, lw, height=0.05,  darkest=1, cmap=None):
    
    hor_gradient_image(ax, extent=(x0, x1, (y - (height / 2))+height, (y + (height / 2))+height), cmap=cmap, darkest=darkest)
    
    rect = mpatches.Rectangle((x0, (y - (height / 2))+height), x1 - x0, height, edgecolor='black', facecolor='none', lw=lw)
    ax.add_patch(rect)
    
    # Add white band number to bar
    left, width = x0, x1-x0
    bottom, height = y+height , 0.05
    right = left + width
    top = bottom + height  

    if (sensor_print_dict[sensor]['bands'][band]['labeltop'] == True):
        
        if re.search('[a-zA-Z]', str(sensor_print_dict[sensor]['bands'][band]['label'])):
            ax.text(0.5 * (left + right), top+(height+4), sensor_print_dict[sensor]['bands'][band]['label'], 
                 horizontalalignment='center', 
                 verticalalignment='baseline',
                 transform=ax.transData, color='black', fontsize=8, rotation=90,weight="bold")
        else:
            ax.text(0.5 * (left + right), top+(height+4), sensor_print_dict[sensor]['bands'][band]['label'], 
                 horizontalalignment='center', 
                 verticalalignment='center',
                 transform=ax.transData, color='black', fontsize=8,weight="bold")            
    else:
        ax.text(0.5 * (left + right), 0.5 * (bottom + top), sensor_print_dict[sensor]['bands'][band]['label'], 
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transData, color='white', fontsize=8,weight="bold")
      

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value#12141207
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

def colourpicker(centrewavelength, y, height, colourlut):

    colourcentre = take_closest(list(colourlut.keys()), centrewavelength)
    hexcolour = colourlut[colourcentre]['hexcolour']      
    return(y, height, hexcolour)

def getcentrewavelength(sensordict, sensor, band):
    bounds = []
    bounds.append(sensordict[sensor][(band)].wavelength\
                  [sensordict[sensor][(band)].rsr.replace(0., np.nan).first_valid_index()].astype(int))
    bounds.append(sensordict[sensor][(band)].wavelength\
                  [sensordict[sensor][(band)].rsr.replace(0., np.nan).last_valid_index()].astype(int))
    # Interpolate rsr 
    sensor1wl, sensor1rsr = \
    reshape_interpolate(min(bounds),max(bounds),max(bounds)-min(bounds)+1,'float', (sensordict[sensor][(band)].wavelength).astype(int),\
                        sensordict[sensor][(band)].rsr.replace(0., np.nan), 1)
    return int(np.average(sensor1wl, weights=sensor1rsr))
  

def plothorizontalallbands(sensordict, sensorlist, xlimmin, xlimmax, plotticklist, xlimmin2, xlimmax2, plotticklist2, sensor_print_dict, title, heightW, figureWidth, figureHeight , colourlut, lw):
# https://landsat.gsfc.nasa.gov/article/the-intervening-atmosphere-tracing-the-provenance-of-a-favorite-landsat-infographic
# Adapted from https://stackoverflow.com/questions/63843796/advanced-horizontal-bar-chart-with-python
    
    xlimmin = int(xlimmin)
    xlimmax = int(xlimmax)
    xlimmin2 = int(xlimmin2)
    xlimmax2 = int(xlimmax2)
    plotticklist = [int(str) for str in plotticklist]
    plotticklist2 = [int(str2) for str2 in plotticklist2.split(',')]
    lw = float(lw)
    
    f, (ax, ax2) = pyplot.subplots(1,2, sharex=False, sharey=False, figsize=(int(figureWidth), int(figureHeight)))
      
    ax.set_xlim(xlimmin, xlimmax)  # outliers only
    ax.set_xticks(plotticklist)
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    
    ax2.set_xlim(xlimmin2,xlimmax2)  # most of the data
    ax2.set_xticks(plotticklist2)
    ax2.set_xmargin(0)
    ax2.set_ymargin(0)
    
    # Plot parameters
    # Offset initial sensor plot to enable good fit for multiple resolutions when offset
    y = 5.0
    
    # Add irradiance curve
    xvalue = sensordict['transmission'][('solar')].wavelength
    yvalue = sensordict['transmission'][('solar')].rsr
    
    ax.plot(xvalue, yvalue*100, color='#DADFE2', zorder=-10, lw=4)
    ax.fill_between(xvalue, yvalue*100, 0, color='#DADFE2', zorder=-10)
    
    ax2.plot(xvalue, yvalue*100, color='#DADFE2', zorder=-10, lw=4)
    ax2.fill_between(xvalue, yvalue*100, 0, color='#DADFE2', zorder=-10)
    
    for subplotcount, sensor in enumerate(sensorlist):
        resolution_list=[]
        xmax = 0.0
        y2 = y 
        
        for band in sensor_print_dict[sensor]['bands']:
            resolution_list.append(sensor_print_dict[sensor]['bands'][band]['resolution'])
        mode = stats.mode(np.array(resolution_list)).mode 
        
        for index, band in enumerate(sensordict[sensor].keys()): 
        
            height = int(heightW)
            x0, x1 = fwhm_gaussian(sensordict, sensor, band)
            
            # Change the colour if band is Panchromatic or equivalent band designation 
            # "pan": True in sensor_print_dict  
            # set the centrewavelength to match colourlut
            try: 
                if (sensor_print_dict[sensor]['bands'][band]['pan'] == True):
                    centrewavelength = 0
            except:
                centrewavelength = getcentrewavelength(sensordict, sensor, band)
            
            y2, height2, hexcolour = colourpicker(centrewavelength, y, height, colourlut)
            
            # Determine if plot position needs to be offset
            # Find the mode resolution 
            # - if less than mode change y position, if greater add, if equal don't change
            
            if sensor_print_dict[sensor]['bands'][band]['resolution'] < mode:
                y2 = y - (height*0.75)
                height2 = height/1.5

            if sensor_print_dict[sensor]['bands'][band]['resolution'] > mode:
                y2 = y + (height+(height/1.75))
                height2 = height/1.5
          
            cmap = mcolors.LinearSegmentedColormap.from_list(hexcolour, [hexcolour, hexcolour])
            
            # In order to not duplicate plot labels at different scales, add them selectively depending on the bounds of each subplot
            if x1 < xlimmax:
                gradient_hbar(sensor_print_dict, sensor, band, y2, x0, x1, hexcolour, ax, lw, height=height2, darkest=0.5, cmap=cmap)
    
            else:
                gradient_hbar(sensor_print_dict, sensor, band, y2, x0, x1, hexcolour, ax2, lw, height=height2, darkest=0.5, cmap=cmap)
    
            if (x1 > xmax) and (x1 < xlimmax2):
                xmax=x1
  
        ax.set_xlim(xlimmin, xlimmax)  # outliers only
        ax.set_xticks(plotticklist)
        ax.set_xmargin(0)
        ax.set_ymargin(0)
        ax.set_aspect('auto')
        ax.use_sticky_edges = False
        ax.autoscale(enable=True, tight=False)
        
        ax2.set_xlim(xlimmin2,xlimmax2)  # most of the data
        ax2.set_xticks(plotticklist2)
        ax2.set_xmargin(0)
        ax2.set_ymargin(0)   
        ax2.set_aspect('auto')
        ax2.use_sticky_edges = False
        ax2.autoscale(enable=True, tight=False) 
        #Get launch year
        year = sensor_print_dict[sensor]['launched']

        # Use the plot specific transform to determine text placement for sensor label      
        if xmax < xlimmax:
            transform = ax.transData
            axscale = (xlimmax-xlimmin)
        else:
            transform = ax2.transData
            axscale = (xlimmax2-xlimmin2)

        textoffset = axscale/10

        # Plot Satellite Sensor name at right of bands - include year of launch
        ax2.text(xmax+(axscale/30), y+height, "}", horizontalalignment='left', verticalalignment='center',  transform=transform, color='silver', fontsize=70) 
        # Plot sensor resolution
        for resolution in set(resolution_list):
            
            if resolution < mode:
                y2 = y - (height*0.75)
                height2 = height/1.5
                ax2.text(xmax+(axscale/20), y2+height2, str(resolution)+"m", horizontalalignment='right', verticalalignment='center',  transform=transform, color='darkgray', fontsize=8) 
                
            if resolution > mode:
                y2 = y + (height+(height/1.75))
                height2 = height/1.5
                ax2.text(xmax+(axscale/20), y2+height2, str(resolution)+"m", horizontalalignment='right', verticalalignment='center',  transform=transform, color='darkgray', fontsize=8) 
            if  resolution == mode:
                ax2.text(xmax+(axscale/20), y+height, str(resolution)+"m", horizontalalignment='right', verticalalignment='center',  transform=transform, color='darkgray', fontsize=8) 
                      
        # Plot sensor name at far right of sensor 
        ax2.text(xmax+textoffset, y+height, sensor_print_dict[sensor]['shortname'], horizontalalignment='left', verticalalignment='center',  transform=transform, color='black', fontsize=12, family='sans-serif')
        
        # Plot year at far fight of sensor
        ax2.text(xlimmax2, y+height, "-"+str(year), horizontalalignment='left', verticalalignment='center',  transform=ax2.transData, color='silver', fontsize=26)

        ax.set_xlim(xlimmin, xlimmax) 
        ax2.set_xlim(xlimmin2,xlimmax2)
        ax.set_ylim(0, 100) 
        ax2.set_ylim(0,100)
        
        ax.set_yticks([0,100])
        ax.set_ylabel('Atmospheric Transmission (%)', fontsize=12)
        ax.yaxis.set_label_coords(-.02, .5)
        ax.spines.left.set_linestyle('--')
        ax.spines.left.set_color('black')
        ax.spines.left.set_linewidth(2)
        ax.spines.left.set_visible(True)

        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_color('black')
        ax.spines.bottom.set_linewidth(1)
        
        ax2.set_yticks([0,100])
        ax2.spines.left.set_visible(False)
        ax2.spines.right.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax2.spines.bottom.set_color('black')
        ax2.spines.bottom.set_linewidth(1)
        ax2.get_yaxis().set_visible(False)

        ax2.text(xlimmax, -10, 'Wavelength (nm)',horizontalalignment='center', transform=ax.transData, color='black', fontsize=12, family='sans-serif')
        
        # Link the two adjacent plots with a squiggle
        joinx, joiny = np.array([[xlimmax,xlimmax+30,xlimmax+60,xlimmax+90,xlimmax+200],[0,-2,2,0,0]])
        join = Line2D(joinx,joiny, lw=0.5, color='black', clip_on=False)
        ax.add_line(join)
       
        y = y +(100/len(sensorlist))
    
    pyplot.suptitle(title, fontsize=26)
    # Adjust distance between adjacent plots
    pyplot.subplots_adjust(wspace=0.05, hspace=0)
    pyplot.savefig('nasa_homage.png')
    pyplot.show()
