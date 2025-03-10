**Datacube Spectral** - presents an approach to statistical determination of equivalence of optical satellite sensor spectral bands. 
Originall designed to run direction on OpenDataCube, the approach is represented here as an independent library that could be adapted to run as a band selection tool for OpenDataCube.

**Determining_Equivalence_Relative_Spectral_Response.ipynb** - provides tools to enable the import and synthesis of relative spectral responses
for many of the well known commercial and government satellite sensors.

**Import sensor data** - provides import and translation functions to represent a common dataframe for spectral response functions
**Sensor plots** - demonstrates plotting functions to enable interaction with the satellite spectra
**Solar spectrum** -demonstrates atmospheric transmission relative to spectral bands
**Synthetic bands** - presents an approach to the generation of synthetic spectral responses from published band intervals (assuming Full Width Half Maximum and Gaussian distribution)
**Hyperspectral** - presents routines for import and visualisation of hyperspectral sensors
**Reference colours** - overviews an approach to creation of a synthetic spectral response for common base colours. These can then be used to determine equivalent sensor bands.
**Band equivalence** - demonstrates an approach to statistical determination of spectral equivalence between one or many sensors. Widgets enable the equivalence metrics to be modified to suit user needs.
![Band equivalence](https://github.com/simonaoliver/datacube.spectral/blob/master/equivalence.png)
**Spectral libraries** - provides access to common spectral libraries and allows their visual comparison with sensor spectral bands through interactive widgets.
![Spectral libraries](https://github.com/simonaoliver/datacube.spectral/blob/master/spectral_library.png)
**NASA sensor plot emulation** - emulates the well known NASA Landsat and Sentinel-2 plots. Controls are provided that allow the user to customise the plots.
![NASA sensor plot multispectral sensors example](https://github.com/simonaoliver/datacube.spectral/blob/master/future_optical.png)
![NASA sensor plot hyperspectral sensors example](https://github.com/simonaoliver/datacube.spectral/blob/master/future_hyperspectral.png)
**/data** - contains the source data for the creation of the sensor spectral responses in the routines outlined above
**/imagery** - contains routines for visualisation of a well known set of hyperspectral missions

**sensor_plot_spec.json** - documents configuration of the plot displays for the NASA sensor plot emulation. These can be modified to adjust labels and enable position of plot annotations.

This repository remains a work in progress.
