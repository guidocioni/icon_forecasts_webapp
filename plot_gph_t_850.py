import matplotlib.pyplot as plt
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
from glob import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
import os
from utils import *
import sys
from matplotlib.colors import BoundaryNorm

# The one employed for the figure name when exported
variable_name = 'gph_t_850'

if not sys.argv[1:]:
    print_message('Projection not defined, falling back to default (euratl, it, de)')
    projection = 'euratl'
else:    
    projection=sys.argv[1:]

"""In the main function we basically read the files and prepare the variables to be plotted.
This is not included in utils.py as it can change from case to case."""
dset = get_dset(vars_3d=['t@850','fi@500']).squeeze()
time = pd.to_datetime(dset.valid_time.values)
cum_hour = dset.step.values.astype(int)

# Select 850 hPa level using metpy
temp_850 = dset['t']
temp_850.metpy.convert_units('degC')
z_500 = dset['z']
gph_500 = mpcalc.geopotential_to_height(z_500)
gph_500 = xr.DataArray(gph_500, coords=z_500.coords,
                       attrs={'standard_name': 'geopotential height',
                              'units': gph_500.units})

levels_temp = np.arange(-30., 30., 1.)
levels_gph = np.arange(4700., 6000., 70.)

cmap = get_colormap('temp')

fig = plt.figure(figsize=(figsize_x, figsize_y))

ax = plt.gca()
temp_850, gph_500 = subset_arrays([temp_850, gph_500], projection)

lon, lat = get_coordinates(temp_850)
lon2d, lat2d = np.meshgrid(lon, lat)

ax = get_projection_cartopy(plt, projection, regions=False, compute_projection=True)

# All the arguments that need to be passed to the plotting function
print_message('Pre-processing finished, launching plotting scripts')

# Build the name of the output image
filename = projection + '_' + variable_name + '_%s.png' % cum_hour

if projection == 'euratl':
    norm = BoundaryNorm(levels_temp, ncolors=cmap.N)
    cs = ax.pcolormesh(lon2d, lat2d, temp_850, cmap=cmap, norm=norm)
else:
    cs = ax.contourf(lon2d, lat2d, temp_850, extend='both', cmap=cmap, levels=levels_temp)

c = ax.contour(lon2d, lat2d, gph_500, levels=levels_gph, colors='white', linewidths=1.)

labels = ax.clabel(c, c.levels, inline=True, fmt='%4.0f', fontsize=6)

maxlabels = plot_maxmin_points(ax, lon, lat, gph_500,
                               'max', 80, symbol='H', color='royalblue', random=True)
minlabels = plot_maxmin_points(ax, lon, lat, gph_500,
                               'min', 80, symbol='L', color='coral', random=True)

an_fc = annotation_forecast(ax, time)
an_var = annotation(
    ax, 'Geopotential height @500hPa [m] and temperature @850hPa [C]', loc='lower left', fontsize=6)
an_run = annotation_run(ax, time)

plt.colorbar(cs, orientation='horizontal',
             label='Temperature', pad=0.03, fraction=0.04)

plt.show(block=True)
#plt.savefig(filename, **options_savefig)
