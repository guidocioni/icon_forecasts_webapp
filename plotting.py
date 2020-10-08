import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
import metpy.calc as mpcalc
from glob import glob
import numpy as np
import pandas as pd
from utils import get_run, get_dset, get_colormap, subset_arrays, get_coordinates, \
    get_projection_cartopy, plot_maxmin_points, annotation_forecast,\
    annotation, annotation_run, options_savefig, figsize_x, figsize_y
from matplotlib.colors import BoundaryNorm
from multiprocessing import Pool, cpu_count
from functools import partial
import os


def plot_var(f_step, projection):
    # NOTE!
    # If we are inside this function it means that the picture does not exist
    # The one employed for the figure name when exported
    variable_name = 'gph_t_850'
    # Build the name of the output image
    run_string, _ = get_run()
    filename = '/tmp/' + projection + '_' + \
        variable_name + '_%s_%03d.png' % (run_string, f_step)

    """In the main function we basically read the files and prepare the variables to be plotted.
  This is not included in utils.py as it can change from case to case."""
    dset = get_dset(vars_3d=['t@850', 'fi@500'], f_times=f_step).squeeze()
    dset = subset_arrays(dset, projection)
    time = pd.to_datetime(dset.valid_time.values)
    cum_hour = dset.step.values.astype(int)

    temp_850 = dset['t'] - 273.15
    z_500 = dset['z']
    gph_500 = mpcalc.geopotential_to_height(z_500)
    gph_500 = xr.DataArray(gph_500.magnitude, coords=z_500.coords,
                           attrs={'standard_name': 'geopotential height',
                                  'units': gph_500.units})

    levels_temp = np.arange(-30., 30., 1.)
    levels_gph = np.arange(4700., 6000., 70.)

    cmap = get_colormap('temp')

    fig = plt.figure(figsize=(figsize_x, figsize_y))

    ax = plt.gca()

    lon, lat = get_coordinates(temp_850)
    lon2d, lat2d = np.meshgrid(lon, lat)

    ax = get_projection_cartopy(plt, projection, compute_projection=True)

    if projection == 'euratl':
        norm = BoundaryNorm(levels_temp, ncolors=cmap.N)
        cs = ax.pcolormesh(lon2d, lat2d, temp_850, cmap=cmap, norm=norm)
    else:
        cs = ax.contourf(lon2d, lat2d, temp_850, extend='both',
                         cmap=cmap, levels=levels_temp)

    c = ax.contour(lon2d, lat2d, gph_500, levels=levels_gph,
                   colors='white', linewidths=1.)

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

    plt.savefig(filename, **options_savefig)
    plt.clf()

    return filename


def plot_vars(f_step, projection, load_all=False):
    # The one employed for the figure name when exported
    variable_name = 'gph_t_850'
    # Build the name of the output image
    run_string, _ = get_run()

    if load_all:
        f_steps = list(range(0, 79)) + list(range(81, 121, 3))
    else:
        f_steps = [f_step]

    filenames = ['/tmp/' + projection + '_' + variable_name +
                 '_%s_%03d.png' % (run_string, f_step) for f_step in f_steps]
    test_filenames = [os.path.exists(f) for f in filenames]

    if all(test_filenames):  # means the files already exist
        return filenames

    # otherwise do the plots
    dset = get_dset(vars_3d=['t@850', 'fi@500'], f_times=f_steps).squeeze()
    # Add a fictictious 1-D time dimension just to avoid problems
    if 'step' not in dset.dims.keys():
        dset = dset.expand_dims('step')
    #
    dset = subset_arrays(dset, projection)
    time = pd.to_datetime(dset.valid_time.values)
    cum_hour = dset.step.values.astype(int)

    temp_850 = dset['t'] - 273.15
    z_500 = dset['z']
    gph_500 = mpcalc.geopotential_to_height(z_500)
    gph_500 = xr.DataArray(gph_500.magnitude, coords=z_500.coords,
                           attrs={'standard_name': 'geopotential height',
                                  'units': gph_500.units})

    levels_temp = np.arange(-30., 30., 1.)
    levels_gph = np.arange(4700., 6000., 70.)

    lon, lat = get_coordinates(temp_850)
    lon2d, lat2d = np.meshgrid(lon, lat)

    cmap = get_colormap('temp')

    args = dict(filenames=filenames, projection=projection, levels_temp=levels_temp,
                cmap=cmap, lon2d=lon2d, lat2d=lat2d, lon=lon, lat=lat, temp_850=temp_850.values,
                gph_500=gph_500.values, levels_gph=levels_gph, time=time, run_string=run_string)

    if load_all:
        single_plot_param = partial(single_plot, **args)
        iterator = range(0, len(f_steps))
        pool = Pool(cpu_count())
        results = pool.map(single_plot_param, iterator)
        pool.close()
        pool.join()
    else:
        results = single_plot(0, **args)

    return results


def single_plot(it, **args):
    filename = args['filenames'][it]

    if os.path.exists(filename):
        return filename

    fig = plt.figure(figsize=(figsize_x, figsize_y))
    ax = plt.gca()
    ax = get_projection_cartopy(
        plt, args['projection'], compute_projection=True)

    if args['projection'] == 'euratl':
        norm = BoundaryNorm(args['levels_temp'], ncolors=args['cmap'].N)
        cs = ax.pcolormesh(args['lon2d'], args['lat2d'], args['temp_850'][it],
                           cmap=args['cmap'], norm=norm)
    else:
        cs = ax.contourf(args['lon2d'], args['lat2d'], args['temp_850'][it], extend='both',
                         cmap=args['cmap'], levels=args['levels_temp'])

    c = ax.contour(args['lon2d'], args['lat2d'], args['gph_500'][it],
                   levels=args['levels_gph'], colors='white', linewidths=1.)

    labels = ax.clabel(c, c.levels, inline=True, fmt='%4.0f', fontsize=6)

    maxlabels = plot_maxmin_points(ax, args['lon'], args['lat'], args['gph_500'][it],
                                   'max', 80, symbol='H', color='royalblue', random=True)
    minlabels = plot_maxmin_points(ax, args['lon'], args['lat'], args['gph_500'][it],
                                   'min', 80, symbol='L', color='coral', random=True)

    try:
        an_fc = annotation_forecast(ax, args['time'][it])
    except TypeError:
        an_fc = annotation_forecast(ax, args['time'])

    an_var = annotation(
        ax, 'Geopotential height @500hPa [m] and temperature @850hPa [C]', loc='lower left', fontsize=6)
    an_run = annotation(
        ax, 'Run: ' + args['run_string'] + ' UTC', loc='upper right')

    plt.colorbar(cs, orientation='horizontal',
                 label='Temperature', pad=0.03, fraction=0.04)

    plt.savefig(filename, **options_savefig)
    plt.clf()

    return(filename)
