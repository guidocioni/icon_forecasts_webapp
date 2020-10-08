import numpy as np
from matplotlib.offsetbox import AnchoredText
import matplotlib.colors as colors
import pandas as pd
from matplotlib.colors import from_levels_and_colors
import seaborn as sns
import os
import matplotlib.patheffects as path_effects
import sys
import xarray as xr
# import metpy
from datetime import datetime
import requests
import bz2
from multiprocessing import Pool, cpu_count
import base64
import cfgrib


import warnings
warnings.filterwarnings(
    action='ignore',
    message='The unit of the quantity is stripped.'
)

os.environ['HOME_FOLDER'] = os.getcwd()
figsize_x = 10 
figsize_y = 8

# Options for savefig
options_savefig = {
    'dpi':100,
    'bbox_inches':'tight'
}

proj_defs = {
    'euratl':
    {
        'extents':[-23.5, 45, 29.5, 70.5],
        'resolution': '50m',
        'regions':False
    },
    'it':
    {
        'extents':[6, 19, 36, 48],
        'resolution': '10m',
        'regions':True
    },
    'de':
    {
        'extents':[5, 16, 46.5, 56],
        'resolution': '10m',
        'regions':True
    }
}

var_2d_list = ['alb_rad','alhfl_s','ashfl_s','asob_s','asob_t','aswdifd_s','aswdifu_s',
          'aswdir_s','athb_s','cape_con','cape_ml','clch','clcl','clcm','clct',
          'clct_mod','cldepth','h_snow','hbas_con','htop_con','htop_dc','hzerocl',
          'pmsl','ps','qv_2m','qv_s','rain_con','rain_gsp','relhum_2m','rho_snow',
          'runoff_g','runoff_s','snow_con','snow_gsp','snowlmt','synmsg_bt_cl_ir10.8',
          't_2m','t_g','t_snow','tch','tcm','td_2m','tmax_2m','tmin_2m','tot_prec',
          'u_10m','v_10m','vmax_10m','w_snow','w_so','ww','z0']

var_3d_list = ['clc','fi','omega','p','qv','relhum','t','tke','u','v','w']

def get_run():
    now = datetime.now()
    date_string = now.strftime('%Y%m%d')
    utc_now = datetime.utcnow()
    
    if (utc_now.replace(hour=4, minute=0, second=0, microsecond=0) 
        <= utc_now < utc_now.replace(hour=9, minute=0, second=0, microsecond=0)):
        run="00"
    elif (utc_now.replace(hour=9, minute=0, second=0, microsecond=0) 
        <= utc_now < utc_now.replace(hour=16, minute=0, second=0, microsecond=0)):
        run="06"
    elif (utc_now.replace(hour=16, minute=0, second=0, microsecond=0) 
        <= utc_now < utc_now.replace(hour=21, minute=0, second=0, microsecond=0)):
        run="12"
    elif (utc_now.replace(hour=21, minute=0, second=0, microsecond=0) 
        <= utc_now):
        run="18"
    
    return now.strftime('%Y%m%d')+run, run

def find_file_name(vars_2d=None,
                   vars_3d=None,
                   f_times=0, 
                   base_url = "https://opendata.dwd.de/weather/nwp",
                   model_url = "icon-eu/grib"):
    '''Find file names to be downloaded given input variables and
    a forecast lead time f_time (in hours).
    - vars_2d, a list of 2d variables to download, e.g. ['t_2m']
    - vars_3d, a list of 3d variables to download with pressure
      level, e.g. ['t@850','fi@500']
    - f_times, forecast steps, e.g. 0 or list(np.arange(1, 79))
    Note that this function WILL NOT check if the files exist on
    the server to avoid wasting time. When they're passed
    to the download_extract_files function if the file does not
    exist it will simply not be downloaded.
      '''
    date_string, run_string = get_run()
    if type(f_times) is not list:
        f_times = [f_times]
    if (vars_2d is None) and (vars_3d is None):
        raise ValueError('You need to specify at least one 2D or one 3D variable')

    if vars_2d is not None:
        if type(vars_2d) is not list:
            vars_2d = [vars_2d]
    if vars_3d is not None:
        if type(vars_3d) is not list:
            vars_3d = [vars_3d]

    urls = []
    for f_time in f_times:
        if vars_2d is not None:
            for var in vars_2d:
                if var not in var_2d_list:
                    raise ValueError('accepted 2d variables are %s' % var_2d_list)
                var_url="icon-eu_europe_regular-lat-lon_single-level"
                urls.append("%s/%s/%s/%s/%s_%s_%03d_%s.grib2.bz2" % 
                            (base_url, model_url, run_string, var,
                              var_url, date_string, f_time, var.upper()) )
        if vars_3d is not None:
            for var in vars_3d:
                var_t, plev = var.split('@')
                if var_t not in var_3d_list:
                    raise ValueError('accepted 3d variables are %s' % var_3d_list)
                var_url="icon-eu_europe_regular-lat-lon_pressure-level"
                urls.append("%s/%s/%s/%s/%s_%s_%03d_%s_%s.grib2.bz2" % 
                            (base_url, model_url, run_string, var_t,
                              var_url, date_string, f_time, plev, var_t.upper()) )

    return urls

def download_extract_files(urls):
    '''Given a list of urls download and bunzip2 them.
    Return a list of the path of the extracted files'''

    if type(urls) is list:
        urls_list = urls
    else:
        urls_list = [urls]

    # We only parallelize if we have a number of files
    # larger than the cpu count 
    if len(urls_list) > cpu_count():    
        pool = Pool(cpu_count())
        results = pool.map(download_extract_url, urls_list)
        pool.close()
        pool.join()
    else:
        results = []
        for url in urls_list:
            results.append(download_extract_url(url))

    return results

def download_extract_url(url):
    folder = '/tmp/'
    filename = folder+os.path.basename(url).replace('.bz2','')

    if os.path.exists(filename):
        extracted_files = filename
    else:
        r = requests.get(url, stream=True)
        if r.status_code == requests.codes.ok:
            with r.raw as source, open(filename, 'wb') as dest:
                dest.write(bz2.decompress(source.read()))
            extracted_files = filename
        else:
            r.raise_for_status()

    return extracted_files

def get_dset(vars_2d=[], vars_3d=[], f_times=0):
    if vars_2d or vars_3d:
        date_string, _ = get_run()
        urls = find_file_name(vars_2d=vars_2d,
                              vars_3d=vars_3d,
                              f_times=f_times)
        fils = download_extract_files(urls)
        # We cat the files on Linux and read the resulting grib, this is much
        # much faster!! But it will not work everywhere 
        if (type(fils) is list and len(fils) > 3): # multiple files extractor
            merged_file = '/tmp/'+date_string + '_' + '_'.join(vars_3d+vars_2d) +'.grib2'
            
            if os.path.exists(merged_file) == False:
                os.system('cat %s > %s' % (' '.join(fils), merged_file) )
            
            dss = cfgrib.open_datasets(merged_file)

            for i,_ in enumerate(dss):
                dss[i] = preprocess(dss[i])
            ds = xr.merge(dss)

        else:
            ds = xr.open_mfdataset(fils, engine='cfgrib', preprocess=preprocess,
                  combine="by_coords", concat_dim='step', parallel=False)

    return ds

def preprocess(ds):
    if 'isobaricInhPa' in ds.coords.keys():
        ds = ds.drop('isobaricInhPa')
    if 'heightAboveGround' in ds.coords.keys():
        ds = ds.drop('heightAboveGround')
    # Use valid_time as coordinate so that we can 
    # use this to concatenate afterwards
    if 'step' not in ds.dims.keys():
        ds = ds.expand_dims(dim='step')

    return ds

def read_time(dset):
    """Read time properly (as datetime object) from dataset
    and compute forecast lead time as cumulative hour"""
    time = pd.to_datetime(dset.valid_time.values)
    if len(time) > 1:
        cum_hour = np.array((time - time[0]) /
                            pd.Timedelta('1 hour')).astype("int")
    else:
        cum_hour = 0

    return time, cum_hour

def get_weather_icons(ww, time):
    #from matplotlib._png import read_png
    from matplotlib.image import imread as read_png
    """
    Get the path to a png given the weather representation 
    """
    weather = [WMO_GLYPH_LOOKUP_PNG[w.astype(int).astype(str)] for w in ww.values]
    weather_icons=[]
    for date, weath in zip(time, weather):
        if date.hour >= 6 and date.hour <= 18:
            add_string='d'
        elif date.hour >=0 and date.hour < 6:
            add_string='n'
        elif date.hour >18 and date.hour < 24:
            add_string='n'

        pngfile=folder_glyph+'%s.png' % (weath+add_string)
        if os.path.isfile(pngfile):
            weather_icons.append(read_png(pngfile))
        else:
            pngfile=folder_glyph+'%s.png' % weath
            weather_icons.append(read_png(pngfile))

    return(weather_icons)

def subset_arrays(arrs, proj):
    """Given an input projection created with basemap or cartopy subset the input arrays 
    on the boundaries"""
    proj_options = proj_defs[proj]
    if type(arrs) is list:
        out = []
        for arr in arrs:
            out.append(arr.sel(latitude=slice(proj_options['extents'][2], proj_options['extents'][3]),
                                longitude=slice(proj_options['extents'][0], proj_options['extents'][1])))
    else:
        out = arrs.sel(latitude=slice(proj_options['extents'][2], proj_options['extents'][3]),
                                longitude=slice(proj_options['extents'][0], proj_options['extents'][1]))

    return out

def read_time(dset):
    """Read time properly (as datetime object) from dataset
    and compute forecast lead time as cumulative hour"""
    time = pd.to_datetime(dset.time.values)
    cum_hour = np.array((time - time[0]) /
                        pd.Timedelta('1 hour')).astype("int")

    return time, cum_hour

def print_message(message):
    """Formatted print"""
    print(os.path.basename(sys.argv[0])+' : '+message)


def get_coordinates(ds):
    """Get the lat/lon coordinates from the ds and convert them to degrees.
    Usually this is only used to prepare the plotting."""
    if ('lat' in ds.coords.keys()) and ('lon' in ds.coords.keys()):
        longitude = ds['lon']
        latitude = ds['lat']
    elif ('latitude' in ds.coords.keys()) and ('longitude' in ds.coords.keys()):
        longitude = ds['longitude']
        latitude = ds['latitude']
    elif ('lat2d' in ds.coords.keys()) and ('lon2d' in ds.coords.keys()):
        longitude = ds['lon2d']
        latitude = ds['lat2d']

    if longitude.max() > 180:
        longitude = (((longitude.lon + 180) % 360) - 180)

    return(longitude.values, latitude.values)


def get_city_coordinates(city):
    """Get the lat/lon coordinates of a city given its name using geopy."""
    from geopy.geocoders import Nominatim
    geolocator =Nominatim(user_agent='meteogram')
    loc = geolocator.geocode(city)
    return(loc.longitude, loc.latitude)

def get_projection_cartopy(plt, projection="euratl", compute_projection=False):
    '''Retrieve the projection using cartopy'''
    if compute_projection:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import cartopy.io.shapereader as shpreader

        proj_opts = proj_defs[projection]

        ax = plt.axes(projection=ccrs.PlateCarree())

        ax.set_extent(proj_opts['extents'], ccrs.PlateCarree())
        ax.coastlines(resolution=proj_opts['resolution'])
        ax.add_feature(cfeature.BORDERS.with_scale(proj_opts['resolution']))

        if proj_opts['regions']:
            states_provinces = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_1_states_provinces_lines',
                scale=proj_opts['resolution'],
                facecolor='none')
            ax.add_feature(states_provinces, edgecolor='black', alpha=.5)

        return(ax)
    else:
        return(add_background(plt, projection, image=projection+"_background.png"))

def add_background(plt, projection, image):
    ''''Add a background image to the plot'''
    proj_opts = proj_defs[projection]
    extents = proj_opts['extents']

    img = plt.imread(image)
    plt.axis('off')
    plt.imshow(img, zorder=10, extent=extents)

    return plt.gca()

def b64_image(image_filename): 
    with open(image_filename, 'rb') as f: 
        image = f.read() 
        return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

# Annotation run, models 
def annotation_run(ax, time, loc='upper right',fontsize=8):
    """Put annotation of the run obtaining it from the
    time array passed to the function."""
    at = AnchoredText('Run %s'% time.strftime('%Y%m%d %H UTC'), 
                       prop=dict(size=fontsize), frameon=True, loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.1")
    at.zorder = 10
    ax.add_artist(at)
    return(at)


def annotation_forecast(ax, time, loc='upper left',fontsize=8, local=True):
    """Put annotation of the forecast time."""
    if local: # convert to local time
        time = convert_timezone(time)
        at = AnchoredText('Valid %s' % time.strftime('%A %d %b %Y at %H (Berlin)'), 
                       prop=dict(size=fontsize), frameon=True, loc=loc)
    else:
        at = AnchoredText('Forecast for %s' % time.strftime('%A %d %b %Y at %H UTC'), 
                       prop=dict(size=fontsize), frameon=True, loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.1")
    at.zorder = 10
    ax.add_artist(at)
    return(at) 


def convert_timezone(dt_from, from_tz='utc', to_tz='Europe/Berlin'):
    """Convert between two timezones. dt_from needs to be a Timestamp 
    object, don't know if it works otherwise."""
    dt_to = dt_from.tz_localize(from_tz).tz_convert(to_tz)
    # remove again the timezone information
    return dt_to.tz_localize(None)   


def annotation(ax, text, loc='upper right',fontsize=8):
    """Put a general annotation in the plot."""
    at = AnchoredText('%s'% text, prop=dict(size=fontsize), frameon=True, loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.1")
    at.zorder = 10
    ax.add_artist(at)
    return(at)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """Truncate a colormap by specifying the start and endpoint."""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return(new_cmap)


def get_colormap(cmap_type):
    """Create a custom colormap."""
    colors_tuple = pd.read_csv(os.environ['HOME_FOLDER'] + '/cmap_%s.rgba' % cmap_type).values 
    
    cmap = colors.LinearSegmentedColormap.from_list(cmap_type, colors_tuple, colors_tuple.shape[0])
    return(cmap)


def get_colormap_norm(cmap_type, levels):
    """Create a custom colormap."""
    if cmap_type == "rain":
        cmap, norm = from_levels_and_colors(levels, sns.color_palette("Blues", n_colors=len(levels)),
                                                    extend='max')
    elif cmap_type == "snow":
        cmap, norm = from_levels_and_colors(levels, sns.color_palette("PuRd", n_colors=len(levels)),
                                                    extend='max')
    elif cmap_type == "snow_discrete":    
        colors = ["#DBF069","#5AE463","#E3BE45","#65F8CA","#32B8EB",
                    "#1D64DE","#E97BE4","#F4F476","#E78340","#D73782","#702072"]
        cmap, norm = from_levels_and_colors(levels, colors, extend='max')
    elif cmap_type == "rain_acc":    
        cmap, norm = from_levels_and_colors(levels, sns.color_palette('gist_stern_r', n_colors=len(levels)),
                         extend='max')
    elif cmap_type == "rain_new":
        colors_tuple = pd.read_csv(os.environ['HOME_FOLDER'] + '/cmap_prec.rgba').values    
        cmap, norm = from_levels_and_colors(levels, sns.color_palette(colors_tuple, n_colors=len(levels)),
                         extend='max')
    elif cmap_type == "winds":
        colors_tuple = pd.read_csv(os.environ['HOME_FOLDER'] + '/cmap_winds.rgba').values    
        cmap, norm = from_levels_and_colors(levels, sns.color_palette(colors_tuple, n_colors=len(levels)),
                         extend='max')

    return(cmap, norm)


# def remove_collections(elements):
#     """Remove the collections of an artist to clear the plot without
#     touching the background, which can then be used afterwards."""
#     for element in elements:
#         try:
#             for coll in element.collections: 
#                 coll.remove()
#         except AttributeError:
#             try:
#                 for coll in element:
#                     coll.remove()
#             except ValueError:
#                 print_message('WARNING: Element is empty')
#             except TypeError:
#                 element.remove()
#         except ValueError:
#             print_message('WARNING: Collection is empty')


def plot_maxmin_points(ax, lon, lat, data, extrema, nsize, symbol, color='k',
                       random=False):
    """
    This function will find and plot relative maximum and minimum for a 2D grid. The function
    can be used to plot an H for maximum values (e.g., High pressure) and an L for minimum
    values (e.g., low pressue). It is best to used filetered data to obtain  a synoptic scale
    max/min value. The symbol text can be set to a string value and optionally the color of the
    symbol and any plotted value can be set with the parameter color
    lon = plotting longitude values (2D)
    lat = plotting latitude values (2D)
    data = 2D data that you wish to plot the max/min symbol placement
    extrema = Either a value of max for Maximum Values or min for Minimum Values
    nsize = Size of the grid box to filter the max and min values to plot a reasonable number
    symbol = String to be placed at location of max/min value
    color = String matplotlib colorname to plot the symbol (and numerica value, if plotted)
    plot_value = Boolean (True/False) of whether to plot the numeric value of max/min point
    The max/min symbol will be plotted on the current axes within the bounding frame
    (e.g., clip_on=True)
    """
    from scipy.ndimage.filters import maximum_filter, minimum_filter

    # We have to first add some random noise to the field, otherwise it will find many maxima
    # close to each other. This is not the best solution, though...
    if random:
        data = np.random.normal(data, 0.2)

    if len(lon.shape) == 1 and len(lat.shape) == 1:
        lon, lat = np.meshgrid(lon, lat)

    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')

    mxy, mxx = np.where(data_ext == data)
    # Filter out points on the border 
    mxx, mxy = mxx[(mxy != 0) & (mxx != 0)], mxy[(mxy != 0) & (mxx != 0)]

    texts = []
    for i in range(len(mxy)):
        texts.append( ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]], symbol, color=color, size=15,
                clip_on=True, horizontalalignment='center', verticalalignment='center',
                path_effects=[path_effects.withStroke(linewidth=1, foreground="black")], zorder=6) )
        texts.append( ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]], '\n' + str(data[mxy[i], mxx[i]].astype('int')),
                color="gray", size=10, clip_on=True, fontweight='bold',
                horizontalalignment='center', verticalalignment='top', zorder=6) )

    return(texts)


# def add_vals_on_map(ax, bmap, var, levels, density=50,
#                      cmap='rainbow', shift_x=0., shift_y=0., fontsize=8, lcolors=True):
#     '''Given an input projection, a variable containing the values and a plot put
#     the values on a map exlcuing NaNs and taking care of not going
#     outside of the map boundaries, which can happen.
#     - shift_x and shift_y apply a shifting offset to all text labels
#     - colors indicate whether the colorscale cmap should be used to map the values of the array'''

#     norm = colors.Normalize(vmin=levels.min(), vmax=levels.max())
#     m = mplcm.ScalarMappable(norm=norm, cmap=cmap)
    
#     lon_min, lon_max, lat_min, lat_max = bmap.llcrnrlon, bmap.urcrnrlon, bmap.llcrnrlat, bmap.urcrnrlat

#     # Remove values outside of the extents
#     var = var.sel(lat=slice(lat_min+0.15, lat_max-0.15), lon=slice(lon_min+0.15, lon_max-0.15))[::density, ::density]
#     lons = var.lon
#     lats = var.lat

#     at = []
#     for ilat, ilon in np.ndindex(var.shape):
#         if lcolors:
#             at.append(ax.annotate(('%d'%var[ilat, ilon]), (lons[ilon]+shift_x, lats[ilat]+shift_y),
#                              color = m.to_rgba(float(var[ilat, ilon])), weight='bold', fontsize=fontsize,
#                               path_effects=[path_effects.withStroke(linewidth=1, foreground="black")], zorder=5))
#         else:
#             at.append(ax.annotate(('%d'%var[ilat, ilon]), (lons[i]+shift_x, lats[i]+shift_y),
#                              color = 'white', weight='bold', fontsize=fontsize,
#                               path_effects=[path_effects.withStroke(linewidth=1, foreground="black")], zorder=5))

#     return at
