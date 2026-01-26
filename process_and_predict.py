import glob
import argparse
import pathlib
import pygrib
import pygridder as pg
import pickle

import numpy as np
import pandas as pd

import pyproj
import datetime as dt
import glob
import os

import warnings
warnings.filterwarnings('ignore')

import utils as u
import fipsvars as fv
from make_plots import make_plots

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--otlkfile", required=True)
parser.add_argument("-c", "--confile", required=True)
parser.add_argument("-out", "--outpath", required=False, default='../web/images/')
parser.add_argument("-ml", "--mlpath", required=False, default='../../ml-data/trained-models/')
parser.add_argument("-d", "--datapath", required=False, default='../data/')
parser.add_argument("-hp", "--hrefpath", required=False, default='../test-files/href-calthunder/')
parser.add_argument("-t", "--test", default=0, type=int, required=False)
parser.add_argument("-ht", "--hazard", required=False, type=str, default='hail')

args = parser.parse_args()

### Parse CLI Arguments ###
ndfd_file = pathlib.Path(args.otlkfile)
con_file = pathlib.Path(args.confile)
ml_path = pathlib.Path(args.mlpath)
data_path = pathlib.Path(args.datapath)
href_path = pathlib.Path(args.hrefpath)
out_path = pathlib.Path(args.outpath)
haz_type = args.hazard

isTest = bool(args.test)

# Set proper permission structure
os.umask(0o022)

# Data files
impacts_grids_file = f'{data_path}/impact-grids-5km.npz'
cwa_file = f'{data_path}/cwas.npz'

# ML models
with open(f'{ml_path.as_posix()}/wfo-label-encoder.model','rb') as f:
    wfo_label_encoder = pickle.load(f)

with open(f'{ml_path.as_posix()}/det/hgb-det_wind.model','rb') as f:
    wind_model = pickle.load(f)

with open(f'{ml_path.as_posix()}/det/hgb-det_hail.model','rb') as f:
    hail_model = pickle.load(f)

models = {
    'le': wfo_label_encoder,
    'wind': wind_model,
    'hail': hail_model
}

otlk_ts = ndfd_file.as_posix().split('_')[-1]
year = int(otlk_ts[:4])
month = int(otlk_ts[4:6])
day = int(otlk_ts[6:8])
valid_hr = int(ndfd_file.as_posix().split('_')[-2][:2])
otlkdt = dt.datetime(year, month, day, valid_hr)

with np.load(impacts_grids_file) as NPZ:
    population = NPZ["population"]
    proj = pyproj.Proj(NPZ["srs"].item())
    geod = pyproj.Geod(f'{proj} +a=6371200 +b=6371200')
    lons = NPZ["lons"]
    lats = NPZ["lats"]
    X = NPZ["X"]
    Y = NPZ["Y"]
    dx = NPZ["dx"]
    dy = NPZ["dy"]
    state = NPZ["state"]
    county = NPZ["county"]

with np.load(cwa_file) as NPZ:
    wfo = NPZ['cwas']

map_func = np.vectorize(lambda x: fv.fipsToState.get(x, '0'))

### Outlook feature processing
# Read grib file
def read_ndfd_grib_file(grbfile,which='torn'):
    """ Read an SPC Outlook NDFD Grib2 File """
    if which == 'torn':
        with pygrib.open(grbfile.as_posix()) as GRB:
            try:
                vals = GRB[1].values.filled(-1)
            except AttributeError:
                vals = GRB[1].values
        return vals
    else:
        with pygrib.open(grbfile.as_posix().replace('torn',which)) as GRB:
            try:
                vals = GRB[1].values.filled(-1)
            except AttributeError:
                vals = GRB[1].values
        return vals
    
def read_con_npz_file(npzfile,which='torn'):
    # Read continuous cig file
    if which == 'torn':
        with np.load(npzfile.as_posix()) as NPZ:
            vals = NPZ['vals']
    else:
        with np.load(npzfile.as_posix().replace('torn',which)) as NPZ:
            vals = NPZ['vals']

    vals[vals < 1] = 0
    vals[np.logical_and(vals >= 1, vals < 2)] = 1
    vals[vals >= 2] = 2

    return vals

# Needed to convert toy_sin and toy_cos back to day of year
def get_doy(X):
    angle = np.arctan2(X.toy_sin[0],X.toy_cos[0])

    # Normalize to [0, 1)
    normalized_doy = (angle % (2 * np.pi)) / (2 * np.pi)

    # Convert to day of year (1–366)
    day_of_year = int(round(normalized_doy * 366))

    return day_of_year

# read coverage files
torn_cov = read_ndfd_grib_file(ndfd_file, which='torn')
hail_cov = read_ndfd_grib_file(ndfd_file, which='hail')
wind_cov = read_ndfd_grib_file(ndfd_file, which='wind')

# read cig files
torn_con = read_con_npz_file(con_file, which='torn')
hail_con = read_con_npz_file(con_file, which='hail')
wind_con = read_con_npz_file(con_file, which='wind')

# get wfo-states in outlooks
wfo_windoutlook = np.unique(wfo[wind_cov > 0])
wfo_hailoutlook = np.unique(wfo[hail_cov > 0])
wfo_tornoutlook = np.unique(wfo[torn_cov > 0])

wfo_alloutlook = np.concatenate([wfo_windoutlook,wfo_hailoutlook,wfo_tornoutlook])
wfo_unique = np.unique(wfo_alloutlook[[ '0' not in s for s in wfo_alloutlook]])

# Outlook based features
df_otlk = u.gather_features(wfo_unique, wfo, otlkdt, 
                    wind_cov, hail_cov, torn_cov, 
                    wind_con, hail_con, torn_con, 
                    population)
df_otlk.columns = fv.col_names_outlook  # Rename columns to assist merge with HREF features

# HREF feature processing
# if isTest:
ct_files = glob.glob(f'{href_path.as_posix()}/{otlkdt.year}{otlkdt.month:02d}{otlkdt.day:02d}/thunder/spc_post.t12z.hrefct.1hr.f*')
# else:
#     ct_files = glob.glob(f'/nfsops/ops_users/nadata2/awips2/grib2/spcpost/{otlkdt.year}{otlkdt.month:02d}{otlkdt.day:02d}/thunder/spc_post.t12z.hrefct.1hr.f*')

if len(ct_files) != 48:
    print("12Z HREF files incomplete, checking 00Z files...")
    # if isTest:
    ct_files = glob.glob(f'{href_path.as_posix()}/{otlkdt.year}{otlkdt.month:02d}{otlkdt.day:02d}/thunder/spc_post.t00z.hrefct.1hr.f*')
    # else:
    #     ct_files = glob.glob(f'/nfsops/ops_users/nadata2/awips2/grib2/spcpost/{otlkdt.year}{otlkdt.month:02d}{otlkdt.day:02d}/thunder/spc_post.t00z.hrefct.1hr.f*')
    
    which_href = 0
    
    if len(ct_files) != 48:
        import sys
        print("Some 00Z HREF files missing as well, exiting...")
        sys.exit(0)
else:
    which_href = 12

ct_files.sort()
ct_arrs = []

# Need to slice appropriate HREF files, based on initialization time
if which_href == 0:
    sliced_ct_files = ct_files[otlkdt.hour-1:35]
else:
    # Quick fix for running archive of 12z outlooks
    if otlkdt.hour == 12:
        sliced_ct_files = ct_files[0:23]
    else:
        sliced_ct_files = ct_files[otlkdt.hour-13:23]

for i,file in enumerate(sliced_ct_files):

    with pygrib.open(file) as GRB:
        
        try:
            vals = GRB.values.filled(-1)
            lats_href, lons_href = GRB.latlons()
        except AttributeError:
            vals = GRB[1].values
            lats_href, lons_href = GRB[1].latlons()

    ct_arrs.append(vals)

val_stack = np.stack(ct_arrs,axis=0)

# get all ndfd grid boxes associated with wfo-st's touched by an outlook
wfo_impacted = np.isin(wfo,wfo_unique)

# get 1d list of wfo-st for each ndfd grid box
wfo_1d = wfo[wfo_impacted]

G_href = pg.Gridder(lons_href, lats_href)

# find the href grid indices for these ndfd grids
try:
    idx_href = G_href.grid_points(lons[wfo_impacted],lats[wfo_impacted])
except ValueError:
    import sys
    print("No severe hazard probabilities. Exiting...")
    sys.exit(0)

# Get all HREFCT probabilities for all times in each CWA-ST
ct_vals = []
for idx in idx_href:
    ct_vals.append(val_stack[:,idx[0],idx[1]])

ct_vals = np.array(ct_vals)

df_href = u.gather_ct_features(wfo_unique, wfo_1d, otlkdt, ct_vals, population, wfo)
df_href.columns = fv.col_names_href # Rename columns to assist merge with outlook features

df_all = df_otlk.merge(df_href, on=['otlk_timestamp','wfo_list'], how='inner').fillna(0)

wfo_list_trans = models['le'].transform(df_all.wfo_list.str.slice(0,3))
X = df_all.iloc[:,2:]
X['doy'] = get_doy(X)
X['wfo'] = wfo_list_trans

mask_wfo = np.char.find(wfo, '0') != -1
mask_st = state == 0

mask = np.logical_or(mask_wfo, mask_st)

if haz_type == 'hail':
    if hail_cov.max() == 0:
        print(f'All hail coverage probabilities less than 5%. No hail predictions made for {otlk_ts}.')
        import sys
        sys.exit(0)

    prob_holder = hail_cov

    hail_preds = np.round(models['hail'].predict(X))
    hail_preds = u.fix_neg(hail_preds)
    
    df_preds = pd.DataFrame({
        'wfo': df_all.wfo_list,
        'hail': hail_preds
    })

    hail_bool = np.array([int(bool((hail_cov[wfo == place]).sum())) for place in df_preds.wfo])
    df_preds['hail'] = df_preds['hail']*hail_bool

    nat_preds = df_preds[['hail']].sum()

    if nat_preds.values[0] == 0:
        # Use a simple percentage of sims equaling 1 (from historical data in condint era)
        nat_hail_dist = np.random.choice([0,1], size=fv.nsims,replace=True, p=[fv.zero_pct_hail, 1-fv.zero_pct_hail])
    else:
        # Create distribution with negative binomial
        nat_hail_dist = np.random.negative_binomial(fv.alpha_hail, fv.alpha_hail/(fv.alpha_hail + nat_preds.values[0]), size=fv.nsims)

    # Create weight grids to place reports
    hail_cov[hail_cov < 0] = 0

    # continuous coverage files
    hail_cov_con = u.make_continuous(hail_cov, haz='hail')

    rel_freq_hail = np.zeros(wfo.shape)

    for _,row in df_preds.iterrows():
        area = row.wfo
        hail_pred = row.hail
        
        rel_freq_hail[wfo == area] = hail_pred

    if rel_freq_hail.max() == 0:
        rel_freq_hail_norm = rel_freq_hail
    else:
        rel_freq_hail_norm = 100*rel_freq_hail / rel_freq_hail.max()

    hail_cov_norm = 100*hail_cov / hail_cov.max()

    if not isTest:
        hail_weight = rel_freq_hail_norm + hail_cov_norm
    else:
        hail_weight = rel_freq_hail_norm
    hail_weight[hail_cov_norm == 0] = 0
    hail_weight[mask] = 0

    cumulative_weights_hail = hail_weight.cumsum().astype(int)
    all_reps_hail_sum = nat_hail_dist.sum()

    # Get hail locations and sig count numbers
    _locs = np.random.randint(
                0.001, cumulative_weights_hail.max(), size=all_reps_hail_sum)
    locs = cumulative_weights_hail.searchsorted(_locs)

    con_reports = hail_con.flatten()[locs]
    wfo_hail = wfo.flatten()[locs]
    state_hail = state.flatten()[locs]
    state_hail = np.array([fv.fipsToState[st] for st in state_hail])

    all_ratings = []

    hail_df = pd.DataFrame({'cig': con_reports.astype(int), 
                            'wfo': wfo_hail,
                            'st': state_hail})

    group_ids = np.repeat(np.arange(1, len(nat_hail_dist)+1), nat_hail_dist)

    hail_df['sim'] = group_ids
    if hail_df.empty:
        hail_df['sig'] = None
    else:
        hail_df['sig'] = hail_df.apply(lambda row: u.return_mag(row.cig, month, row.wfo, 'hail'),axis=1)

    reports_df = hail_df

elif haz_type == 'wind':  
    if wind_cov.max() == 0:
        print(f'All wind coverage probabilities less than 5%. No wind predictions made for {otlk_ts}.')
        import sys
        sys.exit(0)

    prob_holder = wind_cov

    wind_preds = np.round(models['wind'].predict(X))
    wind_preds = u.fix_neg(wind_preds)

    df_preds = pd.DataFrame({
        'wfo': df_all.wfo_list,
        'wind': wind_preds
    })

    wind_bool = np.array([int(bool((wind_cov[wfo == place]).sum())) for place in df_preds.wfo])
    df_preds['wind'] = df_preds['wind']*wind_bool

    nat_preds = df_preds[['wind']].sum()

    if nat_preds.values[0] == 0:
        # Use a simple percentage of sims equaling 1 (from historical data in condint era)
        nat_wind_dist = np.random.choice([0,1], size=fv.nsims,replace=True, p=[fv.zero_pct_wind, 1-fv.zero_pct_wind])
    else:
        # Create distribution with negative binomial
        nat_wind_dist = np.random.negative_binomial(fv.alpha_wind, fv.alpha_wind/(fv.alpha_wind + nat_preds.values[0]), size=fv.nsims)

    # Create weight grids to place reports
    wind_cov[wind_cov < 0] = 0

    # continuous coverage files
    wind_cov_con = u.make_continuous(wind_cov, haz='wind')

    rel_freq_wind = np.zeros(wfo.shape)

    for _,row in df_preds.iterrows():
        area = row.wfo
        wind_pred = row.wind
        
        rel_freq_wind[wfo == area] = wind_pred

    if rel_freq_wind.max() == 0:
        rel_freq_wind_norm = rel_freq_wind
    else:
        rel_freq_wind_norm = 100*rel_freq_wind / rel_freq_wind.max()

    wind_cov_norm = 100*wind_cov / wind_cov.max()

    if not isTest:
        wind_weight = wind_cov_norm + rel_freq_wind_norm
    else:
        wind_weight = rel_freq_wind_norm
    wind_weight[wind_cov_norm == 0] = 0
    wind_weight[mask] = 0

    cumulative_weights_wind = wind_weight.cumsum().astype(int)
    all_reps_wind_sum = nat_wind_dist.sum()

    # Get wind locations and sig count numbers
    _locs = np.random.randint(
                0.001, cumulative_weights_wind.max(), size=all_reps_wind_sum)
    locs = cumulative_weights_wind.searchsorted(_locs)

    con_reports = wind_con.flatten()[locs]
    wfo_wind = wfo.flatten()[locs]
    state_wind = state.flatten()[locs]
    state_wind = np.array([fv.fipsToState[st] for st in state_wind])

    all_ratings = []

    wind_df = pd.DataFrame({'cig': con_reports.astype(int), 
                            'wfo': wfo_wind,
                            'st': state_wind})

    group_ids = np.repeat(np.arange(1, len(nat_wind_dist)+1), nat_wind_dist)

    wind_df['sim'] = group_ids

    if wind_df.empty:
        wind_df['sig'] = None
    else:
        wind_df['sig'] = wind_df.apply(lambda row: u.return_mag(row.cig, month, row.wfo, 'wind'),axis=1)

    nonsig_lists = []
    sig_lists = []

    reports_df = wind_df

# Get affected wfos
affected_wfos = np.unique(wfo[prob_holder > 0])
affected_wfos = [s for s in affected_wfos if '0' not in s]

# Get affected states
affected_states = np.unique(map_func(state[prob_holder > 0]))
affected_states = [s for s in affected_states if '0' not in s]

outdir = pathlib.Path(out_path,'dates',otlk_ts,'lsr',haz_type).resolve()
outdir.mkdir(parents=True,exist_ok=True)

outdir_json = str(outdir).replace('/images/','/jsons/')
pathlib.Path(outdir_json).mkdir(parents=True,exist_ok=True)

make_plots(reports_df, otlk_ts, outdir, haz_type, 
           affected_wfos, affected_states,only_nat=False)