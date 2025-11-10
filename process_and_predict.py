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
parser.add_argument("-t", "--test", default=1, type=int, required=False)
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

# Bias files
bias_wind_file = f'{data_path}/wind_bias_table.csv'
bias_hail_file = f'{data_path}/hail_bias_table.csv'

# ML models
with open(f'{ml_path.as_posix()}/wfo-label-encoder.model','rb') as f:
    wfo_label_encoder = pickle.load(f)

with open(f'{ml_path.as_posix()}/det/hgb-det_wind-simple.model','rb') as f:
    wind_model = pickle.load(f)

with open(f'{ml_path.as_posix()}/det/hgb-det_hail-simple.model','rb') as f:
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
wfo_state_2d = np.char.add(wfo.astype('str'),map_func(state))

bias_wind_df = pd.read_csv(bias_wind_file)
bias_hail_df = pd.read_csv(bias_hail_file)

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
wfo_st_windoutlook = np.unique(wfo_state_2d[wind_cov > 0])
wfo_st_hailoutlook = np.unique(wfo_state_2d[hail_cov > 0])
wfo_st_tornoutlook = np.unique(wfo_state_2d[torn_cov > 0])

wfo_st_alloutlook = np.concatenate([wfo_st_windoutlook,wfo_st_hailoutlook,wfo_st_tornoutlook])
wfo_st_unique = np.unique(wfo_st_alloutlook[[ '0' not in s for s in wfo_st_alloutlook]])

# Outlook based features
df_otlk = u.gather_features(wfo_st_unique, wfo_state_2d, otlkdt, 
                    wind_cov, hail_cov, torn_cov, 
                    wind_con, hail_con, torn_con, 
                    population)
df_otlk.columns = fv.col_names_outlook  # Rename columns to assist merge with HREF features
df_otlk['doy'] = get_doy(df_otlk)

# HREF feature processing
if isTest:
    ct_files = glob.glob(f'{href_path.as_posix()}/{otlkdt.year}{otlkdt.month:02d}{otlkdt.day:02d}/thunder/spc_post.t00z.hrefct.1hr.f*')
else:
    ct_files = glob.glob(f'/nfsops/ops_users/nadata2/awips2/grib2/spcpost/{otlkdt.year}{otlkdt.month:02d}{otlkdt.day:02d}/thunder/spc_post.t00z.hrefct.1hr.f*')

ct_files.sort()

ct_arrs = []

if len(ct_files) != 48:
    import sys
    print("Some HREF files missing, exiting...")
    sys.exit(0)

for i,file in enumerate(ct_files[otlkdt.hour-1:35]):

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
wfo_st_impacted = np.isin(wfo_state_2d,wfo_st_unique)

# get 1d list of wfo-st for each ndfd grid box
wfo_st_1d = wfo_state_2d[wfo_st_impacted]

G_href = pg.Gridder(lons_href, lats_href)

# find the href grid indices for these ndfd grids
idx_href = G_href.grid_points(lons[wfo_st_impacted],lats[wfo_st_impacted])

# Get all HREFCT probabilities for all times in each CWA-ST
ct_vals = []
for idx in idx_href:
    ct_vals.append(val_stack[:,idx[0],idx[1]])

ct_vals = np.array(ct_vals)

df_href = u.gather_ct_features(wfo_st_unique, wfo_st_1d, otlkdt, ct_vals, population, wfo_state_2d)
df_href.columns = fv.col_names_href # Rename columns to assist merge with outlook features

df_all = df_otlk.merge(df_href, on=['otlk_timestamp','wfo_st_list'], how='inner').fillna(0)

wfo_list_trans = models['le'].transform(df_all.wfo_st_list.str.slice(0,3))
X = df_all.iloc[:,2:]
X['wfo'] = wfo_list_trans
Xwind = X[fv.col_names_wind]
Xhail = X[fv.col_names_hail]

mask = np.char.find(wfo_state_2d, '0') != -1

## Make bias correction dataframe
regions = df_all.wfo_st_list.str[:3].apply(lambda x: u.find_region(x))
seasons = df_all.otlk_timestamp.str[4:6].astype(int).apply(lambda x: u.find_season(x,temp=True))

df_bias = pd.DataFrame({'region': regions, 'season': seasons})

if haz_type == 'hail':
    if hail_cov.max() == 0:
        print(f'All hail coverage probabilities less than 5%. No hail predictions made for {otlk_ts}.')
        import sys
        sys.exit(0)

    hail_preds = np.round(models['hail'].predict(Xhail))
    hail_preds = u.fix_neg(hail_preds)
    
    df_preds = pd.DataFrame({
        'wfost': df_all.wfo_st_list,
        'hail': hail_preds
    })

    # Incorporate bias correction
    df_bias['magnitude_bin'] = df_preds.hail.apply(lambda x: u.return_mag_bin(x))
    df_bias['threat_level'] = df_all.maxhail/100

    bias_calc = df_bias.merge(bias_hail_df, on=['season', 'region', 'threat_level', 'magnitude_bin'], how='left')
    bias_calc.fillna(0, inplace=True)

    # df_preds['hail_orig'] = df_preds['hail'].copy()

    df_preds['hail'] = df_preds['hail'] - bias_calc['median_bias'].values

    nat_preds = df_preds[['hail']].sum()
    nat_hail_dist = u.add_distribution(nat_preds, haz='hail')

    # Create weight grids to place reports
    hail_cov[hail_cov < 0] = 0

    # continuous coverage files
    hail_cov_con = u.make_continuous(hail_cov, haz='hail')

    rel_freq_hail = np.zeros(wfo_state_2d.shape)

    for _,row in df_preds.iterrows():
        area = row.wfost
        hail_pred = row.hail
        
        rel_freq_hail[wfo_state_2d == area] = hail_pred

    hail_cov_norm = 100*hail_cov / hail_cov.max()
    if rel_freq_hail.max() == 0:
        rel_freq_hail_norm = rel_freq_hail
    else:
        rel_freq_hail_norm = 100*rel_freq_hail / rel_freq_hail.max()

    hail_weight = hail_cov_norm + rel_freq_hail_norm
    hail_weight[hail_cov_norm == 0] = 0
    hail_weight[mask] = 0

    cumulative_weights_hail = hail_weight.cumsum().astype(int)
    all_reps_hail_sum = nat_hail_dist.sum()

    # Get hail locations and sig count numbers
    _locs = np.random.randint(
                0.001, cumulative_weights_hail.max(), size=all_reps_hail_sum)
    locs = cumulative_weights_hail.searchsorted(_locs)

    con_reports = hail_con.flatten()[locs]
    wfo_states_hail = wfo_state_2d.flatten()[locs]

    all_ratings = []

    hail_df = pd.DataFrame({'cig': con_reports.astype(int), 'wfo_st': wfo_states_hail})
    hail_df['wfo'] = hail_df.wfo_st.str[:3]

    group_ids = np.repeat(np.arange(1, len(nat_hail_dist)+1), nat_hail_dist)

    hail_df['sim'] = group_ids
    hail_df['sig'] = hail_df.apply(lambda row: u.return_mag(row.cig, month, row.wfo, 'hail'),axis=1)

    nonsig_lists = []
    sig_lists = []

    for place in wfo_st_unique:
        hail_df_filt = hail_df[hail_df.wfo_st == place]
        
        nonsig_counts = hail_df_filt.groupby('sim').count()['cig'].reindex(index=np.arange(1,fv.nsims+1,1),fill_value=0).values
        nonsig_lists.append(nonsig_counts)

        sig_counts = hail_df_filt.groupby('sim').sum()['sig'].reindex(index=np.arange(1,fv.nsims+1,1),fill_value=0).values
        sig_lists.append(sig_counts)

    dists_df = pd.DataFrame({'wfost': wfo_st_unique,'hail_dists': nonsig_lists,'sighail_dists': sig_lists})

elif haz_type == 'wind':  
    if wind_cov.max() == 0:
        print(f'All wind coverage probabilities less than 5%. No wind predictions made for {otlk_ts}.')
        import sys
        sys.exit(0)

    wind_preds = np.round(models['wind'].predict(Xwind))
    wind_preds = u.fix_neg(wind_preds)

    df_preds = pd.DataFrame({
        'wfost': df_all.wfo_st_list,
        'wind': wind_preds
    })

    # Incorporate bias correction
    df_bias['magnitude_bin'] = df_preds.wind.apply(lambda x: u.return_mag_bin(x))
    df_bias['threat_level'] = df_all.maxwind/100

    bias_calc = df_bias.merge(bias_wind_df, on=['season', 'region', 'threat_level', 'magnitude_bin'], how='left')
    bias_calc.fillna(0, inplace=True)

    # df_preds['wind_orig'] = df_preds['wind'].copy()

    df_preds['wind'] = df_preds['wind'] - bias_calc['median_bias'].values

    nat_preds = df_preds[['wind']].sum()
    nat_wind_dist = u.add_distribution(nat_preds, haz='wind')

    # Create weight grids to place reports
    wind_cov[wind_cov < 0] = 0

    # continuous coverage files
    wind_cov_con = u.make_continuous(wind_cov, haz='wind')

    rel_freq_wind = np.zeros(wfo_state_2d.shape)

    for _,row in df_preds.iterrows():
        area = row.wfost
        wind_pred = row.wind
        
        rel_freq_wind[wfo_state_2d == area] = wind_pred

    wind_cov_norm = 100*wind_cov / wind_cov.max()
    if rel_freq_wind.max() == 0:
        rel_freq_wind_norm = rel_freq_wind
    else:
        rel_freq_wind_norm = 100*rel_freq_wind / rel_freq_wind.max()

    wind_weight = wind_cov_norm + rel_freq_wind_norm
    wind_weight[wind_cov_norm == 0] = 0
    wind_weight[mask] = 0

    cumulative_weights_wind = wind_weight.cumsum().astype(int)
    all_reps_wind_sum = nat_wind_dist.sum()
    all_reps_wind_sum = nat_wind_dist.sum()

    # Get wind locations and sig count numbers
    _locs = np.random.randint(
                0.001, cumulative_weights_wind.max(), size=all_reps_wind_sum)
    locs = cumulative_weights_wind.searchsorted(_locs)

    con_reports = wind_con.flatten()[locs]
    wfo_states_wind = wfo_state_2d.flatten()[locs]

    all_ratings = []

    wind_df = pd.DataFrame({'cig': con_reports.astype(int), 'wfo_st': wfo_states_wind})
    wind_df['wfo'] = wind_df.wfo_st.str[:3]

    group_ids = np.repeat(np.arange(1, len(nat_wind_dist)+1), nat_wind_dist)

    wind_df['sim'] = group_ids
    wind_df['sig'] = wind_df.apply(lambda row: u.return_mag(row.cig, month, row.wfo, 'wind'),axis=1)

    nonsig_lists = []
    sig_lists = []

    for place in wfo_st_unique:
        wind_df_filt = wind_df[wind_df.wfo_st == place]
        
        nonsig_counts = wind_df_filt.groupby('sim').count()['cig'].reindex(index=np.arange(1,fv.nsims+1,1),fill_value=0).values
        nonsig_lists.append(nonsig_counts)

        sig_counts = wind_df_filt.groupby('sim').sum()['sig'].reindex(index=np.arange(1,fv.nsims+1,1),fill_value=0).values
        sig_lists.append(sig_counts)

    dists_df = pd.DataFrame({'wfost': wfo_st_unique,'wind_dists': nonsig_lists,'sigwind_dists': sig_lists})

outdir = pathlib.Path(out_path,'dates',otlk_ts,'lsr',haz_type).resolve()
outdir.mkdir(parents=True,exist_ok=True)

make_plots(dists_df, otlk_ts, outdir, haz_type, only_nat=True)