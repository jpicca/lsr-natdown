import pandas as pd
import numpy as np
from scipy import stats
import math
from skimage import measure
from scipy.ndimage import binary_dilation
from scipy import interpolate as I
import fipsvars as fv

from pdb import set_trace as st


def fix_neg(arr):
    """ Fix negative values in an array by replacing them with 0"""
    arr[arr <= 0] = 0
    return arr

def make_continuous(probs, haz='torn'):
    """ Convert categorical probabilities to continuous values using contours """
    if haz == 'torn':
        vals = [1, 2, 5, 10, 15, 30, 45, 60]
        # vals = [0.01, 0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60]
    else:
        vals = [5, 15, 30, 45, 60]
        # vals = [0.05, 0.15, 0.30, 0.45, 0.60]
    continuous = np.zeros_like(probs)
    contours = [measure.find_contours(probs, v-1e-10) for v in vals]
    for tcontours, val in zip(contours, vals):
        for contour in tcontours:
            x, y = zip(*contour.astype(int))
            continuous[x, y] = val
    continuous = interpolate(continuous).astype(int, copy=False)
    continuous[probs < vals[0]] = 0
    return continuous

def interpolate(image):
    valid_mask = image > 0
    coords = np.array(np.nonzero(valid_mask)).T
    values = image[valid_mask]
    INTERP = I.LinearNDInterpolator(coords, values, fill_value=0)
    new_image = INTERP(list(np.ndindex(image.shape))).reshape(image.shape)
    return new_image

def find_neighbor_offices(grid, office_label):
    """
    grid: 2D numpy array of strings, each an office label.
    office_label: the office you want neighboring offices for.
    """
    # Mask of the office
    mask = (grid == office_label)

    # Dilate the mask by 1 cell (8-connectivity)
    dilated = binary_dilation(mask)

    # Neighbors are all labels in the dilated region
    # except the original office and except mask itself
    neighbor_mask = dilated & ~mask

    neighbors = np.unique(grid[neighbor_mask])
    return neighbors

def gather_features(wfo_unique, wfo, bdt, 
                    wind_cov, hail_cov, torn_cov, 
                    wind_cig, hail_cig, torn_cig, 
                    population):
    
    # features
    # all
    toy_sin,toy_cos = [],[]
    otlk_timestamp = []
    wfo_list = []
    maxhaz,medhaz,minhaz = [],[],[]
    popwfo = []
    
    # tor
    maxtor,medtor,mintor = [],[],[]
    areator, area2tor, area5tor, area10tor, area15tor, area30tor, area45tor, area60tor = [],[],[],[],[],[],[],[]
    areacigtor, areacig0tor, areacig1tor, areacig2tor = [],[],[],[]
    poptor, pop2tor, pop5tor, pop10tor, pop15tor, pop30tor, pop45tor, pop60tor = [],[],[],[],[],[],[],[]
    popcig0tor, popcig1tor, popcig2tor = [],[],[]
    poptordensity = []

    maxtor_nat = []
    maxtor_nei, medtor_nei, mintor_nei = [],[],[]
    
    # hail
    maxhail,medhail,minhail = [],[],[]
    areahail, area5hail, area15hail, area30hail, area45hail, area60hail = [],[],[],[],[],[]
    areacighail, areacig0hail, areacig1hail, areacig2hail = [],[],[],[]
    pophail, pop5hail, pop15hail, pop30hail, pop45hail, pop60hail = [],[],[],[],[],[]
    popcig0hail, popcig1hail, popcig2hail = [],[],[]
    pophaildensity = []

    maxhail_nat = []
    maxhail_nei, medhail_nei, minhail_nei = [],[],[]
    
    # wind
    maxwind,medwind,minwind = [],[],[]
    areawind, area5wind, area15wind, area30wind, area45wind, area60wind = [],[],[],[],[],[]
    areacigwind, areacig0wind, areacig1wind, areacig2wind = [],[],[],[]
    popwind, pop5wind, pop15wind, pop30wind, pop45wind, pop60wind = [],[],[],[],[],[]
    popcig0wind, popcig1wind, popcig2wind = [],[],[]
    popwinddensity = []

    maxwind_nat = []
    maxwind_nei, medwind_nei, minwind_nei = [],[],[]
    
    
    for wfo_current in wfo_unique:
        truth_array = wfo_current == wfo

        # neighboring offices
        neighbors = find_neighbor_offices(wfo,wfo_current)
        truth_array_neighbor = np.isin(wfo, neighbors)
        
        # features
        # all
        toy_sin.append(math.sin(2*math.pi*bdt.timetuple().tm_yday/366))
        toy_cos.append(math.cos(2*math.pi*bdt.timetuple().tm_yday/366))
        otlk_timestamp.append(bdt.strftime('%Y%m%d%H'))
        wfo_list.append(wfo_current)
        maxhaz.append(np.max([wind_cov[truth_array],torn_cov[truth_array],hail_cov[truth_array]]))
        minhaz.append(np.min([wind_cov[truth_array],torn_cov[truth_array],hail_cov[truth_array]]))
        medhaz.append(np.median([wind_cov[truth_array],torn_cov[truth_array],hail_cov[truth_array]]))
        popwfo.append(population[truth_array].sum())
        
    
        # wind
        maxwind.append(np.max(wind_cov[truth_array]))
        minwind.append(np.min(wind_cov[truth_array]))
        medwind.append(np.median(wind_cov[truth_array]))
        
        areawind.append(np.sum(wind_cov[truth_array] > 0))
        area5wind.append(np.sum(wind_cov[truth_array] == 5))
        area15wind.append(np.sum(wind_cov[truth_array] == 15))
        area30wind.append(np.sum(wind_cov[truth_array] == 30))
        area45wind.append(np.sum(wind_cov[truth_array] == 45))
        area60wind.append(np.sum(wind_cov[truth_array] == 60))
    
        areacigwind.append(np.sum(wind_cig[truth_array] > 0))
        areacig0wind.append(np.sum(np.logical_and(wind_cig[truth_array] == 0,wind_cov[truth_array] > 0)))
        areacig1wind.append(np.sum(wind_cig[truth_array] == 1))
        areacig2wind.append(np.sum(wind_cig[truth_array] == 2))
    
        popwind.append(np.sum((wind_cov[truth_array] > 0)*population[truth_array]))
        pop5wind.append(np.sum((wind_cov[truth_array] == 5)*population[truth_array]))
        pop15wind.append(np.sum((wind_cov[truth_array] == 15)*population[truth_array]))
        pop30wind.append(np.sum((wind_cov[truth_array] == 30)*population[truth_array]))
        pop45wind.append(np.sum((wind_cov[truth_array] == 45)*population[truth_array]))
        pop60wind.append(np.sum((wind_cov[truth_array] == 60)*population[truth_array]))
    
        popcig0wind.append(np.sum((np.logical_and(wind_cig[truth_array] == 0,wind_cov[truth_array] > 0))*population[truth_array]))
        popcig1wind.append(np.sum((wind_cig[truth_array] == 1)*population[truth_array]))
        popcig2wind.append(np.sum((wind_cig[truth_array] == 2)*population[truth_array]))
    
        popwinddensity.append(np.sum((wind_cov[truth_array] > 0)*population[truth_array]) / np.sum(wind_cov[truth_array] > 0))
        
        maxwind_nat.append(np.max(wind_cov))
        maxwind_nei.append(np.max(wind_cov[truth_array_neighbor]))
        medwind_nei.append(np.median(wind_cov[truth_array_neighbor]))
        minwind_nei.append(np.min(wind_cov[truth_array_neighbor]))

        # hail
        maxhail.append(np.max(hail_cov[truth_array]))
        minhail.append(np.min(hail_cov[truth_array]))
        medhail.append(np.median(hail_cov[truth_array]))
        
        areahail.append(np.sum(hail_cov[truth_array] > 0))
        area5hail.append(np.sum(hail_cov[truth_array] == 5))
        area15hail.append(np.sum(hail_cov[truth_array] == 15))
        area30hail.append(np.sum(hail_cov[truth_array] == 30))
        area45hail.append(np.sum(hail_cov[truth_array] == 45))
        area60hail.append(np.sum(hail_cov[truth_array] == 60))
    
        areacighail.append(np.sum(hail_cig[truth_array] > 0))
        areacig0hail.append(np.sum(np.logical_and(hail_cig[truth_array] == 0,hail_cov[truth_array] > 0)))
        areacig1hail.append(np.sum(hail_cig[truth_array] == 1))
        areacig2hail.append(np.sum(hail_cig[truth_array] == 2))
    
        pophail.append(np.sum((hail_cov[truth_array] > 0)*population[truth_array]))
        pop5hail.append(np.sum((hail_cov[truth_array] == 5)*population[truth_array]))
        pop15hail.append(np.sum((hail_cov[truth_array] == 15)*population[truth_array]))
        pop30hail.append(np.sum((hail_cov[truth_array] == 30)*population[truth_array]))
        pop45hail.append(np.sum((hail_cov[truth_array] == 45)*population[truth_array]))
        pop60hail.append(np.sum((hail_cov[truth_array] == 60)*population[truth_array]))
    
        popcig0hail.append(np.sum((np.logical_and(hail_cig[truth_array] == 0,hail_cov[truth_array] > 0))*population[truth_array]))
        popcig1hail.append(np.sum((hail_cig[truth_array] == 1)*population[truth_array]))
        popcig2hail.append(np.sum((hail_cig[truth_array] == 2)*population[truth_array]))
    
        pophaildensity.append(np.sum((hail_cov[truth_array] > 0)*population[truth_array]) / np.sum(hail_cov[truth_array] > 0))
        
        maxhail_nat.append(np.max(hail_cov))
        maxhail_nei.append(np.max(hail_cov[truth_array_neighbor]))
        medhail_nei.append(np.median(hail_cov[truth_array_neighbor]))
        minhail_nei.append(np.min(hail_cov[truth_array_neighbor]))

        # tor
        maxtor.append(np.max(torn_cov[truth_array]))
        mintor.append(np.min(torn_cov[truth_array]))
        medtor.append(np.median(torn_cov[truth_array]))
        
        areator.append(np.sum(torn_cov[truth_array] > 0))
        area2tor.append(np.sum(torn_cov[truth_array] == 2))
        area5tor.append(np.sum(torn_cov[truth_array] == 5))
        area10tor.append(np.sum(torn_cov[truth_array] == 10))
        area15tor.append(np.sum(torn_cov[truth_array] == 15))
        area30tor.append(np.sum(torn_cov[truth_array] == 30))
        area45tor.append(np.sum(torn_cov[truth_array] == 45))
        area60tor.append(np.sum(torn_cov[truth_array] == 60))
    
        areacigtor.append(np.sum(torn_cig[truth_array] > 0))
        areacig0tor.append(np.sum(np.logical_and(torn_cig[truth_array] == 0,torn_cov[truth_array] > 0)))
        areacig1tor.append(np.sum(torn_cig[truth_array] == 1))
        areacig2tor.append(np.sum(torn_cig[truth_array] == 2))
    
        poptor.append(np.sum((torn_cov[truth_array] > 0)*population[truth_array]))
        pop2tor.append(np.sum((torn_cov[truth_array] == 2)*population[truth_array]))
        pop5tor.append(np.sum((torn_cov[truth_array] == 5)*population[truth_array]))
        pop10tor.append(np.sum((torn_cov[truth_array] == 10)*population[truth_array]))
        pop15tor.append(np.sum((torn_cov[truth_array] == 15)*population[truth_array]))
        pop30tor.append(np.sum((torn_cov[truth_array] == 30)*population[truth_array]))
        pop45tor.append(np.sum((torn_cov[truth_array] == 45)*population[truth_array]))
        pop60tor.append(np.sum((torn_cov[truth_array] == 60)*population[truth_array]))
    
        popcig0tor.append(np.sum((np.logical_and(torn_cig[truth_array] == 0,torn_cov[truth_array] > 0))*population[truth_array]))
        popcig1tor.append(np.sum((torn_cig[truth_array] == 1)*population[truth_array]))
        popcig2tor.append(np.sum((torn_cig[truth_array] == 2)*population[truth_array]))
    
        poptordensity.append(np.sum((torn_cov[truth_array] > 0)*population[truth_array]) / np.sum(torn_cov[truth_array] > 0))

        maxtor_nat.append(np.max(torn_cov))
        maxtor_nei.append(np.max(torn_cov[truth_array_neighbor]))
        medtor_nei.append(np.median(torn_cov[truth_array_neighbor]))
        mintor_nei.append(np.min(torn_cov[truth_array_neighbor]))

    df_features = pd.DataFrame([
        otlk_timestamp,wfo_list,
        toy_sin,toy_cos,maxhaz,medhaz,minhaz,popwfo,maxtor,medtor,mintor,
        areator, area2tor, area5tor, area10tor, area15tor, area30tor, area45tor, area60tor,
        areacigtor, areacig0tor, areacig1tor, areacig2tor,
        poptor, pop2tor, pop5tor, pop10tor, pop15tor, pop30tor, pop45tor, pop60tor,
        popcig0tor, popcig1tor, popcig2tor,
        poptordensity,
        maxtor_nat,maxtor_nei,medtor_nei,mintor_nei,
        maxhail,medhail,minhail,
        areahail, area5hail, area15hail, area30hail, area45hail, area60hail,
        areacighail, areacig0hail, areacig1hail, areacig2hail,
        pophail, pop5hail, pop15hail, pop30hail, pop45hail, pop60hail,
        popcig0hail, popcig1hail, popcig2hail,
        pophaildensity,
        maxhail_nat,maxhail_nei,medhail_nei,minhail_nei,
        maxwind,medwind,minwind,
        areawind, area5wind, area15wind, area30wind, area45wind, area60wind,
        areacigwind, areacig0wind, areacig1wind, areacig2wind,
        popwind, pop5wind, pop15wind, pop30wind, pop45wind, pop60wind,
        popcig0wind, popcig1wind, popcig2wind,
        popwinddensity,
        maxwind_nat,maxwind_nei,medwind_nei,minwind_nei
    ]).T

    return df_features

def gather_ct_features(wfo_unique, wfo_1d, bdt, ct_vals, population, wfo):
    """ Gather features for HREFCT probabilities """
    
    # features
    # all
    otlk_timestamp = []
    wfo_list = []
    
    # wind
    maxct,medct = [],[]
    maxhourct,medhourct = [],[]
    hourofmax_sin,hourofmax_cos = [],[]
    sumct = []
    sumhourct = []
    maxpopct,medpopct = [],[]
    sumpopct = []
    sumhourpopct = []
    maxhourpopct,medhourpopct = [],[]

    # get all HREFCT probabilities for all times, wfo-st by wfo-st
    for wfo_current in wfo_unique:
        wfo_probs = ct_vals[wfo_current == wfo_1d]

        # features
        # all
        otlk_timestamp.append(bdt.strftime('%Y%m%d%H'))
        wfo_list.append(wfo_current)
        
        maxct.append(np.max(wfo_probs))
        medct.append(np.median(wfo_probs))
        
        maxhourct.append(np.mean(wfo_probs,axis=0).max())
        medhourct.append(np.median(np.mean(wfo_probs,axis=0)))

        hourofmax_sin.append(math.sin(((bdt.hour + np.argmax(np.mean(wfo_probs,axis=0))) % 24)/24))
        hourofmax_cos.append(math.cos(((bdt.hour + np.argmax(np.mean(wfo_probs,axis=0))) % 24)/24))

        sumct.append(wfo_probs.sum())
        sumhourct.append(np.mean(wfo_probs,axis=0).sum())
        sumpopct.append((wfo_probs*population[wfo_current == wfo][:, np.newaxis]).sum())
        sumhourpopct.append(np.mean(wfo_probs*population[wfo_current == wfo][:, np.newaxis],axis=0).sum())

        maxpopct.append((wfo_probs*population[wfo_current == wfo][:, np.newaxis]).max())
        medpopct.append(np.median(wfo_probs*population[wfo_current == wfo][:, np.newaxis]))
        maxhourpopct.append(np.mean(wfo_probs*population[wfo_current == wfo][:, np.newaxis],axis=0).max())
        medhourpopct.append(np.median(np.mean(wfo_probs*population[wfo_current == wfo][:, np.newaxis],axis=0)))

    df_ctfeatures = pd.DataFrame([
        otlk_timestamp,wfo_list,
        maxct,medct,
        maxhourct,medhourct,
        hourofmax_sin,hourofmax_cos,
        sumct,sumhourct,sumpopct,sumhourpopct,
        maxpopct,medpopct,maxhourpopct,medhourpopct
    ]).T

    return df_ctfeatures

def find_region(wfo):
    if wfo in ["EKA", "LOX", "STO", "SGX", "MTR", "HNX","MFR", "PDT", "PQR","SEW", "OTX","REV", "VEF", "LKN","PSR", "TWC", "FGZ",
                "SLC","BOI", "PIH","MSO","TFX","RIW","GJT","EPZ"]:
        region = 'west'
    elif wfo in ["BOU", "PUB","CYS","BYZ", "GGW","BIS","ABR", "UNR","LBF", "GID","GLD", "DDC","AMA", "LUB", "MAF","ABQ"]:
        region = 'highplains'
    elif wfo in ["OUN", "TSA","ICT","FWD", "SJT", "EWX"]:
        region = 'plains'
    elif wfo in ["FSD","DLH", "MPX", "FGF","OAX","DMX", "DVN","TOP", "EAX", "SGF", "LSX","LOT", "ILX","IND", "IWX",
                "DTX", "APX", "GRR", "MQT","GRB", "ARX", "MKX"]:
        region = 'midwest'
    elif wfo in ["BMX", "HUN", "MOB","JAN","FFC","CHS", "CAE", "GSP","JAX", "KEY", "MLB", "MFL", "TAE", "TBW","MEG", "MRX", "OHX",
                "LMK", "JKL", "PAH","LCH", "LIX", "SHV","LZK","HGX", "CRP", "BRO"]:
        region = 'southeast'
    elif wfo in ["PHI","PBZ", "CTP","RLX","LWX", "RNK", "AKQ","MHX", "RAH", "ILM","CLE", "ILN","CAR", "GYX","BOX",
                "ALY", "BGM", "BUF", "OKX","BTV"]:
        region = 'northeast'
    else:
        region = 'other'
    
    return region

def find_season(month,temp=False):

    if temp:
        if month in [10,11,12,1,2,3,4]:
            season = 'cool'
        else:
            season = 'warm'
    else:
        if month in [12,1,2]:
            season = 'winter'
        elif month in [3,4,5]:
            season = 'spring'
        elif month in [6,7,8]:
            season = 'summer'
        elif month in [9,10,11]:
            season = 'fall'

    return season

def return_mag_bin(pred):
    if pred < 10:
        return str(int(pred))
    else:
        return '>9'
    

def return_mag(lev,month,wfo,hazard):
    if lev == 0:

        season = find_season(month)
    
        region = find_region(wfo)
    
        proportions = {'wind': {'west': {'winter': 0.09523809523809523,
                        'spring': 0.032490974729241874,
                        'summer': 0.049143708116157855,
                        'fall': 0.07954545454545454},
                        'highplains': {'winter': 0.0392156862745098,
                        'spring': 0.160075329566855,
                        'summer': 0.1625668449197861,
                        'fall': 0.15789473684210525},
                        'plains': {'winter': 0.12017167381974249,
                        'spring': 0.1680933852140078,
                        'summer': 0.16076447442383363,
                        'fall': 0.14008941877794337},
                        'midwest': {'winter': 0.16909620991253643,
                        'spring': 0.09427860696517414,
                        'summer': 0.08757637474541752,
                        'fall': 0.0836092715231788},
                        'southeast': {'winter': 0.056451612903225805,
                        'spring': 0.05987093690248566,
                        'summer': 0.03495624425856984,
                        'fall': 0.037642397226349676},
                        'northeast': {'winter': 0.013916500994035786,
                        'spring': 0.016916780354706683,
                        'summer': 0.019506098022877054,
                        'fall': 0.0076481835564053535},
                        'other': {'winter': 0.082,
                        'spring': 0.0886,
                        'summer': 0.085,
                        'fall': 0.084}},
                        'hail': {'west': {'winter': 0.0,
                        'spring': 0.0,
                        'summer': 0.036036036036036036,
                        'fall': 0.030534351145038167},
                        'highplains': {'winter': 0.06557377049180328,
                        'spring': 0.07942583732057416,
                        'summer': 0.10298507462686567,
                        'fall': 0.04953560371517028},
                        'plains': {'winter': 0.0410958904109589,
                        'spring': 0.07186678352322524,
                        'summer': 0.10559006211180125,
                        'fall': 0.07168458781362007},
                        'midwest': {'winter': 0.045146726862302484,
                        'spring': 0.04868603042876902,
                        'summer': 0.09302325581395349,
                        'fall': 0.07397003745318352},
                        'southeast': {'winter': 0.06479113384484228,
                        'spring': 0.07476212052560036,
                        'summer': 0.05352363960749331,
                        'fall': 0.0603448275862069},
                        'northeast': {'winter': 0.09090909090909091,
                        'spring': 0.03322995126273815,
                        'summer': 0.04234769687964339,
                        'fall': 0.010869565217391304},
                        'other': {'winter': 0.056,
                        'spring': 0.056,
                        'summer': 0.073,
                        'fall': 0.056
                        }}}

        sig_pct = proportions[hazard][region][season]

        return np.random.choice([0,1], size=1, replace=True, p=[1-sig_pct,sig_pct])[0]
        
    else:
        return np.random.choice([0,1], size=1, replace=True, p=fv.cig_dists[hazard][str(lev)])[0]