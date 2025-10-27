import pandas as pd
import numpy as np
from scipy import stats
import math
from skimage import measure
from scipy import interpolate as I
import fipsvars as fv


def fix_neg(arr):
    """ Fix negative values in an array by replacing them with 0"""
    arr[arr <= 0] = 0
    return arr

def make_continuous(probs, haz='torn'):
    """ Convert categorical probabilities to continuous values using contours """
    if haz == 'torn':
        vals = [1, 2, 5, 10, 15, 30, 45, 60]
    else:
        vals = [5, 15, 30, 45, 60]
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

def gather_features(wfo_st_unique, wfo_state_2d, bdt, 
                    wind_cov, hail_cov, torn_cov, 
                    wind_cig, hail_cig, torn_cig, 
                    population):
    
    # features
    # all
    toy_sin,toy_cos = [],[]
    otlk_timestamp = []
    wfo_st_list = []
    maxhaz,medhaz,minhaz = [],[],[]
    popwfo = []
    
    # tor
    maxtor,medtor,mintor = [],[],[]
    areator, area2tor, area5tor, area10tor, area15tor, area30tor, area45tor, area60tor = [],[],[],[],[],[],[],[]
    areacigtor, areacig0tor, areacig1tor, areacig2tor = [],[],[],[]
    poptor, pop2tor, pop5tor, pop10tor, pop15tor, pop30tor, pop45tor, pop60tor = [],[],[],[],[],[],[],[]
    popcig0tor, popcig1tor, popcig2tor = [],[],[]
    poptordensity = []
    
    # hail
    maxhail,medhail,minhail = [],[],[]
    areahail, area5hail, area15hail, area30hail, area45hail, area60hail = [],[],[],[],[],[]
    areacighail, areacig0hail, areacig1hail, areacig2hail = [],[],[],[]
    pophail, pop5hail, pop15hail, pop30hail, pop45hail, pop60hail = [],[],[],[],[],[]
    popcig0hail, popcig1hail, popcig2hail = [],[],[]
    pophaildensity = []
    
    # wind
    maxwind,medwind,minwind = [],[],[]
    areawind, area5wind, area15wind, area30wind, area45wind, area60wind = [],[],[],[],[],[]
    areacigwind, areacig0wind, areacig1wind, areacig2wind = [],[],[],[]
    popwind, pop5wind, pop15wind, pop30wind, pop45wind, pop60wind = [],[],[],[],[],[]
    popcig0wind, popcig1wind, popcig2wind = [],[],[]
    popwinddensity = []
    
    
    for wfo_st in wfo_st_unique:
        truth_array = wfo_st == wfo_state_2d
        
        # features
        # all
        toy_sin.append(math.sin(2*math.pi*bdt.timetuple().tm_yday/366))
        toy_cos.append(math.cos(2*math.pi*bdt.timetuple().tm_yday/366))
        otlk_timestamp.append(bdt.strftime('%Y%m%d%H'))
        wfo_st_list.append(wfo_st)
        maxhaz.append(np.max([wind_cov[truth_array],torn_cov[truth_array],hail_cov[truth_array]]))
        minhaz.append(np.min([wind_cov[truth_array],torn_cov[truth_array],hail_cov[truth_array]]))
        medhaz.append(np.median([wind_cov[truth_array],torn_cov[truth_array],hail_cov[truth_array]]))
        popwfo.append(population[truth_array].sum())
        
    
        # wind
        maxwind.append(np.max(wind_cov[truth_array]))
        minwind.append(np.min(wind_cov[truth_array]))
        medwind.append(np.median(wind_cov[truth_array]))
        
        areawind.append(np.sum(wind_cov[truth_array] > 0))
        area5wind.append(np.sum(wind_cov[truth_array] == 0.05))
        area15wind.append(np.sum(wind_cov[truth_array] == 0.15))
        area30wind.append(np.sum(wind_cov[truth_array] == 0.30))
        area45wind.append(np.sum(wind_cov[truth_array] == 0.45))
        area60wind.append(np.sum(wind_cov[truth_array] == 0.60))
    
        areacigwind.append(np.sum(wind_cig[truth_array] > 0))
        areacig0wind.append(np.sum(np.logical_and(wind_cig[truth_array] == 0,wind_cov[truth_array] > 0)))
        areacig1wind.append(np.sum(wind_cig[truth_array] == 1))
        areacig2wind.append(np.sum(wind_cig[truth_array] == 2))
    
        popwind.append(np.sum((wind_cov[truth_array] > 0)*population[truth_array]))
        pop5wind.append(np.sum((wind_cov[truth_array] == 0.05)*population[truth_array]))
        pop15wind.append(np.sum((wind_cov[truth_array] == 0.15)*population[truth_array]))
        pop30wind.append(np.sum((wind_cov[truth_array] == 0.30)*population[truth_array]))
        pop45wind.append(np.sum((wind_cov[truth_array] == 0.45)*population[truth_array]))
        pop60wind.append(np.sum((wind_cov[truth_array] == 0.60)*population[truth_array]))
    
        popcig0wind.append(np.sum((np.logical_and(wind_cig[truth_array] == 0,wind_cov[truth_array] > 0))*population[truth_array]))
        popcig1wind.append(np.sum((wind_cig[truth_array] == 1)*population[truth_array]))
        popcig2wind.append(np.sum((wind_cig[truth_array] == 2)*population[truth_array]))
    
        popwinddensity.append(np.sum((wind_cov[truth_array] > 0)*population[truth_array]) / np.sum(wind_cov[truth_array] > 0))
        
    
        # hail
        maxhail.append(np.max(hail_cov[truth_array]))
        minhail.append(np.min(hail_cov[truth_array]))
        medhail.append(np.median(hail_cov[truth_array]))
        
        areahail.append(np.sum(hail_cov[truth_array] > 0))
        area5hail.append(np.sum(hail_cov[truth_array] == 0.05))
        area15hail.append(np.sum(hail_cov[truth_array] == 0.15))
        area30hail.append(np.sum(hail_cov[truth_array] == 0.30))
        area45hail.append(np.sum(hail_cov[truth_array] == 0.45))
        area60hail.append(np.sum(hail_cov[truth_array] == 0.60))
    
        areacighail.append(np.sum(hail_cig[truth_array] > 0))
        areacig0hail.append(np.sum(np.logical_and(hail_cig[truth_array] == 0,hail_cov[truth_array] > 0)))
        areacig1hail.append(np.sum(hail_cig[truth_array] == 1))
        areacig2hail.append(np.sum(hail_cig[truth_array] == 2))
    
        pophail.append(np.sum((hail_cov[truth_array] > 0)*population[truth_array]))
        pop5hail.append(np.sum((hail_cov[truth_array] == 0.05)*population[truth_array]))
        pop15hail.append(np.sum((hail_cov[truth_array] == 0.15)*population[truth_array]))
        pop30hail.append(np.sum((hail_cov[truth_array] == 0.30)*population[truth_array]))
        pop45hail.append(np.sum((hail_cov[truth_array] == 0.45)*population[truth_array]))
        pop60hail.append(np.sum((hail_cov[truth_array] == 0.60)*population[truth_array]))
    
        popcig0hail.append(np.sum((np.logical_and(hail_cig[truth_array] == 0,hail_cov[truth_array] > 0))*population[truth_array]))
        popcig1hail.append(np.sum((hail_cig[truth_array] == 1)*population[truth_array]))
        popcig2hail.append(np.sum((hail_cig[truth_array] == 2)*population[truth_array]))
    
        pophaildensity.append(np.sum((hail_cov[truth_array] > 0)*population[truth_array]) / np.sum(hail_cov[truth_array] > 0))
        
        # tor
        maxtor.append(np.max(torn_cov[truth_array]))
        mintor.append(np.min(torn_cov[truth_array]))
        medtor.append(np.median(torn_cov[truth_array]))
        
        areator.append(np.sum(torn_cov[truth_array] > 0))
        area2tor.append(np.sum(torn_cov[truth_array] == 0.02))
        area5tor.append(np.sum(torn_cov[truth_array] == 0.05))
        area10tor.append(np.sum(torn_cov[truth_array] == 0.10))
        area15tor.append(np.sum(torn_cov[truth_array] == 0.15))
        area30tor.append(np.sum(torn_cov[truth_array] == 0.30))
        area45tor.append(np.sum(torn_cov[truth_array] == 0.45))
        area60tor.append(np.sum(torn_cov[truth_array] == 0.60))
    
        areacigtor.append(np.sum(torn_cig[truth_array] > 0))
        areacig0tor.append(np.sum(np.logical_and(torn_cig[truth_array] == 0,torn_cov[truth_array] > 0)))
        areacig1tor.append(np.sum(torn_cig[truth_array] == 1))
        areacig2tor.append(np.sum(torn_cig[truth_array] == 2))
    
        poptor.append(np.sum((torn_cov[truth_array] > 0)*population[truth_array]))
        pop2tor.append(np.sum((torn_cov[truth_array] == 0.02)*population[truth_array]))
        pop5tor.append(np.sum((torn_cov[truth_array] == 0.05)*population[truth_array]))
        pop10tor.append(np.sum((torn_cov[truth_array] == 0.10)*population[truth_array]))
        pop15tor.append(np.sum((torn_cov[truth_array] == 0.15)*population[truth_array]))
        pop30tor.append(np.sum((torn_cov[truth_array] == 0.30)*population[truth_array]))
        pop45tor.append(np.sum((torn_cov[truth_array] == 0.45)*population[truth_array]))
        pop60tor.append(np.sum((torn_cov[truth_array] == 0.60)*population[truth_array]))
    
        popcig0tor.append(np.sum((np.logical_and(torn_cig[truth_array] == 0,torn_cov[truth_array] > 0))*population[truth_array]))
        popcig1tor.append(np.sum((torn_cig[truth_array] == 1)*population[truth_array]))
        popcig2tor.append(np.sum((torn_cig[truth_array] == 2)*population[truth_array]))
    
        poptordensity.append(np.sum((torn_cov[truth_array] > 0)*population[truth_array]) / np.sum(torn_cov[truth_array] > 0))

    df_features = pd.DataFrame([
        otlk_timestamp,wfo_st_list,
        toy_sin,toy_cos,maxhaz,medhaz,minhaz,popwfo,maxtor,medtor,mintor,
        areator, area2tor, area5tor, area10tor, area15tor, area30tor, area45tor, area60tor,
        areacigtor, areacig0tor, areacig1tor, areacig2tor,
        poptor, pop2tor, pop5tor, pop10tor, pop15tor, pop30tor, pop45tor, pop60tor,
        popcig0tor, popcig1tor, popcig2tor,
        poptordensity,
        maxhail,medhail,minhail,
        areahail, area5hail, area15hail, area30hail, area45hail, area60hail,
        areacighail, areacig0hail, areacig1hail, areacig2hail,
        pophail, pop5hail, pop15hail, pop30hail, pop45hail, pop60hail,
        popcig0hail, popcig1hail, popcig2hail,
        pophaildensity,
        maxwind,medwind,minwind,
        areawind, area5wind, area15wind, area30wind, area45wind, area60wind,
        areacigwind, areacig0wind, areacig1wind, areacig2wind,
        popwind, pop5wind, pop15wind, pop30wind, pop45wind, pop60wind,
        popcig0wind, popcig1wind, popcig2wind,
        popwinddensity
    ]).T

    return df_features

def gather_ct_features(wfo_st_unique, wfo_st_1d, bdt, ct_vals, population, wfo_state_2d):
    """ Gather features for HREFCT probabilities """
    
    # features
    # all
    otlk_timestamp = []
    wfo_st_list = []
    
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
    for wfo_st in wfo_st_unique:
        wfo_st_probs = ct_vals[wfo_st == wfo_st_1d]

        # features
        # all
        otlk_timestamp.append(bdt.strftime('%Y%m%d%H'))
        wfo_st_list.append(wfo_st)
        
        maxct.append(np.max(wfo_st_probs))
        medct.append(np.median(wfo_st_probs))
        
        maxhourct.append(np.mean(wfo_st_probs,axis=0).max())
        medhourct.append(np.median(np.mean(wfo_st_probs,axis=0)))

        hourofmax_sin.append(math.sin(((bdt.hour + np.argmax(np.mean(wfo_st_probs,axis=0))) % 24)/24))
        hourofmax_cos.append(math.cos(((bdt.hour + np.argmax(np.mean(wfo_st_probs,axis=0))) % 24)/24))

        sumct.append(wfo_st_probs.sum())
        sumhourct.append(np.mean(wfo_st_probs,axis=0).sum())
        sumpopct.append((wfo_st_probs*population[wfo_st == wfo_state_2d][:, np.newaxis]).sum())
        sumhourpopct.append(np.mean(wfo_st_probs*population[wfo_st == wfo_state_2d][:, np.newaxis],axis=0).sum())

        maxpopct.append((wfo_st_probs*population[wfo_st == wfo_state_2d][:, np.newaxis]).max())
        medpopct.append(np.median(wfo_st_probs*population[wfo_st == wfo_state_2d][:, np.newaxis]))
        maxhourpopct.append(np.mean(wfo_st_probs*population[wfo_st == wfo_state_2d][:, np.newaxis],axis=0).max())
        medhourpopct.append(np.median(np.mean(wfo_st_probs*population[wfo_st == wfo_state_2d][:, np.newaxis],axis=0)))

    df_ctfeatures = pd.DataFrame([
        otlk_timestamp,wfo_st_list,
        maxct,medct,
        maxhourct,medhourct,
        hourofmax_sin,hourofmax_cos,
        sumct,sumhourct,sumpopct,sumhourpopct,
        maxpopct,medpopct,maxhourpopct,medhourpopct
    ]).T

    return df_ctfeatures

def find_dist(val,haz,size):

    val = int(val)

    if val < 6:
        dist2use = dists_final[haz][size][str(val)]
    elif val < 10:
        dist2use = dists_final[haz][size]['6-9']
    elif val < 20:
        dist2use = dists_final[haz][size]['10-19']
    elif val < 30:
        dist2use = dists_final[haz][size]['20-29']
    elif val < 50:
        dist2use = dists_final[haz][size]['30-49']
    elif val < 75:
        dist2use = dists_final[haz][size]['50-74']
    elif val < 100:
        dist2use = dists_final[haz][size]['75-99']
    else:
        dist2use = dists_final[haz][size]['>99']

    return dist2use.rvs(fv.nsims).astype(int)


dists_final = {
    'hail': {
        'national' : {
            '0': stats.loggamma(c=7.41226e-08, loc=1.16108e-06, scale=6.82163e-08),
            '1': stats.genextreme(c=1.2064, loc=-0.154491, scale=1.39278),
            '2': stats.genextreme(c=1.13339, loc=-1.10412, scale=3.51819),
            '3': stats.genextreme(c=1.1102, loc=-0.426044, scale=3.8036),
            '4': stats.genextreme(c=1.1618, loc=-0.078998, scale=4.73899),
            '5': stats.genextreme(c=1.2553, loc=0.760249, scale=5.32216),
            '6-9': stats.genextreme(c=0.817799, loc=-3.23174, scale=10.0797),
            '10-19': stats.loggamma(c=0.304106, loc=12.8153, scale=4.20752),
            '20-29': stats.loggamma(c=0.232072, loc=18.2721, scale=4.92087),
            '30-49': stats.t(df=4.08899, loc=1.15502, scale=17.6415),
            '50-74': stats.loggamma(c=1.03948, loc=8.25854, scale=29.3182),
            '75-99': stats.t(df=1.60133, loc=7.50007, scale=16.1067),
            '>99': stats.loggamma(c=1.02956, loc=9.51261, scale=60.9035),
        }
    },
    'wind': {
        'national': {
            '0': stats.loggamma(c=2.44261e-07, loc=1.7492e-06, scale=9.19515e-08),
            '1': stats.genextreme(c=1.14585, loc=-0.277369, scale=1.46367),
            '2': stats.loggamma(c=6.36961e-08, loc=2.00001, scale=1.3902e-07),
            '3': stats.loggamma(c=1.5396e-07, loc=3.00001, scale=3.00359e-07),
            '4': stats.loggamma(c=5.80831e-07, loc=4.00003, scale=1.00534e-06),
            '5': stats.beta(a=52.6253, b=0.686944, loc=-132.7, scale=137.7),
            '6-9': stats.beta(a=97.2469, b=0.97055, loc=-820.753, scale=829.753),
            '10-19': stats.loggamma(c=0.758362, loc=8.6464, scale=5.63071),
            '20-29': stats.loggamma(c=0.502779, loc=15.2561, scale=6.98456),
            '30-49': stats.loggamma(c=2.01773, loc=-6.57736, scale=24.0644),
            '50-74': stats.t(df=6.2996, loc=4.64933, scale=20.6239),
            '75-99': stats.t(df=709.662, loc=7.66889, scale=35.5976),
            '>99': stats.loggamma(c=5.71804, loc=-232.157, scale=134.196)
        }
    },
}

def add_distribution(df,haz='hail'):
    """ Add distribution columns to the dataframe """

    dist = df[haz] - find_dist(df[haz],haz,'national')

    dist[dist < 0] = 0  

    return dist.astype(int)

def return_mag(lev,month,wfo,hazard):
    if lev == 0:
        if month in [12,1,2]:
            season = 'Winter'
        elif month in [3,4,5]:
            season = 'Spring'
        elif month in [6,7,8]:
            season = 'Summer'
        elif month in [9,10,11]:
            season = 'Fall'
    
        if wfo in ["EKA", "LOX", "STO", "SGX", "MTR", "HNX","MFR", "PDT", "PQR","SEW", "OTX","REV", "VEF", "LKN","PSR", "TWC", "FGZ",
                    "SLC","BOI", "PIH","MSO","TFX","RIW","GJT","EPZ"]:
            region = 'West'
        elif wfo in ["BOU", "PUB","CYS","BYZ", "GGW","BIS","ABR", "UNR","LBF", "GID","GLD", "DDC","AMA", "LUB", "MAF","ABQ"]:
            region = 'High Plains'
        elif wfo in ["OUN", "TSA","ICT","FWD", "SJT", "EWX"]:
            region = 'Plains'
        elif wfo in ["FSD","DLH", "MPX", "FGF","OAX","DMX", "DVN","TOP", "EAX", "SGF", "LSX","LOT", "ILX","IND", "IWX",
                    "DTX", "APX", "GRR", "MQT","GRB", "ARX", "MKX"]:
            region = 'Midwest'
        elif wfo in ["BMX", "HUN", "MOB","JAN","FFC","CHS", "CAE", "GSP","JAX", "KEY", "MLB", "MFL", "TAE", "TBW","MEG", "MRX", "OHX",
                    "LMK", "JKL", "PAH","LCH", "LIX", "SHV","LZK","HGX", "CRP", "BRO"]:
            region = 'Southeast'
        elif wfo in ["PHI","PBZ", "CTP","RLX","LWX", "RNK", "AKQ","MHX", "RAH", "ILM","CLE", "ILN","CAR", "GYX","BOX",
                    "ALY", "BGM", "BUF", "OKX","BTV"]:
            region = 'Northeast'
        else:
            region = 'Other'
    
        proportions = {'wind': {'West': {'Winter': 0.09523809523809523,
                        'Spring': 0.032490974729241874,
                        'Summer': 0.049143708116157855,
                        'Fall': 0.07954545454545454},
                        'High Plains': {'Winter': 0.0392156862745098,
                        'Spring': 0.160075329566855,
                        'Summer': 0.1625668449197861,
                        'Fall': 0.15789473684210525},
                        'Plains': {'Winter': 0.12017167381974249,
                        'Spring': 0.1680933852140078,
                        'Summer': 0.16076447442383363,
                        'Fall': 0.14008941877794337},
                        'Midwest': {'Winter': 0.16909620991253643,
                        'Spring': 0.09427860696517414,
                        'Summer': 0.08757637474541752,
                        'Fall': 0.0836092715231788},
                        'Southeast': {'Winter': 0.056451612903225805,
                        'Spring': 0.05987093690248566,
                        'Summer': 0.03495624425856984,
                        'Fall': 0.037642397226349676},
                        'Northeast': {'Winter': 0.013916500994035786,
                        'Spring': 0.016916780354706683,
                        'Summer': 0.019506098022877054,
                        'Fall': 0.0076481835564053535},
                        'Other': {'Winter': 0.082,
                        'Spring': 0.0886,
                        'Summer': 0.085,
                        'Fall': 0.084}},
                        'hail': {'West': {'Winter': 0.0,
                        'Spring': 0.0,
                        'Summer': 0.036036036036036036,
                        'Fall': 0.030534351145038167},
                        'High Plains': {'Winter': 0.06557377049180328,
                        'Spring': 0.07942583732057416,
                        'Summer': 0.10298507462686567,
                        'Fall': 0.04953560371517028},
                        'Plains': {'Winter': 0.0410958904109589,
                        'Spring': 0.07186678352322524,
                        'Summer': 0.10559006211180125,
                        'Fall': 0.07168458781362007},
                        'Midwest': {'Winter': 0.045146726862302484,
                        'Spring': 0.04868603042876902,
                        'Summer': 0.09302325581395349,
                        'Fall': 0.07397003745318352},
                        'Southeast': {'Winter': 0.06479113384484228,
                        'Spring': 0.07476212052560036,
                        'Summer': 0.05352363960749331,
                        'Fall': 0.0603448275862069},
                        'Northeast': {'Winter': 0.09090909090909091,
                        'Spring': 0.03322995126273815,
                        'Summer': 0.04234769687964339,
                        'Fall': 0.010869565217391304},
                        'Other': {'Winter': 0.056,
                        'Spring': 0.056,
                        'Summer': 0.073,
                        'Fall': 0.056
                        }}}

        sig_pct = proportions[hazard][region][season]

        return np.random.choice([0,1], size=1, replace=True, p=[1-sig_pct,sig_pct])[0]
        
    else:
        return np.random.choice([0,1], size=1, replace=True, p=fv.cig_dists[hazard][str(lev)])[0]
        