nsims = 5000
alpha_hail = 4.7875
alpha_wind = 4.8324

fipsToState = {
    0:"0",
    2:"AK",
    1:"AL",
    5:"AR",
    60:"AS",
    4:"AZ",
    6:"CA",
    8:"CO",
    9:"CT",
    11:"DC",
    10:"DE",
    12:"FL",
    13:"GA",
    66:"GU",
    15:"HI",
    19:"IA",
    16:"ID",
    17:"IL",
    18:"IN",
    20:"KS",
    21:"KY",
    22:"LA",
    25:"MA",
    24:"MD",
    23:"ME",
    26:"MI",
    27:"MN",
    29:"MO",
    28:"MS",
    30:"MT",
    37:"NC",    
    38:"ND",
    31:"NE",
    33:"NH",
    34:"NJ",
    35:"NM",
    32:"NV",
    36:"NY",
    39:"OH",
    40:"OK",
    41:"OR",
    42:"PA",
    72:"PR",
    44:"RI",
    45:"SC",
    46:"SD",
    47:"TN",
    48:"TX",
    49:"UT",
    51:"VA",
    78:"VI",
    50:"VT",
    53:"WA",
    55:"WI",
    54:"WV",
    56:"WY"
}

col_names_outlook = [
    'otlk_timestamp','wfo_list',
        'toy_sin','toy_cos','maxhaz','medhaz','minhaz','popwfo','maxtor','medtor','mintor',
        'areator', 'area2tor', 'area5tor', 'area10tor', 'area15tor', 'area30tor', 'area45tor', 'area60tor',
        'areacigtor', 'areacig0tor', 'areacig1tor', 'areacig2tor',
        'poptor', 'pop2tor', 'pop5tor', 'pop10tor', 'pop15tor', 'pop30tor', 'pop45tor', 'pop60tor',
        'popcig0tor', 'popcig1tor', 'popcig2tor',
        'poptordensity',
        'maxtor_nat','maxtor_nei','medtor_nei','mintor_nei',
        'maxhail','medhail','minhail',
        'areahail', 'area5hail', 'area15hail', 'area30hail', 'area45hail', 'area60hail',
        'areacighail', 'areacig0hail', 'areacig1hail', 'areacig2hail',
        'pophail', 'pop5hail', 'pop15hail', 'pop30hail', 'pop45hail', 'pop60hail',
        'popcig0hail', 'popcig1hail', 'popcig2hail',
        'pophaildensity',
        'maxhail_nat','maxhail_nei','medhail_nei','minhail_nei',
        'maxwind','medwind','minwind',
        'areawind', 'area5wind', 'area15wind', 'area30wind', 'area45wind', 'area60wind',
        'areacigwind', 'areacig0wind', 'areacig1wind', 'areacig2wind',
        'popwind', 'pop5wind', 'pop15wind', 'pop30wind', 'pop45wind', 'pop60wind',
        'popcig0wind', 'popcig1wind', 'popcig2wind',
        'popwinddensity',
        'maxwind_nat','maxwind_nei','medwind_nei','minwind_nei'
]

col_names_wind = ['maxwind', 'medwind', 'minwind', 'areawind', 'area5wind', 'area15wind',
       'area30wind', 'area45wind', 'area60wind', 'areacigwind', 'areacig0wind',
       'areacig1wind', 'areacig2wind', 'popwind', 'pop5wind', 'pop15wind',
       'pop30wind', 'pop45wind', 'pop60wind', 'popcig0wind', 'popcig1wind',
       'popcig2wind', 'popwinddensity', 'wfo', 'doy', 'maxct', 'medct',
       'maxhourct', 'medhourct', 'hourofmax_sin', 'hourofmax_cos', 'sumct',
       'sumhourct', 'sumpopct', 'sumhourpopct', 'maxpopct', 'medpopct',
       'maxhourpopct', 'medhourpopct']

col_names_hail = ['maxhail', 'medhail', 'minhail', 'areahail', 'area5hail', 'area15hail',
       'area30hail', 'area45hail', 'area60hail', 'areacighail', 'areacig0hail',
       'areacig1hail', 'areacig2hail', 'pophail', 'pop5hail', 'pop15hail',
       'pop30hail', 'pop45hail', 'pop60hail', 'popcig0hail', 'popcig1hail',
       'popcig2hail', 'pophaildensity', 'wfo', 'doy', 'maxct', 'medct',
       'maxhourct', 'medhourct', 'hourofmax_sin', 'hourofmax_cos', 'sumct',
       'sumhourct', 'sumpopct', 'sumhourpopct', 'maxpopct', 'medpopct',
       'maxhourpopct', 'medhourpopct']

col_names_href = [
        'otlk_timestamp','wfo_list',
        'maxct','medct',
        'maxhourct','medhourct',
        'hourofmax_sin','hourofmax_cos',
        'sumct','sumhourct','sumpopct','sumhourpopct',
        'maxpopct','medpopct','maxhourpopct','medhourpopct'
]

# 0 non-sig, 1 sig
mags = [0,1]
levs = [0,1,2,3]

cig_dists = {
    'wind': 
    {
        '0': [0.94,0.06],
        '1': [0.87,0.13],
        '2': [0.77,0.23],
        '3': [0.66,0.34]
    },
    'hail': 
    {
        '0': [0.92,0.08],
        '1': [0.83,0.17],
        '2': [0.75,0.25],
        # dummy
        '3': [0.65,0.35]
    }
}

thres_dict = {
    'hail': {
        'state': [1,2,3,10,20,50],
        'wfo': [1,2,3,5,10,30],
        'national': [1,10,20,50,100,250]
    },
    'sighail': {
        'state': [1,2,3,5,10,20],
        'wfo': [1,2,3,4,5,15],
        'national': [1,2,5,10,20,50]
    },
    'wind': {
        'state': [1,2,3,10,25,75],
        'wfo': [1,2,3,5,15,50],
        'national': [1,10,25,100,200,500]
    },
    'sigwind': {
        'state': [1,2,3,5,10,20],
        'wfo': [1,2,3,4,5,15],
        'national': [1,2,5,10,20,50]
    }
}