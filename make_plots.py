# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.cbook as cbook

# import datetime as dt

# import numpy as np
# import fipsvars as fv

# import json
# from pdb import set_trace as st

# def make_plots(reports_df,otlk_ts,outdir, haz_type, 
#                affected_wfos, affected_states, only_nat=False):
#     """
#     Function to create plots based on the provided DataFrame.
#     Parameters:
#     - all_preds: DataFrame containing predictions with columns for 'wfo', 'wind',
#                  'hail', 'sigwind', 'sighail', and their respective distributions.
#     Returns:
#     - None: Writes out the plots.
#     """
    
#     # affected_wfos = reports_df.wfo.unique()
#     # affected_states = reports_df.st.unique()

#     # Dictionary to store all percentile data
#     percentile_data = {}

#     if not only_nat:

#         # For later (when we start to break down by WFOs and States)
#         # Plot WFOs
#         for affected in affected_wfos:

#             if affected == '0':
#                 continue

#             affected_df = reports_df[reports_df.wfo == affected]

#             affected_reg_dist = affected_df.groupby('sim').count()['cig'].reindex(index=np.arange(1,fv.nsims+1,1),fill_value=0).values
#             affected_sig_dist = affected_df.groupby('sim').sum()['sig'].reindex(index=np.arange(1,fv.nsims+1,1),fill_value=0).values

#             affected_lists = [affected_reg_dist, affected_sig_dist]

#             # affected_summed = affected_df.sum()
#             make_images(affected, affected_lists, otlk_ts, 
#                         outdir, haz_type, percentile_data, 'wfo')

#         # # Plot States
#         for affected in affected_states:

#             if affected == '0':
#                 continue

#             affected_df = reports_df[reports_df.st == affected]

#             affected_reg_dist = affected_df.groupby('sim').count()['cig'].reindex(index=np.arange(1,fv.nsims+1,1),fill_value=0).values
#             affected_sig_dist = affected_df.groupby('sim').sum()['sig'].reindex(index=np.arange(1,fv.nsims+1,1),fill_value=0).values

#             affected_lists = [affected_reg_dist, affected_sig_dist]

#             make_images(affected, affected_lists, otlk_ts,
#                         outdir, haz_type, percentile_data, 'state')

#     # Plot National
#     affected = 'National'
#     affected_reg_dist = reports_df.groupby('sim').count()['cig'].reindex(index=np.arange(1,fv.nsims+1,1),fill_value=0).values
#     affected_sig_dist = reports_df.groupby('sim').sum()['sig'].reindex(index=np.arange(1,fv.nsims+1,1),fill_value=0).values
#     affected_lists = [affected_reg_dist, affected_sig_dist]

#     make_images(affected, affected_lists, otlk_ts, outdir, haz_type, percentile_data)

#     # Write out JSON file
#     outdir_json = str(outdir).replace('/images/','/jsons/')

#     json_filename = f'{outdir_json}/{otlk_ts}-{haz_type}.json'
#     with open(json_filename, 'w') as f:
#         json.dump(percentile_data, f, indent=2)

# def make_images(affected, affected_lists, otlk_ts, outdir, haz_type, percentile_data=None, level='national'):

#     issue_time = dt.datetime.strptime(otlk_ts, '%Y%m%d%H%M%S')
#     exp_time = (issue_time + dt.timedelta(days=1)).replace(hour=12)

#     if issue_time.hour < 12:
#         valid_time = issue_time.replace(hour=12)
#     elif issue_time.hour < 14:
#         valid_time = issue_time.replace(hour=13)
#     elif issue_time.hour < 18:
#         valid_time = issue_time.replace(hour=16)
#     elif issue_time.hour < 21:
#         valid_time = issue_time.replace(hour=20)
#     else:
#         valid_time = issue_time.replace(hour=12)

#     # Calculate percentiles and store in dictionary
#     if percentile_data is not None:
#         reg_percentiles = np.percentile(affected_lists[0], [10, 25, 50, 75, 90])
#         sig_percentiles = np.percentile(affected_lists[1], [10, 25, 50, 75, 90])
        
#         percentile_data[affected] = {
#             'percentiles': {
#                 'reg': {
#                     'p10': int(reg_percentiles[0]),
#                     'p25': int(reg_percentiles[1]),
#                     'p50': int(reg_percentiles[2]),
#                     'p75': int(reg_percentiles[3]),
#                     'p90': int(reg_percentiles[4])
#                 },
#                 'sig': {
#                     'p10': int(sig_percentiles[0]),
#                     'p25': int(sig_percentiles[1]),
#                     'p50': int(sig_percentiles[2]),
#                     'p75': int(sig_percentiles[3]),
#                     'p90': int(sig_percentiles[4])
#                 }
#             }
#         }

#     fig = plt.figure(figsize=(6,10))
#     gs = gridspec.GridSpec(12,12)

#     if haz_type == 'hail':
#         title_haz = 'Hail'
#     else:
#         title_haz = 'Wind'

#     fig.text(0.5,0.97,f'{affected} - Predicted {title_haz} Reports (Final)',ha='center',weight='bold',size=16)
#     fig.text(0.5,0.945,f'Valid Period: {valid_time.strftime("%Y%m%d %HZ")} - {exp_time.strftime("%Y%m%d %HZ")}',
#              ha='center',size=14)

#     # Make axes
#     ax1 = fig.add_subplot(gs[1:7,:])
#     ax2 = fig.add_subplot(gs[8:10,:])
#     ax3 = fig.add_subplot(gs[10:,:])

#     kw = dict(horizontalalignment="center",
#             verticalalignment="center",
#             fontsize=14,weight='bold')

#     countsup = [affected_lists[0],affected_lists[1]]
#     starter_list = [[],[]]

#     if haz_type == 'hail':

#         threshes = [fv.thres_dict['hail'][level],
#                     fv.thres_dict['sighail'][level]]

#         # All hail
#         for thresh in threshes[0]:
#             starter_list[0].append(int(np.floor(np.sum(np.array(countsup[0]) >= thresh)/(fv.nsims/100))))
        
#         # Significant hail
#         for thresh in threshes[1]:
#             starter_list[1].append(int(np.floor(np.sum(np.array(countsup[1]) >= thresh)/(fv.nsims/100))))

#         ax2.imshow([starter_list[1]],cmap='Greens',vmin=0,vmax=100)
#         ax2.set_title('Hail Report Count Exceedance Probability',loc='left',size=12,weight='bold',y=1.13)

#         ax3.imshow([starter_list[0]],cmap='Greens',vmin=0,vmax=100)

#         ax1.set_title('Hail Report Count Distributions',loc='left',weight='bold',size=12)
#         ax1.set_yticklabels(['1+"','2+"'])

#     else:

#         threshes = [fv.thres_dict['hail'][level],
#                     fv.thres_dict['sighail'][level]]

#         # All wind
#         for thresh in threshes[0]:
#             starter_list[0].append(int(np.floor(np.sum(np.array(countsup[0]) >= thresh)/(fv.nsims/100))))
        
#         # Significant wind
#         for thresh in threshes[1]:
#             starter_list[1].append(int(np.floor(np.sum(np.array(countsup[1]) >= thresh)/(fv.nsims/100))))

#         ax2.imshow([starter_list[1]],cmap='Blues',vmin=0,vmax=100)
#         ax2.set_title('Wind Report Count Exceedance Probability',loc='left',size=12,weight='bold',y=1.13)

#         ax3.imshow([starter_list[0]],cmap='Blues',vmin=0,vmax=100)

#         ax1.set_title('Wind Report Count Distributions',loc='left',weight='bold',size=12)
#         ax1.set_yticklabels(['50+ kt','65+ kt'])

#     xlabels = [[f'{thresh}+' for thresh in threshes[0]],
#                 [f'{thresh}+' for thresh in threshes[1]]]

#     ax2.spines[:].set_visible(False)
#     ax2.set_xticks(np.arange(len(starter_list[1]))+.5, minor=True)
#     ax2.set_xticks(np.arange(len(starter_list[1])), labels=xlabels[1])
#     ax2.set_yticks([0], minor=True)

#     ax3.spines[:].set_visible(False)
#     ax3.set_xticks(np.arange(len(starter_list[1]))+.5, minor=True)
#     ax3.set_xticks(np.arange(len(starter_list[1])), labels=xlabels[0])
#     ax3.set_yticks([0], minor=True)

#     if haz_type == 'hail':
#         ax2.set_yticks([0], labels=['2+"'])
#         ax3.set_yticks([0], labels=['1+"'])
#     else:
#         ax2.set_yticks([0], labels=['65+ kt'])
#         ax3.set_yticks([0], labels=['50+ kt'])
    
#     ax2.grid(which="minor", color="w", linestyle='-', linewidth=1)
#     ax2.tick_params(which="minor", bottom=False, left=False, right=False)
#     ax2.tick_params(labelsize=14,length=0)
#     ax2.set_ylim([-0.5,0.5])

#     ax3.grid(which="minor", color="w", linestyle='-', linewidth=1)
#     ax3.tick_params(which="minor", bottom=False, left=False, right=False)
#     ax3.tick_params(labelsize=14,length=0)
#     ax3.set_ylim([-0.5,0.5])

#     for i in range(len(starter_list)):
#         for j in range(len(starter_list[0])):
#             kw.update(color=['black','white'][int(starter_list[i][j] > 60)])
#             if starter_list[i][j] == 0:
#                 matrix_val = '<1'
#             elif starter_list[i][j] >= 99:
#                 matrix_val = '>99'
#             else:
#                 matrix_val = f'{starter_list[i][j]}'

#             if i == 0:
#                 text = ax3.axes.text(j, 0, f'{matrix_val}%', **kw)
#             elif i == 1:
#                 text = ax2.axes.text(j, 0, f'{matrix_val}%', **kw)


#     ax1.set_ylim([0,2])
    
#     box_list_counts = cbook.boxplot_stats(countsup)

#     for i in range(0,len(countsup)):
#         box_list_counts[i]['whislo'],box_list_counts[i]['q1'], box_list_counts[i]['q3'], box_list_counts[i]['whishi'] = np.percentile(countsup[i],[10,25,75,90])

#     box_counts = ax1.bxp(box_list_counts,vert=False,showfliers=False, positions=[0.5,1.5],
#                 widths=0.15,showcaps=False,patch_artist=True,
#                 whiskerprops=dict(alpha=0))

#     for idx,whisker in enumerate(box_counts['whiskers']):
#         whisker.set_linewidth(4.5)
#         if idx % 2 == 0:
#             whisker.set_alpha(0.4)
#         else:
#             whisker.set_alpha(0.4)

#             # Add line from right whisker to max value
#             # idx % 2 == 1 means this is the right whisker
#             whisker_data = whisker.get_xdata()
#             whisker_end = whisker_data[1]  # Right end of whisker
#             y_position = whisker.get_ydata()[0]  # Y position of the whisker
            
#             # Get max value for this box (idx // 2 gives box index)
#             box_idx = idx // 2
#             fliers = box_list_counts[box_idx].get('fliers', [])
#             if len(fliers) > 0:
#                 max_value = max(box_list_counts[box_idx]['whishi'], max(fliers))
#             else:
#                 max_value = box_list_counts[box_idx]['whishi']
            
#             # Draw thin line from whisker to max
#             ax1.plot([whisker_end, max_value], [y_position, y_position], 
#                     color='black', linewidth=1, alpha=0.4)

#     ### **** text for medians on counts ***
#     for idx,median in enumerate(box_counts['medians']):
#         text = median.get_xdata()[0]
#         ax1.text(int(text),idx+0.65,f'{int(text)}',ha='center',fontsize=14,weight='bold')

#     ### **** text for 5/95% on counts ***
#     for idx,worst in enumerate(box_counts['whiskers']):
#         text = worst.get_xdata()[1]
#         worst_x_off = int(text) / 5
#         if idx % 2 == 1:
#             ax1.text(int(text),int(idx/2)+0.55,f'{int(text)}',ha='left',va='center',fontsize=12)
#         else:
#             ax1.text(int(text),int(idx/2)+0.55,f'{int(text)}',ha='right',va='center',fontsize=12)

#     ax1.spines[:].set_visible(False)
#     ax1.set_yticks([0.5,1.5])
#     ax1.tick_params(labelsize=14,length=0,pad=10,axis='y')
#     ax1.tick_params(labelsize=14,axis='x')
#     ax1.set_xscale('symlog')
#     if level == 'national':
#         ax1.set_xlim([0,1300])
#         ax1.set_xticks([0,1,5,10,25,50,100,200,500,1000])
#         ax1.set_xticklabels(['0','1','5','10','25','50','100','200','500','1k'])
#     elif level == 'state':
#         ax1.set_xlim([0,600])
#         ax1.set_xticks([0,1,5,10,25,50,100,200,500])
#         ax1.set_xticklabels(['0','1','5','10','25','50','100','200','500'])
#     elif level == 'wfo':
#         ax1.set_xlim([0,300])
#         ax1.set_xticks([0,1,5,10,25,50,100,250])
#         ax1.set_xticklabels(['0','1','5','10','25','50','100','250'])

#     ax1.minorticks_off()
#     ax1.spines['bottom'].set_position(('data', 0.0))

#     ax1.grid(axis = 'x', alpha=0.3)

#     plt.setp(box_counts['boxes'],facecolor='black')
#     plt.setp(box_counts['medians'],linewidth=2,color='white')

#     gs.tight_layout(fig)
#     fig.savefig(f'{outdir}/{affected}-{otlk_ts}-{haz_type}.png',dpi=150)

#     plt.close(fig)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cbook as cbook

import datetime as dt
import pandas as pd
import numpy as np
import fipsvars as fv

import json

# --- Global Cache for Historic Data ---
HISTORIC_DATA = {}

def load_historic_data(haz_type):
    """
    Loads the historic CSV data for the given hazard type once and caches it.
    Calculates the 'conv_date' (Convective Day) for grouping.
    """
    global HISTORIC_DATA
    if haz_type in HISTORIC_DATA:
        return HISTORIC_DATA[haz_type]

    filename = f'./assets/2015-2024_{haz_type}_with_wfo.csv'
    try:
        df = pd.read_csv(filename)
        # Parse datetime
        df['date_time'] = pd.to_datetime(df['date_time'])
        # Define Convective Day (12z - 12z)
        # Shift back 12 hours so that everything before 12:00 UTC belongs to the previous date
        df['conv_date'] = (df['date_time'] - pd.Timedelta(hours=12)).dt.date
        
        HISTORIC_DATA[haz_type] = df
        return df
    except Exception as e:
        print(f"Error loading historic data for {haz_type}: {e}")
        return None

def get_recurrence_text(haz_type, level, affected, count, is_sig):
    """
    Calculates the recurrence of a day with >= 'count' reports.
    Returns a formatted string like '(1/mo)' or '(2/wk)'.
    """
    # If the count is 0 (or negative), we don't display recurrence (it's effectively "Daily")
    if count <= 0:
        return ""

    df = load_historic_data(haz_type)
    if df is None:
        return ""

    # 1. Filter by Area
    if level == 'state':
        subset = df[df['st'] == affected]
    elif level == 'wfo':
        subset = df[df['wfo'] == affected]
    else:
        # National
        subset = df

    if subset.empty:
        return ""

    # 2. Filter by Magnitude (All vs Sig)
    # Hail: All >= 1.0", Sig >= 2.0"
    # Wind: All >= 50kt, Sig >= 65kt
    if haz_type == 'hail':
        thresh = 2.0 if is_sig else 1.0
    else: # wind
        thresh = 65.0 if is_sig else 50.0
    
    subset = subset[subset['mag'] >= thresh]

    # 3. Calculate Frequency
    # Count number of convective days with >= 'count' reports
    daily_counts = subset.groupby('conv_date').size()
    n_days_meeting_criteria = (daily_counts >= count).sum()

    # Dataset is 10 years (2015-2024)
    years = 10.0
    freq_per_year = n_days_meeting_criteria / years

    # 4. Format Text
    if freq_per_year > 10:
        return ">10/yr"
    elif freq_per_year >= 0.95: 
        # Rounds to at least 1/yr
        return f"{int(round(freq_per_year))}/yr"
    else:
        # Less than 1/yr, switch to /decade
        freq_per_decade = freq_per_year * 10
        if freq_per_decade >= 0.5:
             # Rounds to at least 1/decade
            return f"{int(round(freq_per_decade))}/decade"
        else:
            return "<1/decade"

# --------------------------------------

def make_plots(reports_df,otlk_ts,outdir, haz_type, 
               affected_wfos, affected_states, only_nat=False):
    """
    Function to create plots based on the provided DataFrame.
    """
    
    # Pre-load historic data to avoid doing it inside the loop
    load_historic_data(haz_type)

    # Dictionary to store all percentile data
    percentile_data = {}

    if not only_nat:

        # Plot WFOs
        for affected in affected_wfos:

            if affected == '0':
                continue

            affected_df = reports_df[reports_df.wfo == affected]

            affected_reg_dist = affected_df.groupby('sim').count()['cig'].reindex(index=np.arange(1,fv.nsims+1,1),fill_value=0).values
            affected_sig_dist = affected_df.groupby('sim').sum()['sig'].reindex(index=np.arange(1,fv.nsims+1,1),fill_value=0).values

            affected_lists = [affected_reg_dist, affected_sig_dist]

            make_images(affected, affected_lists, otlk_ts, 
                        outdir, haz_type, percentile_data, 'wfo')

        # Plot States
        for affected in affected_states:

            if affected == '0':
                continue

            affected_df = reports_df[reports_df.st == affected]

            affected_reg_dist = affected_df.groupby('sim').count()['cig'].reindex(index=np.arange(1,fv.nsims+1,1),fill_value=0).values
            affected_sig_dist = affected_df.groupby('sim').sum()['sig'].reindex(index=np.arange(1,fv.nsims+1,1),fill_value=0).values

            affected_lists = [affected_reg_dist, affected_sig_dist]

            make_images(affected, affected_lists, otlk_ts,
                        outdir, haz_type, percentile_data, 'state')

    # Plot National
    affected = 'National'
    affected_reg_dist = reports_df.groupby('sim').count()['cig'].reindex(index=np.arange(1,fv.nsims+1,1),fill_value=0).values
    affected_sig_dist = reports_df.groupby('sim').sum()['sig'].reindex(index=np.arange(1,fv.nsims+1,1),fill_value=0).values
    affected_lists = [affected_reg_dist, affected_sig_dist]

    make_images(affected, affected_lists, otlk_ts, outdir, haz_type, percentile_data, 'national')

    # Write out JSON file
    outdir_json = str(outdir).replace('/images/','/jsons/')

    json_filename = f'{outdir_json}/{otlk_ts}-{haz_type}.json'
    with open(json_filename, 'w') as f:
        json.dump(percentile_data, f, indent=2)

def make_images(affected, affected_lists, otlk_ts, outdir, haz_type, percentile_data=None, level='national'):

    issue_time = dt.datetime.strptime(otlk_ts, '%Y%m%d%H%M%S')
    exp_time = (issue_time + dt.timedelta(days=1)).replace(hour=12)

    if issue_time.hour < 12:
        valid_time = issue_time.replace(hour=12)
    elif issue_time.hour < 14:
        valid_time = issue_time.replace(hour=13)
    elif issue_time.hour < 18:
        valid_time = issue_time.replace(hour=16)
    elif issue_time.hour < 21:
        valid_time = issue_time.replace(hour=20)
    else:
        valid_time = issue_time.replace(hour=12)

    # Calculate percentiles and store in dictionary
    if percentile_data is not None:
        reg_percentiles = np.percentile(affected_lists[0], [10, 25, 50, 75, 90])
        sig_percentiles = np.percentile(affected_lists[1], [10, 25, 50, 75, 90])
        
        percentile_data[affected] = {
            'percentiles': {
                'reg': {
                    'p10': int(reg_percentiles[0]),
                    'p25': int(reg_percentiles[1]),
                    'p50': int(reg_percentiles[2]),
                    'p75': int(reg_percentiles[3]),
                    'p90': int(reg_percentiles[4])
                },
                'sig': {
                    'p10': int(sig_percentiles[0]),
                    'p25': int(sig_percentiles[1]),
                    'p50': int(sig_percentiles[2]),
                    'p75': int(sig_percentiles[3]),
                    'p90': int(sig_percentiles[4])
                }
            }
        }

    fig = plt.figure(figsize=(6,10))
    gs = gridspec.GridSpec(12,12)

    if haz_type == 'hail':
        title_haz = 'Hail'
    else:
        title_haz = 'Wind'

    fig.text(0.5,0.97,f'{affected} - Predicted {title_haz} Reports (Final)',ha='center',weight='bold',size=16)
    fig.text(0.5,0.945,f'Valid Period: {valid_time.strftime("%Y%m%d %HZ")} - {exp_time.strftime("%Y%m%d %HZ")}',
             ha='center',size=14)

    # Make axes
    ax1 = fig.add_subplot(gs[1:7,:])
    ax2 = fig.add_subplot(gs[8:10,:])
    ax3 = fig.add_subplot(gs[10:,:])

    kw = dict(horizontalalignment="center",
            verticalalignment="center",
            fontsize=14,weight='bold')

    countsup = [affected_lists[0],affected_lists[1]]
    starter_list = [[],[]]

    if haz_type == 'hail':

        threshes = [fv.thres_dict['hail'][level],
                    fv.thres_dict['sighail'][level]]

        # All hail
        for thresh in threshes[0]:
            starter_list[0].append(int(np.floor(np.sum(np.array(countsup[0]) >= thresh)/(fv.nsims/100))))
        
        # Significant hail
        for thresh in threshes[1]:
            starter_list[1].append(int(np.floor(np.sum(np.array(countsup[1]) >= thresh)/(fv.nsims/100))))

        ax2.imshow([starter_list[1]],cmap='Greens',vmin=0,vmax=100)
        ax2.set_title('Hail Report Count Exceedance Probability',loc='left',size=12,weight='bold',y=1.13)

        ax3.imshow([starter_list[0]],cmap='Greens',vmin=0,vmax=100)

        ax1.set_title('Hail Report Count Distributions',loc='left',weight='bold',size=12)
        ax1.set_yticklabels(['1+"','2+"'])

    else:

        threshes = [fv.thres_dict['hail'][level],
                    fv.thres_dict['sighail'][level]]

        # All wind
        for thresh in threshes[0]:
            starter_list[0].append(int(np.floor(np.sum(np.array(countsup[0]) >= thresh)/(fv.nsims/100))))
        
        # Significant wind
        for thresh in threshes[1]:
            starter_list[1].append(int(np.floor(np.sum(np.array(countsup[1]) >= thresh)/(fv.nsims/100))))

        ax2.imshow([starter_list[1]],cmap='Blues',vmin=0,vmax=100)
        ax2.set_title('Wind Report Count Exceedance Probability',loc='left',size=12,weight='bold',y=1.13)

        ax3.imshow([starter_list[0]],cmap='Blues',vmin=0,vmax=100)

        ax1.set_title('Wind Report Count Distributions',loc='left',weight='bold',size=12)
        ax1.set_yticklabels(['50+ kt','65+ kt'])

    xlabels = [[f'{thresh}+' for thresh in threshes[0]],
                [f'{thresh}+' for thresh in threshes[1]]]

    ax2.spines[:].set_visible(False)
    ax2.set_xticks(np.arange(len(starter_list[1]))+.5, minor=True)
    ax2.set_xticks(np.arange(len(starter_list[1])), labels=xlabels[1])
    ax2.set_yticks([0], minor=True)

    ax3.spines[:].set_visible(False)
    ax3.set_xticks(np.arange(len(starter_list[1]))+.5, minor=True)
    ax3.set_xticks(np.arange(len(starter_list[1])), labels=xlabels[0])
    ax3.set_yticks([0], minor=True)

    if haz_type == 'hail':
        ax2.set_yticks([0], labels=['2+"'])
        ax3.set_yticks([0], labels=['1+"'])
    else:
        ax2.set_yticks([0], labels=['65+ kt'])
        ax3.set_yticks([0], labels=['50+ kt'])
    
    ax2.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax2.tick_params(which="minor", bottom=False, left=False, right=False)
    ax2.tick_params(labelsize=14,length=0)
    ax2.set_ylim([-0.5,0.5])

    ax3.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax3.tick_params(which="minor", bottom=False, left=False, right=False)
    ax3.tick_params(labelsize=14,length=0)
    ax3.set_ylim([-0.5,0.5])

    for i in range(len(starter_list)):
        for j in range(len(starter_list[0])):
            kw.update(color=['black','white'][int(starter_list[i][j] > 60)])
            if starter_list[i][j] == 0:
                matrix_val = '<1'
            elif starter_list[i][j] >= 99:
                matrix_val = '>99'
            else:
                matrix_val = f'{starter_list[i][j]}'

            if i == 0:
                text = ax3.axes.text(j, 0, f'{matrix_val}%', **kw)
            elif i == 1:
                text = ax2.axes.text(j, 0, f'{matrix_val}%', **kw)


    ax1.set_ylim([0,2])
    
    box_list_counts = cbook.boxplot_stats(countsup)

    for i in range(0,len(countsup)):
        box_list_counts[i]['whislo'],box_list_counts[i]['q1'], box_list_counts[i]['q3'], box_list_counts[i]['whishi'] = np.percentile(countsup[i],[10,25,75,90])

    box_counts = ax1.bxp(box_list_counts,vert=False,showfliers=False, positions=[0.5,1.5],
                widths=0.15,showcaps=False,patch_artist=True,
                whiskerprops=dict(alpha=0))

    for idx,whisker in enumerate(box_counts['whiskers']):
        whisker.set_linewidth(4.5)
        if idx % 2 == 0:
            whisker.set_alpha(0.4)
        else:
            whisker.set_alpha(0.4)

            # Add line from right whisker to max value
            # idx % 2 == 1 means this is the right whisker
            whisker_data = whisker.get_xdata()
            whisker_end = whisker_data[1]  # Right end of whisker
            y_position = whisker.get_ydata()[0]  # Y position of the whisker
            
            # Get max value for this box (idx // 2 gives box index)
            box_idx = idx // 2
            fliers = box_list_counts[box_idx].get('fliers', [])
            if len(fliers) > 0:
                max_value = max(box_list_counts[box_idx]['whishi'], max(fliers))
            else:
                max_value = box_list_counts[box_idx]['whishi']
            
            # Draw thin line from whisker to max
            ax1.plot([whisker_end, max_value], [y_position, y_position], 
                    color='black', linewidth=1, alpha=0.4)

    ### **** text for medians on counts ***
    for idx,median in enumerate(box_counts['medians']):
        val = median.get_xdata()[0]
        # Calculate recurrence for the median value
        # idx=0 is All, idx=1 is Sig
        is_sig = (idx == 1)
        rec_text = get_recurrence_text(haz_type, level, affected, int(val), is_sig=is_sig)
        
        ax1.text(int(val), idx+0.65, f'{int(val)}', ha='center', fontsize=12, weight='bold')

        # Recurrence below (and smaller)
        if rec_text:
            ax1.text(int(val),idx+0.35,f'{rec_text}',ha='center',fontsize=11,weight='normal')
    
    ### **** text for 10/90% on counts ***
    for idx,worst in enumerate(box_counts['whiskers']):
        val = worst.get_xdata()[1]
        
        # Calculate recurrence
        # Whisker indices: 0,1 -> Box 0 (All); 2,3 -> Box 1 (Sig)
        is_sig = (idx // 2 == 1)

        if idx % 2 == 1: # Right whisker (90th percentile)
            ax1.text(int(val), int(idx/2)+0.55, f'{int(val)}', ha='left', va='center', fontsize=10)
        else: # Left whisker (10th percentile)
            ax1.text(int(val), int(idx/2)+0.55, f'{int(val)}', ha='right', va='center', fontsize=10)

    ax1.spines[:].set_visible(False)
    ax1.set_yticks([0.5,1.5])
    ax1.tick_params(labelsize=14,length=0,pad=10,axis='y')
    ax1.tick_params(labelsize=14,axis='x')
    ax1.set_xscale('symlog')
    if level == 'national':
        ax1.set_xlim([0,1300])
        ax1.set_xticks([0,1,5,10,25,50,100,200,500,1000])
        ax1.set_xticklabels(['0','1','5','10','25','50','100','200','500','1k'])
    elif level == 'state':
        ax1.set_xlim([0,600])
        ax1.set_xticks([0,1,5,10,25,50,100,200,500])
        ax1.set_xticklabels(['0','1','5','10','25','50','100','200','500'])
    elif level == 'wfo':
        ax1.set_xlim([0,300])
        ax1.set_xticks([0,1,5,10,25,50,100,250])
        ax1.set_xticklabels(['0','1','5','10','25','50','100','250'])

    ax1.minorticks_off()
    ax1.spines['bottom'].set_position(('data', 0.0))

    ax1.grid(axis = 'x', alpha=0.3)

    plt.setp(box_counts['boxes'],facecolor='black')
    plt.setp(box_counts['medians'],linewidth=2,color='white')

    gs.tight_layout(fig)
    fig.savefig(f'{outdir}/{affected}-{otlk_ts}-{haz_type}.png',dpi=150)

    plt.close(fig)