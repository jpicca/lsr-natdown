import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cbook as cbook

import numpy as np
import fipsvars as fv

def make_plots(all_preds,otlk_ts,outdir, haz_type, only_nat=False):
    """
    Function to create plots based on the provided DataFrame.
    Parameters:
    - all_preds: DataFrame containing predictions with columns for 'wfost', 'wind',
                 'hail', 'sigwind', 'sighail', and their respective distributions.
    Returns:
    - None: Writes out the plots.
    """
    
    affected_wfos = np.unique(all_preds.wfost.str.slice(0,3))
    affected_states = np.unique(all_preds.wfost.str.slice(3,5))

    if not only_nat:

        # For later (when we start to break down by WFOs and States)
        # Plot WFOs
        for affected in affected_wfos:
            affected_df = all_preds[all_preds.wfost.str.slice(0,3) == affected]

            # affected_summed = affected_df.sum()
            make_images(affected, affected_df, otlk_ts, outdir, 'wfo')

        # # Plot States
        for affected in affected_states:
            affected_df = all_preds[all_preds.wfost.str.slice(3,5) == affected]
            # affected_summed = affected_df.sum()
            make_images(affected, affected_df, otlk_ts, outdir, 'state')

    # Plot National
    affected = 'National'
    affected_df = all_preds

    make_images(affected, affected_df, otlk_ts, outdir, haz_type)

def make_images(affected, affected_df, otlk_ts, outdir, haz_type, level='national'):

    fig = plt.figure(figsize=(6,10))
    gs = gridspec.GridSpec(8,8)

    fig.suptitle(f'Report Count Predictions ({haz_type}) - {affected} - Outlook {otlk_ts}',ha='left',x=0,weight='bold',size=10)

    # Make axes
    ax1 = fig.add_subplot(gs[4:,:])
    ax2 = fig.add_subplot(gs[:4,:])

    kw = dict(horizontalalignment="center",
            verticalalignment="center",
            fontsize=10,weight='bold')

    if haz_type == 'hail':
        countsup = [affected_df.sum().hail_dists,
                                    affected_df.sum().sighail_dists]
        starter_list = []
        for thresh in [0,4,9,19,49]:
            starter_list.append([int(np.floor(np.sum(np.array(countlist) > thresh)/(fv.nsims/100))) for countlist in countsup])

        heatmap_list = np.array(starter_list).T

        ax1.imshow(heatmap_list,cmap='Greens',vmin=0,vmax=100)
        ax1.set_title('Hail Report Count Exceedance Probability',loc='left',size=10,weight='bold',y=1.13)

        ax2.set_title('Hail Report Count Distributions',loc='left',weight='bold',size=10)
        ax2.set_yticklabels(['1+"','2+"'])
    else:
        countsup = [affected_df.sum().wind_dists,
                                    affected_df.sum().sigwind_dists]

        starter_list = []
        for thresh in [0,4,9,19,49]:
            starter_list.append([int(np.floor(np.sum(np.array(countlist) > thresh)/(fv.nsims/100))) for countlist in countsup])

        heatmap_list = np.array(starter_list).T

        ax1.imshow(heatmap_list,cmap='Blues',vmin=0,vmax=100)
        ax1.set_title('Wind Report Count Exceedance Probability',loc='left',size=10,weight='bold',y=1.13)

        ax2.set_title('Wind Report Count Distributions',loc='left',weight='bold',size=10)
        ax2.set_yticklabels(['50+ kt','65+ kt'])

    ax1.spines[:].set_visible(False)
    ax1.set_xticks(np.arange(heatmap_list.shape[1])+.5, minor=True)
    ax1.set_xticks(np.arange(heatmap_list.shape[1]), labels=['1+', '5+', '10+', '20+', '50+'])
    ax1.set_yticks(np.arange(-.5, 1.5, 1), minor=True)

    if haz_type == 'hail':
        ax1.set_yticks(np.arange(heatmap_list.shape[0]), labels=['1+"','2+"'])
    else:
        ax1.set_yticks(np.arange(heatmap_list.shape[0]), labels=['50+ kt','65+ kt'])
    
    ax1.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax1.tick_params(which="minor", bottom=False, left=False, right=False)
    ax1.tick_params(labelsize=10,length=0)
    ax1.set_ylim([-0.5,1.5])

    for i in range(heatmap_list.shape[0]):
        for j in range(heatmap_list.shape[1]):
            kw.update(color=['black','white'][int(heatmap_list[i, j] > 60)])
            if heatmap_list[i, j] == 0:
                matrix_val = '<1'
            elif heatmap_list[i, j] >= 99:
                matrix_val = '>99'
            else:
                matrix_val = f'{heatmap_list[i, j]}'
            text = ax1.axes.text(j, i, f'{matrix_val}%', **kw)


    ax2.set_ylim([0,2])
    

    box_list_counts = cbook.boxplot_stats(countsup)

    for i in range(0,len(countsup)):
        box_list_counts[i]['whislo'],box_list_counts[i]['q1'], box_list_counts[i]['q3'], box_list_counts[i]['whishi'] = np.percentile(countsup[i],[5,25,75,95])
        
    ### **** plotting and text for max on wind counts ***
    ax2.plot([box_list_counts[0]['whishi'],np.max(countsup[0])], [0.5,0.5], color='k', linestyle='dashed', linewidth=0.25)
    ax2.text(int(np.max(countsup[0])),0.5,f'{int(np.max(countsup[0]))}',ha='left',va='center',fontsize=7)
    ax2.plot([box_list_counts[1]['whishi'],np.max(countsup[1])], [1.5,1.5], color='k', linestyle='dashed', linewidth=0.25)    
    ax2.text(int(np.max(countsup[1])),1.5,f'{int(np.max(countsup[1]))}',ha='left',va='center',fontsize=7)

    box_counts = ax2.bxp(box_list_counts,vert=False,showfliers=False, positions=[0.5,1.5],
                widths=0.15,showcaps=False,patch_artist=True,
                whiskerprops=dict(alpha=0))

    for idx,whisker in enumerate(box_counts['whiskers']):
        whisker.set_linewidth(4.5)
        if idx % 2 == 0:
            whisker.set_alpha(0.4)
        else:
            whisker.set_alpha(0.4)

    ### **** text for medians on wind counts ***
    for idx,median in enumerate(box_counts['medians']):
        text = median.get_xdata()[0]
        ax2.text(int(text),idx+0.65,f'{int(text)}',ha='center',fontsize=9)

    ### **** text for 95% on wind counts ***
    for idx,worst in enumerate(box_counts['whiskers']):
        text = worst.get_xdata()[1]
        worst_x_off = int(text) / 5
        if idx % 2 == 1:
            ax2.text(int(text),int(idx/2)+0.55,f'{int(text)}',ha='left',va='center',fontsize=7)

    ax2.spines[:].set_visible(False)
    ax2.set_yticks([0.5,1.5])
    ax2.tick_params(labelsize=10,length=0,axis='y')
    ax2.tick_params(labelsize=8,axis='x')
    ax2.set_xscale('symlog')
    ax2.set_xlim([0,600])
    ax2.minorticks_off()
    ax2.spines['bottom'].set_position(('data', 0.0))

    ax2.set_xticks([0,1,5,10,25,50,100,200,500])
    ax2.set_xticklabels(['0','1','5','10','25','50','100','200','500'])
    ax2.grid(axis = 'x', alpha=0.3)

    plt.setp(box_counts['boxes'],facecolor='black')
    plt.setp(box_counts['medians'],linewidth=2,color='white')

    gs.tight_layout(fig)
    fig.savefig(f'{outdir}/{affected}-{otlk_ts}-{haz_type}.png',dpi=150)

    plt.close(fig)