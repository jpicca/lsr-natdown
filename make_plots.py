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
    gs = gridspec.GridSpec(12,12)

    fig.suptitle(f'Report Count Predictions ({haz_type}) - {affected} - Outlook {otlk_ts}',ha='left',x=0,weight='bold',size=10)

    # Make axes
    # ax1 = fig.add_subplot(gs[4:,:])

    ax1 = fig.add_subplot(gs[:7,:])
    ax2 = fig.add_subplot(gs[8:10,:])
    ax3 = fig.add_subplot(gs[10:,:])

    kw = dict(horizontalalignment="center",
            verticalalignment="center",
            fontsize=10,weight='bold')

    if haz_type == 'hail':
        countsup = [affected_df.sum().hail_dists,
                                    affected_df.sum().sighail_dists]
        starter_list = [[],[]]
        # All hail
        for thresh in [1,10,20,50,100,250]:
            starter_list[0].append(int(np.floor(np.sum(np.array(countsup[0]) >= thresh)/(fv.nsims/100))))
        
        # Significant hail
        for thresh in [1,2,5,10,20,50]:
            starter_list[1].append(int(np.floor(np.sum(np.array(countsup[1]) >= thresh)/(fv.nsims/100))))

        ax2.imshow([starter_list[1]],cmap='Greens',vmin=0,vmax=100)
        ax2.set_title('Hail Report Count Exceedance Probability',loc='left',size=10,weight='bold',y=1.13)

        ax3.imshow([starter_list[0]],cmap='Greens',vmin=0,vmax=100)

        ax1.set_title('Hail Report Count Distributions',loc='left',weight='bold',size=10)
        ax1.set_yticklabels(['1+"','2+"'])

        xlabels = [['1+','10+','20+','50+','100+','250+'],
                  ['1+','2+','5+','10+','20+','50+']]

    else:
        countsup = [affected_df.sum().wind_dists,
                                    affected_df.sum().sigwind_dists]

        starter_list = [[],[]]
        # All wind
        for thresh in [1,10,25,100,200,500]:
            starter_list[0].append(int(np.floor(np.sum(np.array(countsup[0]) >= thresh)/(fv.nsims/100))))
        
        # Significant wind
        for thresh in [1,2,5,10,20,50]:
            starter_list[1].append(int(np.floor(np.sum(np.array(countsup[1]) >= thresh)/(fv.nsims/100))))

        ax2.imshow([starter_list[1]],cmap='Blues',vmin=0,vmax=100)
        ax2.set_title('Wind Report Count Exceedance Probability',loc='left',size=10,weight='bold',y=1.13)

        ax3.imshow([starter_list[0]],cmap='Blues',vmin=0,vmax=100)

        ax1.set_title('Wind Report Count Distributions',loc='left',weight='bold',size=10)
        ax1.set_yticklabels(['50+ kt','65+ kt'])

        xlabels = [['1+','10+','25+','100+','200+','500+'],
                  ['1+','2+','5+','10+','20+','50+']]

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
    ax2.tick_params(labelsize=10,length=0)
    ax2.set_ylim([-0.5,0.5])

    ax3.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax3.tick_params(which="minor", bottom=False, left=False, right=False)
    ax3.tick_params(labelsize=10,length=0)
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
        box_list_counts[i]['whislo'],box_list_counts[i]['q1'], box_list_counts[i]['q3'], box_list_counts[i]['whishi'] = np.percentile(countsup[i],[5,25,75,95])

    box_counts = ax1.bxp(box_list_counts,vert=False,showfliers=False, positions=[0.5,1.5],
                widths=0.15,showcaps=False,patch_artist=True,
                whiskerprops=dict(alpha=0))

    for idx,whisker in enumerate(box_counts['whiskers']):
        whisker.set_linewidth(4.5)
        if idx % 2 == 0:
            whisker.set_alpha(0.4)
        else:
            whisker.set_alpha(0.4)

    ### **** text for medians on counts ***
    for idx,median in enumerate(box_counts['medians']):
        text = median.get_xdata()[0]
        ax1.text(int(text),idx+0.65,f'{int(text)}',ha='center',fontsize=9)

    ### **** text for 5/95% on counts ***
    for idx,worst in enumerate(box_counts['whiskers']):
        text = worst.get_xdata()[1]
        worst_x_off = int(text) / 5
        if idx % 2 == 1:
            ax1.text(int(text),int(idx/2)+0.55,f'{int(text)}',ha='left',va='center',fontsize=7)
        else:
            ax1.text(int(text),int(idx/2)+0.55,f'{int(text)}',ha='right',va='center',fontsize=7)

    ax1.spines[:].set_visible(False)
    ax1.set_yticks([0.5,1.5])
    ax1.tick_params(labelsize=10,length=0,pad=10,axis='y')
    ax1.tick_params(labelsize=8,axis='x')
    ax1.set_xscale('symlog')
    ax1.set_xlim([0,600])
    ax1.minorticks_off()
    ax1.spines['bottom'].set_position(('data', 0.0))

    ax1.set_xticks([0,1,5,10,25,50,100,200,500])
    ax1.set_xticklabels(['0','1','5','10','25','50','100','200','500'])
    ax1.grid(axis = 'x', alpha=0.3)

    plt.setp(box_counts['boxes'],facecolor='black')
    plt.setp(box_counts['medians'],linewidth=2,color='white')

    gs.tight_layout(fig)
    fig.savefig(f'{outdir}/{affected}-{otlk_ts}-{haz_type}.png',dpi=150)

    plt.close(fig)