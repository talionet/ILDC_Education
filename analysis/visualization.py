import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy import ndimage
import numpy.ma as ma
from settings import *

def simple_df_plot(DataFrame, columns='all', kind = 'line', is_legend = True, is_subplots = True,
                   reset_index = True, figsize=(15,10), secondary_y=False,
                   title=None,  save_name='temp.png', xlabel='', ylabel='', OVERRIDE=False):
    if is_subplots:
        #is_legend=False
        data=DataFrame.copy()
    if reset_index:
        data=DataFrame.reset_index(drop=True)
    if len(columns)==2 and secondary_y:
        figsize = (20, 5)
        data[columns[0]].plot(subplots=is_subplots, kind=kind, figsize=figsize, legend=True)
        data[columns[1]].plot(subplots=is_subplots, color='red', kind=kind, figsize=figsize, secondary_y=True, legend=True)



    elif columns=='all':
        data.plot(subplots=is_subplots, kind=kind, figsize=figsize, secondary_y=secondary_y, legend=is_legend)
        #if is_legend:
         #   plt.legend(columns,prop={'size': 14})

    else:
        data[columns].plot(subplots=is_subplots, kind=kind, figsize=figsize, secondary_y=secondary_y, legend=is_legend)
        #if is_legend:
         #   plt.legend(columns,prop={'size': 14})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.suptitle(title)
    if OVERRIDE:
        plt.savefig(os.path.join(OUTPUT_DIR, save_name))
        plt.close()

def boxplot(DataFrame, column='adjusted_is_correct_at_first_attempt', by_col='q_f_activity_type', figsize=(15,10),save_name='temp_boxplot.png', OVERRIDE=False):
    DataFrame.boxplot(column=[column],by=by_col,figsize=figsize, vert=False, fontsize=12)
    if OVERRIDE:
        save_name='box_%s_VS_%s.png' %(column,by_col)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, save_name))
        plt.close()

def grid_heatmap(DataFrame,index1=[],index2=[], title='',grid_index=[], cmap='YlGnBu', save_name='temp_heatmap.png'):
    if len(index1)>0 and len(index2)>0:
        l=5
        grid_size=(l+2, l+2)
        plt.figure(figsize=(20,20))
        axv1 = plt.subplot2grid(grid_size, (0, 0), rowspan=l)
        axv2 = plt.subplot2grid(grid_size, (0, 1), rowspan=l)
        ax_main = plt.subplot2grid(grid_size, (0, 2), rowspan=l,colspan=l)
        axh1 = plt.subplot2grid(grid_size, (l, 2), colspan=l)
        axh2 = plt.subplot2grid(grid_size, (l+1, 2), colspan=l)

        #plt.savefig(os.path.join(OUTPUT_DIR, 'temp_grid.png'))

        index1.plot(ax=axh1)
        index2.plot(ax=axh2)

        DataFrame.fillna(-1,inplace=True)
        my_cmap = plt.cm.get_cmap(cmap)
        my_cmap.set_under('w')
        ax_main.pcolor(DataFrame,vmin=0,cmap=my_cmap)
        '''plt.xticks(range(len(grid_index)), grid_index)
        plt.autoscale(tight=True)
        plt.grid()
        #plt.set_xticklabels([str(s) for s in DataFrame.index])
        #plt.yticks(DataFrame.index)'''
        plt.suptitle(title,fontsize=16)
        plt.savefig(os.path.join(OUTPUT_DIR, save_name))
        plt.close()