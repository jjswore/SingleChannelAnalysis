import re
import matplotlib.patches as patches
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter, FuncFormatter

def Plot_All_EAGs(df):
    print('building df')
    fig, axes = plt.subplots(nrows=8, ncols=3, sharey=True,figsize=(28, 28))
    Conc=['100','1k','10k']
    print('building plots')
    for x,label in zip(range(len(df['label'].unique())),df['label'].unique()):
        axes[x,0].set_ylabel(label, weight='bold', size=26)
        for y, conc in zip(range(len(Conc)),Conc):
            axes[x,y].spines['top'].set_visible(False)
            axes[x,y].spines['right'].set_visible(False)
            axes[0,y].set_title(conc, weight='bold', size=26)
            plt.ylim(-150,50)
            plt.xlim(-5,8000)
    print('displaying plots')
    for x, label in zip(range(len(df['label'].unique())),df['label'].unique()):
        for y, conc in zip(range(5),Conc):
            temp = df[(df['label'] == label) & (df['concentration'] == conc)]
            for _, row in temp.iterrows():
                print(f'plotting {label} and {conc}')
                row.drop(['label', 'concentration', 'date']).plot(ax=axes[x,y])

    #plt.savefig('Raw_BF.1_6_AllSingleSiteWaves.svg')
    #plt.savefig('Raw_BF.1_6_AllSingleSiteWaves.jpg')
    plt.show()
#if __name__ == "__main__":
#    Plot_All_EAGs()

#'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/'
#                     'ControlSubtracted/LimMin/Butterworth_Optimized_Filter/LimMin_finalDF.csv'
def Find_Prospective_Waves(CSV, Odor, Conc):
    DF = pd.read_csv(CSV, index_col=0)
    for file in DF.index:
        if Odor in file:
            if Conc in file:
                DF.T[file][:-3].plot()
                plt.title(file)
                #plt.ylim()
                plt.xlim(0, 6000)
                plt.ylim(-1,.45)
                plt.show()
                print(file)
                ans = input('does the file look good?')
                if ans.lower() != 'yes':
                    continue
                else:
                    break
def EAG_All_Concentrations_Plot(file, waves):
    file ='/Users/joshswore/PycharmProjects/SingleChannelAnalysis/' \
          'Data/ControlSubtracted/Normalized/BF.1_2_/' \
          'Dataframes/QualityControlled/_QC_T_1.csv'

    #Find_Prospective_Waves(file, 'lemonoil', '100')



    DF = pd.read_csv(file,
                     index_col=0).T
    DF=DF.iloc[:,:-3]
    #DF.index = DF.index.astype(float) / 1000

    #10k
    Limonene10k = DF['090122m2a110klimonene0000wave1'][:-3]
    LemonOil10k = DF['082322m1a110klemonoil0001wave0'][:-3]
    MineralOil10k = DF['072822m1a110kmineraloil0008wave0'][:-3]

    #1k
    Limonene1k = DF['080422m1a11klimonene0001wave0'][:-3]
    LemonOil1k = DF['082222m2a11klemonoil0000wave1'][:-3]
    MineralOil1k = DF['080522m1a11kmineraloil0005wave0'][:-3]

    #100
    Limonene100 = DF['082222m2a1100limonene0000wave1'][:-3]
    LemonOil100 = DF['082222m1a1100lemonoil0001wave1'][:-3]
    MineralOil100 = DF['080422m1a1100mineraloil0004wave2'][:-3]


    colors = plt.cm.Paired(np.linspace(0, 1, 8))

    label_color_dict = {
            'limonene': [colors[0], 'o'],
            'lemonoil': [colors[1], 'o'],
            'ylangylang':[ colors[2], 'o'],
            'roseoil': [colors[3], 's'],
            'benzylalcohol': [colors[4],'s'],
            '1octen3ol': [colors[5], '^'],
            'benzaldehyde': [colors[6], 'o'],
            'linalool': [colors[7], '^'],
            'mineraloil': ['grey', 'P'],}

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10), sharey=True)
    fig.subplots_adjust(wspace=0)
    fig.text(0.5, 0.04, 'Time (s)', ha='center', va='center', fontsize=20)
    fig.set_facecolor('white')

    for i in range(3):
        ax[i].set_facecolor('white')
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        if i > 0:  # Only for the second and third subplots
            # Hide the tick labels for y-axis instead of removing the ticks
            for label in ax[i].get_yticklabels():
                label.set_visible(False)
            ax[i].spines['left'].set_visible(False)
            ax[i].tick_params(axis='y', colors='none')

    ax[0].set_ylabel('Normalized Response', fontsize=25)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax[0].set_ylim(-.4,.4)

    Limonene10k.plot(ax=ax[0], color = label_color_dict.get('limonene')[0])
    LemonOil10k.plot(ax=ax[0], color= label_color_dict.get('lemonoil')[0])
    MineralOil10k.plot(ax=ax[0], color= label_color_dict.get('mineraloil')[0])

    Limonene1k.plot(ax=ax[1], color = label_color_dict.get('limonene')[0])
    LemonOil1k.plot(ax=ax[1], color= label_color_dict.get('lemonoil')[0])
    MineralOil1k.plot(ax=ax[1], color= label_color_dict.get('mineraloil')[0])

    Limonene100.plot(ax=ax[2], color = label_color_dict.get('limonene')[0])
    LemonOil100.plot(ax=ax[2], color = label_color_dict.get('lemonoil')[0])
    MineralOil100.plot(ax=ax[2], color = label_color_dict.get('mineraloil')[0])

    #plt.savefig('/Users/joshswore/PycharmProjects/'
    #            'SingleChannelAnalysis/EAG_WAVE_Plots/LimLoMin1k.svg')
    plt.show()


#def EAG_1_Conc_Plot(file, waves):

#file ='/Users/joshswore/PycharmProjects/SingleChannelAnalysis/' \
#          'Data/ControlSubtracted/Normalized/BF.1_2_/' \
#          'Dataframes/QualityControlled/_QC_T_1.csv'

def Plot_Comparative_EAGS(file, EAGS, SAVEDIR=None):


    DF = pd.read_csv(file, index_col=0).T


    # Find_Prospective_Waves(file, 'lemonoil', '100')

    DF = DF.iloc[:-3, :]
    DF.index = pd.to_numeric(DF.index)
    DF.index = DF.index.astype(float) / 1000
    #print(type(DF.index[0]))

    colors = plt.cm.Paired(np.linspace(0, 1, 8))

    label_color_dict = {
        'limonene': [colors[0], 'o'],
        'lemonoil': [colors[1], 'o'],
        'ylangylang': [colors[2], 'o'],
        'roseoil': [colors[3], 's'],
        'benzylalcohol': [colors[4], 's'],
        '1octen3ol': [colors[5], '^'],
        'benzaldehyde': [colors[6], 'o'],
        'linalool': [colors[7], '^'],
        'mineraloil': ['grey', 'P'], }

    name_map = {
        'ylangylang': 'Ylang Ylang',
        'benzylalcohol': 'Benzylalcohol',
        'benzaldehyde': 'Benzaldehyde',
        '1octen3ol': '1-Octen-3-ol',
        'roseoil': 'Rose Oil',
        'lemonoil': 'Lemon Oil',
        'limonene': 'Limonene',
        'linalool': 'Linalool',
        'mineraloil': 'Mineral Oil (Ctrl)'
    }

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.set_facecolor('white')

    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')

    ax.set_ylabel('Normalized Response', fontsize=25)
    ax.set_xlabel('Time (s)', fontsize=25)
    #ax.set_ylim(-.21, .1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plotted_labels = []

    for n in range(len(EAGS)):
        text = EAGS[n]  # Replace with your text

        pattern = r'(1k|10k|100)(.*?)(000)'

        match = re.search(pattern, text)
        odor = match.group(2)
        print(odor)
        TRACE = DF[EAGS[n]]
        plotted_labels.append(name_map.get(odor))
        TRACE.plot(color=label_color_dict.get(odor)[0], linewidth=3)

    #ax.legend(plotted_labels, markerscale=1.5, fontsize=20, frameon=False)
    if SAVEDIR != None:
        plt.savefig(f'{SAVEDIR}.jpg')
        plt.savefig(f'{SAVEDIR}.svg', transparent=True)
    plt.show()
def Plot_Comparative_EAGS_subplot(file, EAGS, SAVEDIR=None):


    #DF = pd.read_csv(file, index_col=0).T

    DF=file
    # Find_Prospective_Waves(file, 'lemonoil', '100')

    DF = DF.iloc[:-3, :]
    DF.index = pd.to_numeric(DF.index)
    DF.index = DF.index.astype(float) / 1000
    #print(type(DF.index[0]))

    colors = plt.cm.viridis(np.linspace(0, 1, 8))

    label_color_dict = {
        'limonene': [colors[0], 'o'],
        'lemonoil': [colors[1], 'o'],
        'ylangylang': [colors[2], 'o'],
        'roseoil': [colors[3], 's'],
        'benzylalcohol': ['brown', 's'],
        '1octen3ol': [colors[5], '^'],
        'benzaldehyde': [colors[6], 'o'],
        'linalool': [colors[7], '^'],
        'mineraloil': ['grey', 'P'], }

    name_map = {
        'ylangylang': 'Ylang Ylang',
        'benzylalcohol': 'Benzylalcohol',
        'benzaldehyde': 'Benzaldehyde',
        '1octen3ol': '1-Octen-3-ol',
        'roseoil': 'Rose Oil',
        'lemonoil': 'Lemon Oil',
        'limonene': 'Limonene',
        'linalool': 'Linalool',
        'mineraloil': 'Mineral Oil (Ctrl)'
    }

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.set_facecolor('None')

    ax.set_facecolor('None')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    #ax.set_ylabel('Normalized Response', fontsize=25)
    #ax.set_xlabel('Time (s)', fontsize=25)
    #ax.set_ylim(-1, .4)
    #plt.xticks(fontsize=25)
    #plt.yticks(fontsize=25)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    #ax.set_ylim(-.65,.45)
    plotted_labels = []

    x_scale_length = 1  # Adjust this based on your X-axis scale
    y_scale_length = 0.1  # Adjust this based on your Y-axis scale
    x_bar_length = x_scale_length / 8  # X-axis total span is 8
    y_bar_length = y_scale_length
    scale_bar_thickness = 0.005
    # Draw X scale bar
    x_bar = patches.Rectangle((0.05, 0), x_bar_length, 0.01,scale_bar_thickness, transform=ax.transAxes, color='black')
    ax.add_patch(x_bar)

    # Draw Y scale bar
    y_bar = patches.Rectangle((0.05, 0), 0.01, y_bar_length, scale_bar_thickness, transform=ax.transAxes, color='black')
    ax.add_patch(y_bar)
    for n in range(len(EAGS)):
        text = EAGS[n]  # Replace with your text

        pattern = r'(1k|10k|100)(.*?)(000)'

        match = re.search(pattern, text)
        odor = match.group(2)
        print(odor)
        TRACE = DF[EAGS[n]]
        plotted_labels.append(name_map.get(odor))
        TRACE.plot(color=label_color_dict.get(odor)[0], linewidth=5)

    #ax.legend(plotted_labels, markerscale=1.5, fontsize=20, frameon=False)
    if SAVEDIR != None:
        plt.savefig(f'{SAVEDIR}.jpg')
        plt.savefig(f'{SAVEDIR}.svg')
    plt.show()


file = '/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Data/ControlSubtracted/Normalized/BF.1_2_/Dataframes/NoQC/All_Odors.csv'

'''EAGS = ['082322m1a110kroseoil0000wave1',
        '082322m1a110klemonoil0001wave0',
        '082522m2a110kylangylang0001wave1']
        #'080522m1a110kmineraloil0005wave0']'''

'''EAGS = ['082222m1a11kroseoil0000wave2',
        '082222m2a11klemonoil0000wave1',
        '082522m2a11kylangylang0001wave1']
        #'080522m1a11kmineraloil0005wave0']'''

'''EAGS = ['081822m1a1100roseoil0001wave0',
        '082222m1a1100lemonoil0001wave1',
        '081822m1a1100ylangylang0000wave0']
        #'080522m1a1100mineraloil0005wave0']'''

'''EAGS = ['082922m3a11klinalool0000wave2',
        '072822m1a11k1octen3ol0001wave2',
        '072822m1a11kbenzylalcohol0000wave0']'''
'''EAGS=['090122m2a110klimonene0000wave1',
      '080522m1a110kmineraloil0005wave0']'''

'''EAGS=['082922m3a11klinalool0000wave2',
    '082522m2a11kylangylang0001wave1',
    '072822m1a11kbenzylalcohol0000wave0']'''

'''EAGS=['082922m3a11klinalool0000wave2',
      '082222m2a11klemonoil0000wave1',
      '090122m2a11klimonene0000wave1',
      '080522m1a1100mineraloil0005wave0']'''

'''EAGS=['082922m3a11klinalool0000wave2',
      '082222m1a11kroseoil0000wave2',
      '080522m1a1100mineraloil0005wave0']'''


Plot_Comparative_EAGS(file=file, EAGS=EAGS, SAVEDIR='/Users/joshswore/PycharmProjects/'
                                                            'SingleChannelAnalysis/EAG_WAVE_Plots/RoLoMin1k')
#Plot_Comparative_EAGS(file=file, EAGS=EAGS, SAVEDIR=None)
#Find_Prospective_Waves(CSV=file, Odor='benzylalcohol', Conc='1k')
#EAG_1_Conc_Plot(file, EAGS)