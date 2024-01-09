import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


print('loading df')
DFraw = pd.read_csv('/Users/joshswore/Manduca/Single_Channel_Analysis/Extracted_Waves/'
                  'Quality_Controlled_Data/All_Odors.csv_QC_T_1.csv', index_col=0)
DFnorm = pd.read_csv('/Users/joshswore/Manduca/Single_Channel_Analysis/Normalized_Exctracted_Waves/'
                   'Quality_Controlled_Data/All_Odors.csv_QC_T_1.csv', index_col=0)

VOC_L =['lemonoil', 'limonene', 'linalool', 'benzaldehyde', 'benzylalcohol', '1octen3ol', 'roseoil', 'ylangylang']

name_map = {
        'ylangylang': 'Ylang Ylang',
        'benzylalcohol': 'Benzylalcohol',
        'benzaldehyde': 'Benzaldehyde',
        '1octen3ol': '1-Octen-3-ol',
        'roseoil': 'Rose Oil',
        'lemonoil': 'Lemon Oil',
        'limonene': 'Lim',
        'linalool': 'Linalool',
        'mineraloil': 'CTRL'
    }
colors = plt.cm.Paired(np.linspace(0, 1, 8))
label_color_dict = {
        'limonene': colors[0],
        'lemonoil': colors[1],
        'ylangylang': colors[2],
        'roseoil': colors[3],
        'benzylalcohol': colors[4],
        '1octen3ol': colors[5],
        'benzaldehyde': colors[6],
        'linalool': colors[7]
    }
for VOC in VOC_L:

    # Filter data for 'ylangylang'
    print(VOC)
    Rd = DFraw[DFraw['label'].str.contains(VOC)]
    Nd = DFnorm[DFnorm['label'].str.contains(VOC)]

    # Extract labels before transposing
    label = name_map[VOC]
    print(label)

    # Transpose and separate numeric data and metadata
    Rdata = Rd.T.iloc[:-3, :]
    Ndata = Nd.T.iloc[:-3, :]

    # Convert index to numeric and scale
    Rdata.index = pd.to_numeric(Rdata.index) / 1000
    Ndata.index = pd.to_numeric(Ndata.index) / 1000

    # Plotting setup
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(15, 3.3))
    Conc = ['Raw', 'Normalized']

    print('building plots')
    for y, conc in enumerate(Conc):
        axes[y].spines['top'].set_visible(False)
        axes[y].spines['right'].set_visible(False)
        axes[y].spines['bottom'].set_position('zero')
        axes[y].spines['bottom'].set_position('zero')
        axes[y].set_title(conc, weight='bold', size=26)
        axes[y].set_xlim(-.5, 9)

    axes[0].set_ylabel(label, weight='bold', size=18)
    axes[0].set_ylim(-125, 35)
    axes[1].set_ylim(-1.25, .35)

    print('building dataframe')
    for VOC in VOC_L:
        Rtemp = Rdata.loc[:, Rd['label'] == VOC].T
        Ntemp = Ndata.loc[:, Nd['label'] == VOC].T
        color = label_color_dict.get(label)

        # Process and plot for each label
        for temp, ax in zip([Rtemp, Ntemp], axes):
            temp = temp.T
            temp['Mean'] = temp.mean(axis=1)
            temp['STD'] = temp.std(axis=1)

            print(f'plotting {label} and {ax.get_title()}')
            temp.plot(kind='line', y='Mean', color='black', legend=None, ax=ax)
            ax.fill_between(temp.index, temp['Mean'] - temp['STD'], temp['Mean'] + temp['STD'], color='green', alpha=.5,
                            edgecolor='black', linewidth=0.5)
    plt.savefig(f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/EAG_WAVE_Plots/{VOC}RawNorm.svg',transparent=True)
    plt.show()

