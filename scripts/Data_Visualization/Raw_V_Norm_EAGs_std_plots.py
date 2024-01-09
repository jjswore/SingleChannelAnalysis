import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def main():
    print('loading df')
    raw = pd.read_csv('/Users/joshswore/Manduca/Single_Channel_Analysis/Extracted_Waves/'
                      'Quality_Controlled_Data/All_Odors.csv_QC_T_1.csv', index_col=0)
    norm = pd.read_csv('/Users/joshswore/Manduca/Single_Channel_Analysis/Normalized_Exctracted_Waves/'
                       'Quality_Controlled_Data/All_Odors.csv_QC_T_1.csv', index_col=0)

    fig, axes = plt.subplots(nrows=8, ncols=2, sharey=False, sharex=True, figsize=(15, 25))
    Conc = ['Raw', 'Normalized']
    #fig.patch.set_facecolor('gainsboro')  # Setting the figure background to gre

    #for ax in axes.flatten():
    #    ax.set_facecolor('gainsboro')

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





    print('building plots')
    for x, label in zip(enumerate(raw['label'].unique()), enumerate(norm['label'].unique())):
        axes[x[0], 0].set_ylabel(label[1], weight='bold', size=18)
        for y, conc in zip(range(len(Conc)), Conc):
            axes[x[0], y].spines['top'].set_visible(False)
            axes[x[0], y].spines['right'].set_visible(False)
            axes[0, y].set_title(conc, weight='bold', size=26)
            axes[x[0], y].set_xlim(-5, 5000)
            axes[x[0], 0].set_ylim(-125, 25)
            axes[x[0], 1].set_ylim(-1.25, .25)

    print('building dataframe')
    for x, label in zip(range(len(raw['label'].unique())), raw['label'].unique()):
        Rtemp = raw[(raw['label'] == label) & (raw['concentration'] == '1k')]
        Ntemp = norm[(norm['label'] == label) & (norm['concentration'] == '1k')]
        color = label_color_dict.get(label)  # fetch color from the dictionary
        Rtemp = Rtemp.T
        Rtemp['Mean'] = Rtemp.iloc[:-3,:].mean(axis=1)
        #Rtemp['Var'] = Rtemp.iloc[:-3,:-1].var(axis=1)
        #Rtemp['SEM'] = Rtemp.iloc[:-3,:-2].sem(axis=1)
        Rtemp['STD'] = Rtemp.iloc[:-3,:-3].std(axis=1)
        Rtemp.drop(['label', 'concentration', 'date'], inplace=True)

        Ntemp = Ntemp.T
        Ntemp['Mean'] = Ntemp.iloc[:-3, :].mean(axis=1)
        #Ntemp['Var'] = Ntemp.iloc[:-3, :-1].var(axis=1)
        #Ntemp['SEM'] = Ntemp.iloc[:-3, :-2].sem(axis=1)
        Ntemp['STD'] = Ntemp.iloc[:-3, :-3].std(axis=1)
        Ntemp.drop(['label', 'concentration', 'date'], inplace=True)

        print('displaying plots')
        print(f'plotting {label} and Raw')
        Rtemp.plot(kind='line', y='Mean', color='black', legend=None, ax=axes[x, 0])
        #Rtemp.plot(kind='line', y='Mean', color=color, alpha=.01, legend=None, yerr='STD', ax=axes[x, 0])
        axes[x, 0].fill_between(Rtemp.index,
                        Rtemp['Mean'] - Rtemp['STD'],
                        Rtemp['Mean'] + Rtemp['STD'],
                        color=color, alpha=.5, edgecolor='black', linewidth=0.5)

        print(f'plotting {label} and Normalized')
        Ntemp.plot(kind='line', y='Mean', color='black', legend=None, ax=axes[x, 1])
        #Ntemp.plot(kind='line', y='Mean', color=color, alpha=.01, legend=None, yerr='STD',
        #                  ax=axes[x, 1])
        axes[x, 1].fill_between(Ntemp.index,
                                Ntemp['Mean'] - Ntemp['STD'],
                                Ntemp['Mean'] + Ntemp['STD'],
                                color=color, alpha=.5, edgecolor='black', linewidth=0.5)




    plt.savefig('/Users/joshswore/Manduca/Single_Channel_Analysis/Figures/Norm_v_Raw_PreFilter_MeanAndSTD.svg')
    plt.savefig('/Users/joshswore/Manduca/Single_Channel_Analysis/Figures/Norm_v_Raw_PreFilter_MeanAndSTD.jpg')
    plt.show()


if __name__ == "__main__":
    main()

