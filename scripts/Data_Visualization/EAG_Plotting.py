import os
import pandas as pd
import glob
import csv
import os
from matplotlib import pyplot as plt
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
def Find_Prospective_Waves(CSV, Odor):
    DF = pd.read_csv(CSV, index_col=0)
    for file in DF.index:
        if Odor in file:
            DF.T[file][:-3].plot()
            plt.title(file)
            #plt.ylim()
            plt.show()

file ='/Users/joshswore/PycharmProjects/' \
      'SingleChannelAnalysis/Results/' \
    'ControlSubtracted/LimLoMin/' \
    'Butterworth_Optimized_Filter/LimLoMin_finalDF.csv'

Find_Prospective_Waves(file, 'limonene')
    #'082922m3a11klimonene0001wave2'
    #'082922m3a11klimonene0000wave2'

    #'080522m1a11kmineraloil0005wave0'
'''DF = pd.read_csv('/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/'
                'ControlSubtracted/LimMin/Butterworth_Optimized_Filter/LimMin_finalDF.csv',
                 index_col=0).T

Limonene = DF['082922m3a11klimonene0001wave2'][:-3]
MineralOil = DF['080522m1a11kmineraloil0005wave0'][:-3]

fig, ax = plt.subplots(figsize=(10, 10))

fig.set_facecolor('lightgrey')
ax.set_facecolor('lightgrey')
#ax.set_alpha(0.25)

plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Time', fontsize=25)
plt.ylabel('Normalized Response', fontsize=25)
#plt.title(TITLE, fontsize=25)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
ax.set_ylim(-.5,.25)

Limonene.plot(color='limegreen')
MineralOil.plot(color='black')

plt.savefig('/Users/joshswore/PycharmProjects/'
            'SingleChannelAnalysis/EAG_WAVE_Plots/LimMin.svg')
plt.show()'''

