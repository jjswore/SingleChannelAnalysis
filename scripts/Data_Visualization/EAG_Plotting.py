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
#    EAG_Plotting()
'''df= pd.read_csv('/Users/joshswore/Manduca/Single_Channel_Analysis/DataFrames/Normalized/GA_Butter_Optimized/LLL/LLL_finalDF.csv', index_col=0)
fig, axes = plt.subplots(nrows=8, ncols=3, sharey=True,figsize=(28, 28))
Conc=['100','1k','10k']
print('building plots')
for x,label in zip(range(len(df['label'].unique())),df['label'].unique()):
    axes[x,0].set_ylabel(label, weight='bold', size=26)
    for y, conc in zip(range(len(Conc)),Conc):
        axes[x,y].spines['top'].set_visible(False)
        axes[x,y].spines['right'].set_visible(False)
        axes[0,y].set_title(conc, weight='bold', size=26)
        plt.ylim(-1.50,.50)
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
plt.show()'''

'''df= pd.read_csv('/Users/joshswore/Manduca/Single_Channel_Analysis/DataFrames/Normalized/GA_Butter_Optimized/LLL/LLL_finalDF.csv', index_col=0)
#row = df[(df['label'] == 'limonene') & (df['concentration'] == '1k') & (df['date']) == '082533m2a1'].T
row = df[df.index == '082522m2a11klimonene0001wave1']
row.iloc[:,:-3].T.plot(color='black', legend = False)
#row['082522m2a11klimonene0001wave1'].plot()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_title('Butterworth Filter', size=18)
plt.ylim(-1.50,1.50)
plt.xlim(-5,5000)
plt.axvspan(500, 1000, color='red', alpha=.05)
print('displaying plots')


#plt.savefig('Raw_BF.1_6_AllSingleSiteWaves.svg')
#plt.savefig('Raw_BF.1_6_AllSingleSiteWaves.jpg')
plt.show()'''