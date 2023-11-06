import os
import pandas as pd
import glob
import csv
import os
from matplotlib import pyplot as plt


def gather_csv_files_from_directory(root_dir):
    # This function searches for all CSV files from root directory and its sub-directories.

    # Use os.walk and glob to find all csv files in the directory and sub-directories
    csv_files = []
    for dir_name, _, _ in os.walk(root_dir):
        csv_files.extend(glob.glob(os.path.join(dir_name, "*.csv")))
    return csv_files

def open_wave(FILE):
    with open(FILE, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    f.close()
    l=data[0]
    l = list(map(float, l))
    return l

def EAG_df_build(DIR):
    F_List=[]
    for dirpath, dirnames, filenames in os.walk(DIR):
        for filename in filenames:
            F_List.append(os.path.join(dirpath, filename))
    master=[]
    mastern=[]
    masterl=[]
    masterc=[]
    masterDate=[]
    odor100=[]
    odor300=[]
    odor1K=[]
    odor3K=[]
    odor10K=[]
    o100n=[]
    o300n=[]
    o1kn=[]
    o3kn=[]
    o10kn=[]


    for file in F_List:
        if file.endswith('.csv'):
            #print('filename: ', file)
            fbase = os.path.basename(file)
            cat=fbase.lower().split('_')
            print(cat)
            n=os.path.basename(fbase.lower()).split("_")
            name=n[0]+n[1] +n[2] + n[3] + n[4].replace('.csv','')
            lab=n[2]
            conc=n[1]
            date=n[0]
            x=open_wave(file)
            if cat[1] == '100':
                odor100.append(x)
                master.append(x)
                o100n.append(name)
                mastern.append(name)
                masterl.append(lab)
                masterc.append(conc)
                masterDate.append(date)

            if cat[1] == '300':
                odor300.append(x)
                master.append(x)
                o300n.append(name)
                mastern.append(name)
                masterl.append(lab)
                masterc.append(conc)
                masterDate.append(date)

            elif cat[1] == '1k':
                odor1K.append(x)
                master.append(x)
                o1kn.append(name)
                mastern.append(name)
                masterl.append(lab)
                masterc.append(conc)
                masterDate.append(date)

            elif cat[1] == '3k':
                odor3K.append(x)
                master.append(x)
                o3kn.append(name)
                mastern.append(name)
                masterl.append(lab)
                masterc.append(conc)
                masterDate.append(date)

            elif cat[1] == '10k':
                odor10K.append(x)
                master.append(x)
                o10kn.append(name)
                mastern.append(name)
                masterl.append(lab)
                masterc.append(conc)
                masterDate.append(date)

    odor100_df=pd.DataFrame(dict(zip(o100n,odor100)),index=[x for x in range(0,9000)])
    odor300_df=pd.DataFrame(dict(zip(o300n,odor300)),index=[x for x in range(0,9000)])
    odor10K_df=pd.DataFrame(dict(zip(o10kn,odor10K)),index=[x for x in range(0,9000)])
    odor1K_df=pd.DataFrame(dict(zip(o1kn,odor1K)),index=[x for x in range(0,9000)])
    odor3K_df=pd.DataFrame(dict(zip(o3kn,odor3K)),index=[x for x in range(0,9000)])
    master_df=pd.DataFrame(dict(zip(mastern,master)),index=[x for x in range(0,9000)])
    print(master_df)
    master_df=master_df.T
    master_df['label']=masterl
    master_df['concentration']=masterc
    master_df['date']=masterDate

    odor100_df['Mean']=odor100_df.mean(axis=1)
    odor300_df['Mean']=odor300_df.mean(axis=1)
    odor10K_df['Mean']=odor10K_df.mean(axis=1)
    odor1K_df['Mean']=odor1K_df.mean(axis=1)
    odor3K_df['Mean']=odor3K_df.mean(axis=1)

    odor100_df['Var']=odor100_df.iloc[:,:-1].var(axis=1)
    odor300_df['Var']=odor300_df.iloc[:,:-1].var(axis=1)
    odor10K_df['Var']=odor10K_df.iloc[:,:-1].var(axis=1)
    odor1K_df['Var']=odor1K_df.iloc[:,:-1].var(axis=1)
    odor3K_df['Var']=odor3K_df.iloc[:,:-1].var(axis=1)

    odor100_df['SEM']=odor100_df.iloc[:,:-2].sem(axis=1)
    odor300_df['SEM']=odor300_df.iloc[:,:-2].sem(axis=1)
    odor10K_df['SEM']=odor10K_df.iloc[:,:-2].sem(axis=1)
    odor1K_df['SEM']=odor1K_df.iloc[:,:-2].sem(axis=1)
    odor3K_df['SEM']=odor3K_df.iloc[:,:-2].sem(axis=1)

    odor100_df['STD']=odor100_df.iloc[:,:-2].std(axis=1)
    odor300_df['STD']=odor300_df.iloc[:,:-2].std(axis=1)
    odor10K_df['STD']=odor10K_df.iloc[:,:-2].std(axis=1)
    odor1K_df['STD']=odor1K_df.iloc[:,:-2].std(axis=1)
    odor3K_df['STD']=odor3K_df.iloc[:,:-2].std(axis=1)

    return master_df,odor100_df,odor300_df,odor1K_df,odor3K_df,odor10K_df

def main():
    print('building df')
    df,_,_,_,_,_=EAG_df_build('/Users/joshswore/Manduca/Single_Channel_Analysis/Raw/Butter.1_6')
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
#    main()
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