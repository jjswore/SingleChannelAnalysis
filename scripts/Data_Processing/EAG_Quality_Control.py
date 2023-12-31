import pandas as pd

from utils.EAG_DataProcessing_Library import *

data = '/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Data/ControlSubtracted/' \
       'Normalized/BF.1_2_/Dataframes/NoQC/All_Odors.csv'
SaveDir = '/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Data/ControlSubtracted/' \
          'Normalized/BF.1_2_/Dataframes/QualityControlled/'

if not os.path.exists(SaveDir):
    print('making directory...')
    os.makedirs(SaveDir)
    print('directory made')


ThreshList=[.5,1,10]
all_df=pd.read_csv(data, index_col=0)
#ctrl_DF = all_df[all_df['label'] == 'mineraloil']
#all_df = all_df[all_df['label'] != 'mineraloil']

print(len(all_df.T))
for t in ThreshList:
        print(f"begining quality control at threshold of {t}")
        final = FFT_LSTSQ_QC(all_df, t)
        final_T = pd.DataFrame(final.T.dropna(axis=0))
        #final_T = pd.concat([ctrl_DF,final_T], axis=0)
        filename = f"_QC_T_{str(t)}.csv"

        final_T.to_csv(f'{SaveDir}{filename}')
