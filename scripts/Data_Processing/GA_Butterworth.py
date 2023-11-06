from utils.GA_Butter_Library import *
from utils.EAG_Classifier_Library import TT_Split
import time
import os
from EAG_PCA import EAG_PCA
from Plot_PCA import Plot_PCA
#==========================================================================================
#Run the GA
print('beginning Optimization')
start = time.time()

data='/Users/joshswore/analysis_python_code/EAG_and_VOC_Project/Single_Channel_Analysis/Data/' \
     '/Normalized/NoFilt/Dataframes/QualityControlled/_QC_T_1.csv'
odors = 'mineraloil|benzaldehyde|roseoil'
Odenote = 'RoBhydeMin'
concentration = '1k'
SaveDir=f'/Users/joshswore/analysis_python_code/EAG_and_VOC_Project/Single_Channel_Analysis/' \
        f'Results/ControlSubtracted/'

print('checking if Save Directory exists')
if not os.path.exists(SaveDir):
    os.makedirs(SaveDir)

data_df = pd.read_csv(data, index_col=0)

DF=data_df[data_df['concentration'].str.contains(concentration)]
DF=DF[DF['label'].str.contains(odors)]
print('first split')
print(DF['label'].unique())
train_features, test_features, train_labels, test_labels =TT_Split(DF, .5)
Training_data = pd.concat([train_features, train_labels], axis=1)
Test_data = pd.concat([test_features, test_labels], axis=1)
Training_data.to_csv(f'{SaveDir}{Odenote}_trainingDF.csv')
Test_data.to_csv(f'{SaveDir}{Odenote}_testingDF.csv')
# Get the indices of the train_features DataFrame
train_indices = train_features.index
# Select the corresponding rows from the LLL_df
train_labels_df = DF.iloc[:,-3:].loc[train_indices]
# Concatenate train_features and train_labels_df
df = pd.concat([train_features, train_labels_df], axis=1)

params, statistics=main(data=df, POPULATION_SIZE=100, TOURNAMENT_SIZE=3, CROSS_PROB=.5, MUT_PROB=.25, G=150)

buttered_df = apply_filter_to_dataframe(dataframe=DF.iloc[:, :5001],
                                            lowcut=params['lowcut'],
                                            highcut=params['highcut'],
                                            order=params['order'])
print('here are the parameters:', params)
print(' here are the statistics:', statistics)

end = time.time()
total_time = end - start
print("\n"+ str(total_time))

STATSDF=pd.DataFrame(statistics)
BDF = pd.concat([buttered_df, DF.iloc[:,-3:]], axis=1)
PDF = pd.DataFrame.from_dict(params, orient='index').T


PDF.to_csv(f'{SaveDir}/{Odenote}_BestParams.csv')
BDF.to_csv(f'{SaveDir}{Odenote}_finalDF.csv')
STATSDF.to_csv(f'{SaveDir}{Odenote}_STATS.csv')
print('Computing PC')
EAG_PCA(BDF,SaveDir,concentration,odors, Odenote)
Plot_PCA(DATADIR=f'{SaveDir}/PCA/',ODENOTE=Odenote,CONC=concentration,ODORS=odors,TITLE='')
print('the entire code has finished')

## from here I need to decide how I will proceed... Do I reprocess my data? no I think I return the best
## data frame right?
