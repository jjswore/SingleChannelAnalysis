from utils.GA_Butter_Library import *
from utils.EAG_Classifier_Library import TT_Split
from utils.SubSample_Control import Reduce_Ctrl_Samples
import time
import os
from EAG_PCA import EAG_PCA
from utils.Plot_PCA import Plot_2D_PCA
#==========================================================================================
#Run the GA

ODORS_L = 'ylangylang|roseoil|mineraloil|lemonoil'
ODEABEV_L = [ 'YYLoRoMin-100']
DILUTIONS =['100']

data='/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Data/ControlSubtracted/Normalized/NoFilt/' \
         'Dataframes/QualityControlled/_QC_T_0.5.csv'

df = pd.read_csv(data, index_col=0)

data_df = Reduce_Ctrl_Samples(DF=df)

for ob, dil in zip(ODEABEV_L, DILUTIONS):
    # ... rest of the code ...

    print('beginning Optimization')
    start = time.time()


    odors = ODORS_L
    OdeAbrev = ob
    concentration = dil
    SaveDir=f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/' \
            f'Results/ControlSubtracted/{OdeAbrev}/'

    BFSaveDir = f'{SaveDir}/Butterworth_Optimized_Filter/'
    f'{OdeAbrev}_BestParams.csv'

    print('checking if Save Directory exists')
    if not os.path.exists(SaveDir):
        os.makedirs(SaveDir)

    if not os.path.exists(BFSaveDir):
        os.makedirs(BFSaveDir)


    DF=data_df[data_df['concentration'].str.contains(concentration)]
    DF=DF[DF['label'].str.contains(odors)]
    print('first split')
    print(DF['label'].unique())
    train_features, test_features, train_labels, test_labels =TT_Split(DF, .5)
    Training_data = pd.concat([train_features, train_labels], axis=1)
    Test_data = pd.concat([test_features, test_labels], axis=1)
    Training_data.to_csv(f'{BFSaveDir}{OdeAbrev}_trainingDF.csv')
    Test_data.to_csv(f'{BFSaveDir}{OdeAbrev}_testingDF.csv')
    # Get the indices of the train_features DataFrame
    train_indices = train_features.index
    # Select the corresponding rows from the LLL_df
    train_labels_df = DF.iloc[:,-3:].loc[train_indices]
    # Concatenate train_features and train_labels_df
    df = pd.concat([train_features, train_labels_df], axis=1)

    params, statistics=main(data=df, POPULATION_SIZE=98, TOURNAMENT_SIZE=3, CROSS_PROB=.5, MUT_PROB=.35, Early_Stop=15, G=150)

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


    PDF.to_csv(f'{BFSaveDir}'
               f'{OdeAbrev}_BestParams.csv')
    BDF.to_csv(f'{BFSaveDir}'
               f'{OdeAbrev}_finalDF.csv')
    STATSDF.to_csv(f'{BFSaveDir}'
                   f'{OdeAbrev}_STATS.csv')
    print('Computing PC')
    EAG_PCA(BDF,f'{SaveDir}',concentration,odors, OdeAbrev)
    Plot_2D_PCA(DATADIR=f'{SaveDir}/PCA/',OA=OdeAbrev,CONC=concentration,ODORS=odors,TITLE='',SAVE=True)

    print('the entire code has finished')

## from here I need to decide how I will proceed... Do I reprocess my data? no I think I return the best
## data frame right?
