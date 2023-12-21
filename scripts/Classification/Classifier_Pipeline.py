from utils.EAG_Classifier_Library import *
import pandas as pd

#OdAbrev='YYRoLinMin'


ODEABEV_L = ['YYLoRoMin-1k10k100']
#ODOR_L = ['limonene|linalool|1octen3ol|benzylalcohol']
#DILUTIONS =['1k','1k|10k','1k|100','1k|10k|100']


for OdAbrev in ODEABEV_L:
#read in
    TeDF = pd.read_csv(f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/'
                       f'{OdAbrev}/Butterworth_Optimized_Filter/{OdAbrev}_testingDF.csv',index_col=0, dtype={'concentration': 'string'})

    PCA_DF = pd.read_csv(f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/'
                         f'{OdAbrev}/PCA/{OdAbrev}_PCA.csv',index_col=0, dtype={'concentration': 'string'})

    Test_PCA_DF = PCA_DF.loc[PCA_DF.index.intersection(TeDF.index)]

    TS_Data = TeDF.iloc[:,250:5500]
    Meta_Data = Test_PCA_DF.iloc[:,-3:]
    Test_DF = pd.concat([TS_Data, Meta_Data], axis=1)

    Save_Directory = f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/' \
                     f'ControlSubtracted/{OdAbrev}/ClassifierResults/'

    if not os.path.exists(Save_Directory):
        os.makedirs(Save_Directory)

    Test_PCA_DF = pd.concat([Test_PCA_DF.iloc[:, :25], Test_PCA_DF.iloc[:, -3:]], axis=1)
    #TeDF=pd.concat([TeDF.iloc[:,:-1], Test_PCA_DF.iloc[:,-3:]], axis=1)

    ODOR_L = Test_DF['label'].unique() #'linalool|lemonoil|mineraloil'
    CONC_L = Test_DF['concentration'].unique()
    ODOR_L = [odor for odor in ODOR_L if odor != 'mineraloil']
    Odors = '|'.join(ODOR_L)
    #Concs = '|'.join(CONC_L)
    Concs = '1k'
    print(Odors)
    print(Concs)
    #print(f'beginng Classification of YY_Normalized channels summed Data')

    #data to input can be time series data or PCs Usin PC's we can expect training to occur faster since there are fewer "features"

    print(f'beginning SVM...')
    SVM_Results=SVMmodel(concentrations=Concs, data=[Test_DF], odor=Odors,PosL='lemonoil', repeats=100)
    pickle_Saver(savedir=Save_Directory,ext='SVM_Results',data=SVM_Results)

    print(f'beginning Random Forest')
    RF_results=RFmodel(concentrations=Concs,data=[Test_PCA_DF],odor=Odors,PosL='lemonoil', repeats=100)
    pickle_Saver(savedir=Save_Directory,ext='RF_Results',data=RF_results)