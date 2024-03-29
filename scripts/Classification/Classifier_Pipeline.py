from utils.EAG_Classifier_Library import *
import pandas as pd

#OdAbrev='YYRoLinMin'


ODEABEV_L = ['YYLoRoMin-10k', 'YYLoRoMin-100']
#DILUTIONS =['1k','1k|10k','1k|100','1k|10k|100']


for OdAbrev in ODEABEV_L:
#read in
    PCA_DF = pd.read_csv(f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/'
                         f'{OdAbrev}/PCA/{OdAbrev}_PCA.csv',index_col=0)

    TeDF = pd.read_csv(f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/'
                       f'{OdAbrev}/Butterworth_Optimized_Filter/{OdAbrev}_testingDF.csv',index_col=0)

    Test_PCA_DF = PCA_DF.loc[PCA_DF.index.intersection(TeDF.index)]
    #PCA_DF.drop(columns='label.1',inplace=True)
    Save_Directory = f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/' \
                     f'ControlSubtracted/{OdAbrev}/ClassifierResults/'
    Test_PCA_DF=pd.concat([Test_PCA_DF.iloc[:,:5],Test_PCA_DF.iloc[:,-3:]], axis=1)

    #TeDF=pd.concat([TeDF.iloc[:,:-1], Test_PCA_DF.iloc[:,-3:]], axis=1)

    ODOR_L =  Test_PCA_DF['label'].unique() #'linalool|lemonoil|mineraloil'
    CONC_L = Test_PCA_DF['concentration'].unique()
    Odors = '|'.join(ODOR_L)
    #Concs = '|'.join(CONC_L)
    Concs = '100'
    print(Odors)
    print(Concs)
    #print(f'beginng Classification of YY_Normalized channels summed Data')

    #data to input can be time series data or PCs Usin PC's we can expect training to occur faster since there are fewer "features"

    print(f'beginning SVM...')
    SVM_Results=SVMmodel(concentrations=Concs,data=[Test_PCA_DF],odor=Odors,PosL='lemonoil', repeats=100)
    pickle_Saver(savedir=Save_Directory,ext='SVM_Results',data=SVM_Results)

    print(f'beginning Random Forest')
    RF_results=RFmodel(concentrations=Concs,data=[Test_PCA_DF],odor=Odors,PosL='lemonoil', repeats=100)
    pickle_Saver(savedir=Save_Directory,ext='RF_Results',data=RF_results)