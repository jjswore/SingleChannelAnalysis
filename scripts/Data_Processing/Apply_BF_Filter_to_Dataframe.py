import pandas as pd
from utils.EAG_DataProcessing_Library import butter_bandpass_filter, butter_bandpass
from utils.GA_Butter_Library import apply_filter_to_dataframe
import matplotlib.pyplot as plt
from EAG_PCA import EAG_PCA
from utils.Plot_PCA import Plot_2D_PCA
from scripts.Data_Visualization.EAG_Plotting import Plot_Comparative_EAGS_subplot

ResultDIR = '/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/'

DataDIR = '/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Data/' \
          'ControlSubtracted/Normalized/NoFilt/Dataframes/'

DF = pd.read_csv(f'{DataDIR}QualityControlled/_QC_T_0.5.csv', index_col=0)


PARAMS = pd.read_csv(f'{ResultDIR}LimLinOctBolMin/Butterworth_Optimized_Filter/LimLinOctBolMin_BestParams.csv', index_col=0)


LoLimDF = apply_filter_to_dataframe(DF.iloc[:, :5501],lowcut=PARAMS['lowcut'][0],highcut=PARAMS['highcut'][0],order=PARAMS['order'][0])
LoLimDF = pd.concat([LoLimDF, DF.iloc[:,-3:]], axis=1)


EAGS = ['082222m1a11kroseoil0000wave2',
        '082222m2a11klemonoil0000wave1',
        '090122m2a110klimonene0000wave1',
        '082922m3a11klinalool0000wave2']

Plot_Comparative_EAGS_subplot(file=LoLimDF.T, EAGS=EAGS,
                              SAVEDIR='/Users/joshswore/PycharmProjects/'
                                                 'SingleChannelAnalysis/TEST_Results/RoLinparams')

SaveDir = '/Users/joshswore/PycharmProjects/SingleChannelAnalysis/TEST_Results/'
concentration = '1k'
LoLimOdors = 'limonene|lemonoil|linalool|1octen3ol|roseoil|benzaldehyde'
LL_OdeAbrev = 'ALL_pca'
EAG_PCA(LoLimDF,f'{SaveDir}',concentration,LoLimOdors, LL_OdeAbrev)
Plot_2D_PCA(DATADIR=f'{SaveDir}/PCA/',OA=LL_OdeAbrev,CONC=concentration,ODORS=LoLimOdors,TITLE='',SAVE=True)

