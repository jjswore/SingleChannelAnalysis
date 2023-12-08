from utils.Plot_PCA import *
from Config.Config import ControlSUB_ResultsDir

OdeAbreve = 'YYLoRoMin-10k'
odors = 'roseoil|lemonoil|ylangylang'


data=f'{ControlSUB_ResultsDir}/{OdeAbreve}/PCA/'
concentration = '10k'
SaveDir=f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/' \
        f'{OdeAbreve}/PCA/figures/'

Plot_PCA_Explained_Variance(DATADIR=data , ODENOTE=OdeAbreve , ODORS=concentration , CONC= odors, TITLE= '', SAVE=True)

Plot_2D_PCA(DATADIR=data,OA=OdeAbreve,CONC=concentration,ODORS=odors,TITLE='', SAVE=True)