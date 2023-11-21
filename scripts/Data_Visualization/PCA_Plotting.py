from utils.Plot_PCA import *
from Config.Config import ControlSUB_ResultsDir

OdeAbreve = 'LemLinMin-1k10k'
odors = 'mineraloil|lemonoil|limonene'


data=f'{ControlSUB_ResultsDir}{OdeAbreve}/PCA/'
concentration = '1k'
SaveDir=f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/' \
        f'{OdeAbreve}/PCA/'

Plot_2D_PCA(DATADIR=data,OA=OdeAbreve,CONC=concentration,ODORS=odors,TITLE='', SAVE=True)