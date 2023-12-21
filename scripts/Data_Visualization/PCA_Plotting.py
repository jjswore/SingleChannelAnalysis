from utils.Plot_PCA import *
from Config.Config import ControlSUB_ResultsDir

OdeAbreve = 'RoLoYYMin'
odors = 'ylangylang|roseoil|lemonoil'


data=f'{ControlSUB_ResultsDir}/{OdeAbreve}/PCA/'
concentration = '1k'
SaveDir=f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/{OdeAbreve}/PCA/'

Plot_2D_PCA(DATADIR=data,OA=OdeAbreve,CONC=concentration,ODORS=odors,TITLE='', SAVE=True)