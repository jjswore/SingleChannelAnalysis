from utils.Plot_PCA import *

OdeAbreve = 'YYLinMin'
odors = 'ylangylang|linalool|mineraloil'


data=f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/' \
     f'{OdeAbreve}/PCA/'
concentration = '1k'
SaveDir=f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/' \
        f'{OdeAbreve}/PCA/'

Plot_2D_PCA(DATADIR=data,ODENOTE=OdeAbreve,CONC=concentration,ODORS=odors,TITLE='', SAVE=False)