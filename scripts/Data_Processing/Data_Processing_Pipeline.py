from utils.EAG_DataProcessing_Library import *

def run():
    SourceD='/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Data/'
    D = f'{SourceD}raw'

    s = f'{SourceD}ControlSubtracted2/'

    #process_data(D, savedir=f'{s}/Raw/Butter.1_6/',
    #             Norm=False, Smoothen=False, LOG=False,Butter=[.1, 6], B_filt=True, RETURN='SAVE')

    #process_data(D, savedir=f'{s}/Normalized/NoFilt/',
    #Norm='YY', Smoothen=False, LOG=False, Butter=[.1, 6], B_filt=False, RETURN='SAVE')

    process_data(D, savedir=f'{s}/Normalized/NoFilt/Extracted_Waves/',
                 Norm='YY', Smoothen=False, LOG=False, Butter=[.1, 6, 1], B_filt=False, RETURN='SAVE')
run()