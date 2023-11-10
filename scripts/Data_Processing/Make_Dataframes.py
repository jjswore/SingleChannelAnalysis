from utils.EAG_DataProcessing_Library import *

df = EAG_df_build('/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Data/'
                  'ControlSubtracted/'
                  'Normalized/BF.1_2_/Extracted_Waves/')
save = '/Users/joshswore/PycharmProjects/SingleChannelAnalysis/' \
       'Data/ControlSubtracted/' \
       'Normalized/BF.1_2_/Dataframes/NoQC/'
if not os.path.exists(save):
    os.makedirs(save)
df.to_csv(f'{save}All_Odors.csv')
