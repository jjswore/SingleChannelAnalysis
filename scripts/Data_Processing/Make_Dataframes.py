from utils.EAG_DataProcessing_Library import *

df = EAG_df_build('/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Data/'
                  'ControlSubtracted/'
                  'No_processing/Extracted_Waves/')
save = '/Users/joshswore/PycharmProjects/SingleChannelAnalysis/' \
       'Data/ControlSubtracted/' \
       'No_processing/Dataframes/NoQC/'
if not os.path.exists(save):
    os.makedirs(save)
df.to_csv(f'{save}All_Odors.csv')
