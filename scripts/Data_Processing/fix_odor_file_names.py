import os

DIR = '/Users/joshswore/analysis_python_code/EAG_and_VOC_Project/Single_Channel_Analysis/Data/NoControlSubtraction/' \
      'Normalized/NoFilt/Extracted_Waves/mineraloil/'

for file in os.listdir(DIR):
    if '_mineraloil_' in file:
        new_name = file.replace('_mineraloil_', '_1k_mineraloil_')
        os.rename(os.path.join(DIR, file), os.path.join(DIR, new_name))

