import os
from utils.EAG_SIngleChannel_DataProcessing_Library import csv_plot
# Path to the parent directory
parent_directory = '/Users/joshswore/analysis_python_code/EAG_and_VOC_Project/Single_Channel_Analysis/' \
                   'Data/NoControlSubtraction/Normalized/NoFilt/'

# Use os.walk to go through each directory and subdirectory
for dirpath, dirnames, filenames in os.walk(f'{parent_directory}Extracted_Waves/'):
    for file in filenames:
        # Full path to the file
        file_path = os.path.join(dirpath, file)

        # Process the file
        # Add your logic here
        print(file_path)
        print(dirpath)
        if file.endswith('.csv'):
            n = file.split('.')
            csv_plot(file_path,n[0],f'{parent_directory}EAG_Plots/')