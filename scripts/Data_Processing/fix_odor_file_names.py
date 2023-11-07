import os

DIR = '/Users/joshswore/analysis_python_code/EAG_and_VOC_Project/Single_Channel_Analysis/Data/NoControlSubtraction/' \
      'Normalized/NoFilt/Extracted_Waves/mineraloil/'

for file in os.listdir(DIR):
    if '_mineraloil_' in file:
        new_name = file.replace('_mineraloil_', '_1k_mineraloil_')
        os.rename(os.path.join(DIR, file), os.path.join(DIR, new_name))


for file in os.listdir(DIR):
    if '_mineraloil_' in file:
        date = os.path.basename(file1).split('_')[0]
        print(f'date of file is: {date}')
        pfiles2 = [x for x in folder if date.lower() in os.path.basename(x.lower())]
        print(pfiles2)

        for x in pfiles2:
            #get the time difference between experiment and the control. repeat for all files in folder
            tt = abs(os.path.getmtime(file1) - os.path.getmtime(x))
            if tt < result:
                #if the difference is smaller than the result variable then replace "result" with new dif
                result = tt
                ctrl = x