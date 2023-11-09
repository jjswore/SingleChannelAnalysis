import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def Plot_PCA(DATADIR, ODENOTE, ODORS, CONC, TITLE):
    # set the data to be loaded and set a save location
    Odenotation = ODENOTE
    Concentration = CONC
    TITLE = TITLE
    DIR = f'{DATADIR}'

    PCA_df = f'{DIR}/{Odenotation}_PCA.csv'
    PCA_obj = f'{DIR}/{Odenotation}_PCA.pickle'
    SaveDir = f'{DIR}/'

    # make sure the folder for saving exists
    if not os.path.exists(SaveDir):
        os.makedirs(SaveDir)

    # open the PCA_DF into a dataframe
    data_df = pd.read_csv(PCA_df, index_col=0)
    DF = data_df[data_df['concentration'].str.contains(CONC)]
    #print(DF)
    PCA_DF = DF[DF['label'].str.contains(ODORS)]
    print(PCA_DF['label'])
    # open the pickle file
    pca_obj = open(PCA_obj, 'rb')
    PCAobj = pickle.load(pca_obj)
    pca_obj.close()

    text_x_pos = 0.4 * len(PCAobj.explained_variance_ratio_[:100])
    # For y-axis: set the position a bit (e.g., 5%) above the maximum value
    text_y_pos = .4 * max(PCAobj.explained_variance_ratio_[:100])

    # Plot the explained variance
    plt.scatter(range(100), PCAobj.explained_variance_ratio_[:100], color='brown')
    plt.title('PCA Explained Variance')
    plt.ylabel('Explained Variance (%)')
    plt.xlabel('Principal Component')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.text(text_x_pos, text_y_pos, f'Explained Variance of First 2 PC\'s:{round(sum(PCAobj.explained_variance_ratio_[:2]), 2)}')
    #plt.savefig(f'{SaveDir}{Odenotation}_ExplainedVariance.jpg')
    #plt.savefig(f'{SaveDir}{Odenotation}_ExplainedVariance.svg')
    plt.show()

    # Plot Results
    # ============================================================================================

    plt.figure()
    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('PC 1', fontsize=25)
    plt.ylabel('PC 2', fontsize=25)
    plt.title(TITLE, fontsize=25)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)


    # Create a dictionary mapping labels to colors

    colors = plt.cm.Paired(np.linspace(0, 1, 8))
    label_color_dict = {
        'limonene': [colors[0], 'o'],
        'lemonoil': [colors[1], 'o'],
        'ylangylang':[ colors[2], 'o'],
        'roseoil': [colors[3], 's'],
        'benzylalcohol': [colors[4],'s'],
        '1octen3ol': [colors[5], '^'],
        'benzaldehyde': [colors[6], 'o'],
        'linalool': [colors[7], '^'],
        'mineraloil': ['grey', 'P'],
    }

    Targets = list(PCA_DF['label'].unique())

    plotted_labels = []  # List to keep track of labels we have plotted

    for target in Targets:
        print(target)
        color = label_color_dict.get(target)  # fetch color from the dictionary
        print(color)
        if color is None:
            continue
        indicesToKeep = (PCA_DF['label']) == target
        plt.scatter(PCA_DF.loc[indicesToKeep, 'PC 1']
                    , PCA_DF.loc[indicesToKeep, 'PC 2'], color=color[0], marker=color[1], edgecolors='black', s=60, label=target)
        plotted_labels.append(target)

    plt.legend(plotted_labels, markerscale=1.5
               , fontsize=20, frameon=False)

    plt.savefig(f'{SaveDir}{Odenotation}_PCA.jpg')
    plt.savefig(f'{SaveDir}{Odenotation}_PCA.svg')
    plt.show()




'''OdeAbreve = 'BolBhydeMin'
odors = 'benzylalcohol|benzaldehyde|mineraloil'
'''

'''data=f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/' \
     f'{OdeAbreve}/PCA/'
concentration = '1k'
SaveDir=f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/' \
        f'{OdeAbreve}/PCA/'

Plot_PCA(DATADIR=data,ODENOTE=OdeAbreve,CONC=concentration,ODORS=odors,TITLE=f'')
'''
