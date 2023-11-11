import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import transforms
import numpy as np
from sklearn.decomposition import PCA


def draw_confidence_ellipse(ax, data, n_std, label, color, marker):
    cov = np.cov(data, rowvar=False)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    print(color)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=color,
                      alpha=0.3)
    # Calculating the standard deviation of x from the square root of the variance and multiplying with the number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(data[:, 0])

    # Calculating the standard deviation of y from the square root of the variance and multiplying with the number of standard deviations.
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(data[:, 1])

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

    # Add scatter plot for this class
    ax.scatter(data[:, 0], data[:, 1], color=color, marker=marker, edgecolors='black', s=60, label=label)


def Plot_PCA_ExplainedVariance(DATADIR, ODENOTE, ODORS, CONC, TITLE):
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
def Plot_PCA(DATADIR, ODENOTE, ODORS, CONC, TITLE, ANNOTATE=False):
    Odenotation = ODENOTE
    Concentration = CONC
    TITLE = TITLE
    DIR = f'{DATADIR}'

    PCA_df = f'{DIR}/{Odenotation}_PCA.csv'
    SaveDir = f'{DIR}/'

    # make sure the folder for saving exists
    if not os.path.exists(SaveDir):
        os.makedirs(SaveDir)

    # open the PCA_DF into a dataframe
    data_df = pd.read_csv(PCA_df, index_col=0)
    DF = data_df[data_df['concentration'].str.contains(CONC)]
    PCA_DF = DF[DF['label'].str.contains(ODORS)]
    print(PCA_DF['label'])

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
    print(Targets)

    plotted_labels = []  # List to keep track of labels we have plotted

    for target in Targets:
        print(target)
        color = label_color_dict.get(target)  # fetch color from the dictionary
        print(color)
        if color is None:
            continue
        indicesToKeep = (PCA_DF['label']) == target
        subset = PCA_DF.loc[indicesToKeep]
        plt.scatter(PCA_DF.loc[indicesToKeep, 'PC 1']
                    , PCA_DF.loc[indicesToKeep, 'PC 2'], color=color[0], marker=color[1], edgecolors='black', s=60,
                    label=target)
        plotted_labels.append(target)
        if ANNOTATE == True:
            for i in subset.index:
                plt.annotate(subset.loc[i, 'date'],  # This is the text to use for the annotation
                             (subset.loc[i, 'PC 1'], subset.loc[i, 'PC 2']),  # This is the point to annotate
                             textcoords="offset points",  # how to position the text
                             xytext=(5, 0),
                             fontsize=6,  # distance from text to points (x,y)
                             ha='left')  # horizontal alignment can be left, right or center



    plt.legend(plotted_labels, markerscale=1.5
               , fontsize=20, frameon=False)

    #plt.savefig(f'{SaveDir}{Odenotation}_PCA.jpg')
    #plt.savefig(f'{SaveDir}{Odenotation}_PCA.svg')
    plt.show()




'''OdeAbreve = 'BolBhydeMin'
odors = 'benzylalcohol|benzaldehyde|mineraloil'


data=f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/' \
     f'{OdeAbreve}/PCA/'
concentration = '1k'
SaveDir=f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/' \
        f'{OdeAbreve}/PCA/'

Plot_PCA(DATADIR=data,ODENOTE=OdeAbreve,CONC=concentration,ODORS=odors,TITLE=f'')'''

