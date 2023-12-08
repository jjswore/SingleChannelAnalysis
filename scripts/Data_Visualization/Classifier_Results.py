from utils.Classifier_Results_Library import *
from Config.Config import ControlSUB_ResultsDir
#
ODEABEV_L = ['BolOctLin']

for o in ODEABEV_L:
    File_Dir=f'{ControlSUB_ResultsDir}/{o}/ClassifierResults/'
    TS_File_Dir  = f'{ControlSUB_ResultsDir}/{o}/ClassifierResults/'
    Save_Dir = None #f'{File_Dir}/Figures/'

    #if not os.path.exists(Save_Dir):
        #os.makedirs(Save_Dir)

    name_map = {
        'ylangylang': 'Ylang Ylang',
        'benzylalcohol': 'Benz',
        'benzaldehyde': 'Benzaldehyde',
        '1octen3ol': 'Octenol',
        'roseoil': 'Rose Oil',
        'lemonoil': 'Lemon Oil',
        'limonene': 'Lim',
        'linalool': 'Linalool',
        'mineraloil': 'CTRL'
    }

    SVM_Results ='SVM_Results.pickle'
    #RF_Results = 'RF_Results.pickle'
    RF_Results = 'RF_Results.pickle'

    SVM_DF = pickle_to_DF(f'{File_Dir}{SVM_Results}')
    RF_DF = pickle_to_DF(f'{TS_File_Dir}{RF_Results}')
    print(SVM_DF['predicted_classes'][5])
    labels = [name_map[label] for label in SVM_DF['predicted_classes'][0] if label in name_map]

    print(len(labels))

    names=['SVM Results', 'RF Results']

    df = pd.concat([SVM_DF['accuracy_score'],RF_DF['accuracy_score']],axis=1,keys=names)


    SVM_CM = extract_CM(SVM_DF)

    RF_CM = extract_CM(RF_DF)
    #title='Lemon Oil, Limonene, 1-Octen-3-ol \n  Ylang Ylang, Benzylalcohol'
    #', '.join(labels)
    plot_CM(SVM_CM,labels,YROT=90, XROT=0, TITLE=None, REARRANGE=False, SAVEDIR=None)#f'{Save_Dir})#SVM_')
    plot_CM(RF_CM, labels, YROT=90, XROT=0, TITLE='1st', REARRANGE=False, SAVEDIR=None)#f'{Save_Dir}RF_')
    #plot_CM(RF_CM, labels, YROT=90, XROT=0, TITLE='2nd', REARRANGE=True, SAVEDIR=f'{Save_Dir}RF_')
    ViPlot(df,'Classifier Results',len(labels), INNER='box', DisplayMean=True, SAVEDIR=None)#Save_Dir)
    #BoxPlot(df,'Classifier Results',len(labels))#,SAVEDIR=Save_Dir)
