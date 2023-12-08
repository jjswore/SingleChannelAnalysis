from utils.EAG_Classifier_Library import TT_Split, pickle_Saver
from sklearn import svm
from utils.Classifier_Results_Library import *
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from collections import Counter
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import pandas as pd

def SVMmodel(concentrations, data, odor, PosL, repeats):
    """
    Trains and evaluates a Support Vector Machine classifier on the provided dataset for each concentration of a given odor.

    Args:
        concentrations (list): A list of concentrations of the odor to be analyzed.
        data (pd.DataFrame): A pandas DataFrame containing the training data.
        odor (str): The name of the odor being analyzed.

    Returns:
        list: A nested list containing the classification performance metrics for each iteration of the Support Vector Machine classifier
            for each concentration of the given odor.
    """
    results = []
    for conc in [concentrations]:
        print(f"Beginning analysis for {odor} at {conc} concentration")
        classifier, params, best_score = SVM_GridSearch(data=data, concentration=conc, odors=odor)
        print(f"Best classifier for {odor} at {conc} concentration is {classifier}")
        print(f"Building SVM for {odor} at {conc} concentration")
        results.append([SVM_Testing(data=data, classifier=classifier, concentration=conc, odors=odor, P=PosL)
                        for _ in range(repeats)])
        print(f"Finished analysis for {odor} at {conc} concentration")
    return results

def SVM_GridSearch(data, concentration, odors):
    """
    Perform a grid search to optimize SVM hyperparameters.

    Args:
    - data (List[pd.DataFrame]): A list of pandas dataframes, each containing the data to be analyzed
    - concentration (str): The concentration of the odor stimuli to be analyzed
    - odors (str): The label of the odor stimuli to be analyzed
    - P (str): The positive class label for computing recall score

    Returns:
    - clf (svm.SVC): The optimized SVM classifier
    - gbp (Dict[str, Any]): The best set of hyperparameters found by grid search
    - gbs (float): The best score found by grid search
    """

    # Concatenate data into a single dataframe
    #Analysis_data = pd.concat(data, axis=1)

    # Filter data based on concentration and odor label
    #data_df = Analysis_data[(Analysis_data['concentration'].str.contains(concentration)) &
                            #(Analysis_data['label'].str.contains(odors))]
    data_df = pd.concat(data, axis=1)
    # Split data into train and test sets
    print('Splitting data...')
    train_features, test_features, train_labels, test_labels = TT_Split(data_df, .7)
    print(train_labels, train_features)

    # Set hyperparameters to search over
    kernel = ['rbf']
    C = [0.5, 1, 4, 5, 7, 7.5, 8, 9, 10, 12.5, 15, 17.5, 20, 22.5, 25, 30, 40]
    degree = [0, 0.01, 0.05, 0.1, 0.5]
    gamma = ['scale', 'auto', 0.1, 0.2, 0.5]
    coef0 = [0, 0.05, 0.1, 0.2]

    # Create parameter grid
    param_grid = {
        "kernel": kernel,
        "C": C,
        "degree": degree,
        "gamma": gamma,
        "coef0": coef0
    }

    print("Beginning grid search...")
    # Perform grid search
    GRID_cv = GridSearchCV(
        svm.SVC(),
        param_grid,
        scoring='accuracy',
        n_jobs=-1,
        error_score='raise',
        cv=15,
        verbose=1
    )
    GRID_cv.fit(train_features, train_labels)

    # Extract best hyperparameters and score
    gbp = GRID_cv.best_params_
    gbs = GRID_cv.best_score_

    # Create optimized classifier
    clf = svm.SVC(
        kernel=gbp['kernel'],
        C=gbp['C'],
        degree=gbp['degree'],
        gamma=gbp['gamma'],
        coef0=gbp['coef0']
    )

    print(f"Best parameters found: {gbp}")
    print(f"Best score found: {gbs}")
    print(f"Optimized classifier: {clf}")

    return clf, gbp, gbs
'''def Data_for_Classification(OdAbrev):
    PCA_DF = pd.read_csv(f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/'
                     f'{OdAbrev}/PCA/{OdAbrev}_PCA.csv',index_col=0, dtype={'concentration': 'string'})

    TeDF = pd.read_csv(f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/'
                       f'{OdAbrev}/Butterworth_Optimized_Filter/{OdAbrev}_testingDF.csv',index_col=0, dtype={'concentration': 'string'})

    Test_PCA_DF = PCA_DF.loc[PCA_DF.index.intersection(TeDF.index)]
    Test_PCA_DF=pd.concat([Test_PCA_DF.iloc[:,:5],Test_PCA_DF.iloc[:,-3:]], axis=1)
    DF = pickle_to_DF(f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/'
                     f'{OdAbrev}/ClassifierResults/SVM_Results.pickle')

    classifier = DF['classifier'][0]
    return Test_PCA_DF, classifier'''

def SVM_Testing(data, concentration, odors, classifier, P):
    """
    Applies SVM classification on the provided data.

    Args:
    - data: pandas.DataFrame, input dataset
    - concentration: str, the concentration of odorant to classify
    - odors: str, the odorants to classify
    - classifier: sklearn.svm.SVC, the classifier to use
    - P: str, the positive class

    Returns:
    - results_dict: dict, a dictionary containing the results of the classification
    """

    # Filter the dataset based on concentration and odorants
    # Concatenate the input data
    Analysis_data=pd.concat(data)

    # Get the data for the given concentration and odor label
    title=concentration
    data_df=Analysis_data[Analysis_data['concentration'].str.contains(title)]
    data_df=data_df[data_df['label'].str.contains(odors)]

    # Split the dataset into training and testing sets
    print('Splitting data...')
    train_features, test_features, train_labels, test_labels = TT_Split(data_df, .7)

    # Train the classifier
    print('Training model...')
    classifier.fit(train_features, train_labels)

    # Predict the response for test dataset
    # Train the classifier
    print('Training model...')
    classifier.fit(train_features, train_labels)
    Results_data = pd.concat([test_features, test_labels], axis=1)
    # Extract Trial Names
    extracted_trial = Results_data.index.to_series().str.extract(r'(.+)000(.)wave\d+')[0]

    # Map Extracted Trials to DataFrame
    Results_data['Trial'] = extracted_trial
    # Classify Each Wave and Store Predictions
    Results_data['Prediction'] = classifier.predict(test_features)

    # Aggregate Predictions by Trial
    trial_predictions = Results_data.groupby('Trial')['Prediction'].apply(list)
    votes = trial_predictions.apply(lambda x: Counter(x))
    # Determine the Majority Vote
    majority_vote = trial_predictions.apply(lambda x: Counter(x).most_common(1)[0][0])

    # Update Classifications for All Waves in a Trial
    for trial in majority_vote.index:
        Results_data.loc[Results_data['Trial'] == trial, 'Corrected_Prediction'] = majority_vote[trial]

    # Calculate accuracy and recall
    corrected_acc_score = balanced_accuracy_score(Results_data['Corrected_Prediction'], test_labels)
    acc_score = balanced_accuracy_score(Results_data['Prediction'], test_labels)
    print(f'{corrected_acc_score} vs. {acc_score}')
    predictions = Results_data['Corrected_Prediction']
    r_score = recall_score(test_labels, predictions, zero_division=0, average='macro', labels=[P])

    # Compute the confusion matrix
    CM = confusion_matrix(test_labels, predictions, labels=classifier.classes_).astype(float)

    # Compute the decision function values
    probabilities = classifier.decision_function(test_features)
    predictions = Results_data['Corrected_Prediction']
    # Print the results
    print('Accuracy: ', acc_score, '\n')
    print('Sensitivity: ', r_score, '\n')

    # Create a dictionary containing the results
    results_dict = {'classifier': classifier,
                    'accuracy_score': corrected_acc_score,
                    'sensitivity': r_score,
                    'confusion_matrix': CM,
                    'true classes': test_labels,
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'predicted_classes': classifier.classes_}

    return results_dict

import pandas as pd

#OdAbrev='YYRoLinMin'


ODEABEV_L = ['LimMin-smalltrainforBF']
#DILUTIONS =['1k','1k|10k','1k|100','1k|10k|100']


for OdAbrev in ODEABEV_L:
#read in
    PCA_DF = pd.read_csv(f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/'
                         f'{OdAbrev}/PCA/{OdAbrev}_PCA.csv',index_col=0, dtype={'concentration': 'string'})

    TeDF = pd.read_csv(f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/ControlSubtracted/'
                       f'{OdAbrev}/Butterworth_Optimized_Filter/{OdAbrev}_testingDF.csv',index_col=0, dtype={'concentration': 'string'})

    Test_PCA_DF = PCA_DF.loc[PCA_DF.index.intersection(TeDF.index)]
    #PCA_DF.drop(columns='label.1',inplace=True)
    Save_Directory = f'/Users/joshswore/PycharmProjects/SingleChannelAnalysis/Results/' \
                     f'ControlSubtracted/{OdAbrev}/ClassifierResults/'
    Test_PCA_DF=pd.concat([Test_PCA_DF.iloc[:,:5],Test_PCA_DF.iloc[:,-3:]], axis=1)

    #TeDF=pd.concat([TeDF.iloc[:,:-1], Test_PCA_DF.iloc[:,-3:]], axis=1)

    ODOR_L =  Test_PCA_DF['label'].unique() #'linalool|lemonoil|mineraloil'
    CONC_L = Test_PCA_DF['concentration'].unique()
    Odors = '|'.join(ODOR_L)
    #Concs = '|'.join(CONC_L)
    Concs = '1k'
    print(Odors)
    print(Concs)
    #print(f'beginng Classification of YY_Normalized channels summed Data')

    #data to input can be time series data or PCs Usin PC's we can expect training to occur faster since there are fewer "features"

    print(f'beginning SVM...')
    SVM_Results=SVMmodel(concentrations=Concs,data=[Test_PCA_DF],odor=Odors,PosL='limonene', repeats=100)
    pickle_Saver(savedir=Save_Directory,ext='SVM_Results',data=SVM_Results)

'''Test_PCA_DF, classifier = Data_for_Classification('LimMin')

# Filter the dataset based on concentration and odorants
# Concatenate the input data
Analysis_data=pd.concat([Test_PCA_DF])

#print(len(Analysis_data['date'].unique()))


# Get the data for the given concentration and odor label
#title=concentration
data_df=Analysis_data[Analysis_data['concentration'].str.contains('1k')]
data_df=data_df[data_df['label'].str.contains('limonene|mineraloil')]

# Split the dataset into training and testing sets
print('Splitting data...')
train_features, test_features, train_labels, test_labels = TT_Split(data_df, .7)


# Train the classifier
print('Training model...')
classifier.fit(train_features, train_labels)
Results_data = pd.concat([test_features,test_labels], axis=1)
# Extract Trial Names
extracted_trial = Results_data.index.to_series().str.extract(r'(.+)000(.)wave\d+')[0]

# Map Extracted Trials to DataFrame
Results_data['Trial'] = extracted_trial
# Classify Each Wave and Store Predictions
Results_data['Prediction'] = classifier.predict(test_features)

# Aggregate Predictions by Trial
trial_predictions = Results_data.groupby('Trial')['Prediction'].apply(list)
votes = trial_predictions.apply(lambda x: Counter(x))
# Determine the Majority Vote
majority_vote = trial_predictions.apply(lambda x: Counter(x).most_common(1)[0][0])

# Update Classifications for All Waves in a Trial
for trial in majority_vote.index:
    Results_data.loc[Results_data['Trial'] == trial, 'Corrected_Prediction'] = majority_vote[trial]

#Calculate accuracy and recall
corrected_acc_score = balanced_accuracy_score(Results_data['Corrected_Prediction'], test_labels)
acc_score = balanced_accuracy_score(Results_data['Prediction'], test_labels)
print(f'{corrected_acc_score} vs. {acc_score}')
#r_score = recall_score(test_labels, predictions, zero_division=0, average='macro', labels=[P])
'''
# Now, Analysis_data['Corrected_Prediction'] contains the updated classifications