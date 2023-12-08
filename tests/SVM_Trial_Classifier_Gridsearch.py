import pandas as pd

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
    train_features, test_features, train_labels, test_labels = TT_Split(data_df, .75)
    print(train_labels, train_features)

    # Set hyperparameters to search over
    kernel = ['rbf']
    C = [0.5, 1, 4, 5, 7, 7.5, 8, 9, 10, 12.5, 15, 17.5, 20, 22.5, 25, 30, 40, 50, 65, 80]
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