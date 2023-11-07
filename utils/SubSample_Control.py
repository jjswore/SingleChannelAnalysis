import pandas as pd
import numpy as np

def Reduce_Ctrl_Samples(DF):
    classes = DF['label'].unique()
    class_lengths = []

    for c in classes:
        if c != 'mineraloil':
            class_lengths.append(len(DF[(DF['label'] == c) & (DF['concentration'] == '1k')]))
        else:
            pass
    Mean_Samples = np.mean(class_lengths)
    All_VOCs = DF[DF['label'] != 'mineraloil']
    CTRL_1k_Subsample = DF[(DF['label'] == 'mineraloil') & (DF['concentration'] == '1k')].sample(n=int(Mean_Samples.round(0)))
    CTRL_10k_Subsample = DF[(DF['label'] == 'mineraloil') & (DF['concentration'] == '10k')].sample(n=int(Mean_Samples.round(0)))
    CTRL_100_Subsample = DF[(DF['label'] == 'mineraloil') & (DF['concentration'] == '100')].sample(n=int(Mean_Samples.round(0)))

    DF_Subsampled = pd.concat([All_VOCs, CTRL_100_Subsample, CTRL_1k_Subsample, CTRL_10k_Subsample], axis=0)
    return DF_Subsampled

