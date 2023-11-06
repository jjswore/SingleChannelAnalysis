import pandas as pd

DF = pd.read_csv('/Users/joshswore/Manduca/Single_Channel_Analysis/PCA/1k_Concentration/Normalized/Butter.1_6/PCA_DATA/LLL/LLL_PCA.csv', index_col=0)
Parent = pd.read_csv('/Users/joshswore/Manduca/Single_Channel_Analysis/DataFrames/Normalized/Butter.1_6/Quality_Controlled_Data/All_Odors.csv_QC_T_1.csv',index_col=0)

filtered_DF = Parent[Parent['concentration'] == '1k']
filtered_DF = filtered_DF[(filtered_DF['label'] == 'lemonoil') |
                          (filtered_DF['label'] == 'limonene') |
                          (filtered_DF['label'] == 'linalool') ]
print(len(DF))
print(len(filtered_DF))

fixed_df = pd.concat([DF, filtered_DF.iloc[:,-2:]], axis=1)
fixed_df.to_csv('/Users/joshswore/Manduca/Single_Channel_Analysis/DataFrames/Normalized/GA_Butter_Optimized/LLL/PCA/LLL/LLL_PCA.csv')