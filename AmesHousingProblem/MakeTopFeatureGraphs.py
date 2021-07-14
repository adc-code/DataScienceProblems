#
# MakeTopFeatureGraphs.py
#
# Small tool used to make graphs of the top most features based on the permutation importance
# for the different models.
#


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


SourceData = [ 'FeatureImportance_LinearReg_Permutations.csv',
               'FeatureImportance_RandomForest_Permutations.csv',
               'FeatureImportance_XGBoost_Permutations.csv',
               'FeatureImportance_CatBoost_Permutations.csv' ]

Titles = [ 'Top Features: Linear Regression',
           'Top Features: Random Forest Regressor',
           'Top Features: XGBoost Regressor',
           'Top Features: CatBoost Regressor' ]     

for i in range(len(SourceData)):

    print ('--> ', SourceData[i])

    # load the data...
    featImportData = pd.read_csv (SourceData[i])

    # print (featImportData.columns)
    # print (featImportData)

    # uncomment for testing... note that the bottom most features all will have importance values close
    # to zero.  Also, since there can be various issues like rounding errors, rates of convergence, 
    # and internal algorithm heuristics most of these features vary with the various models.
    # bottomMost = sortedDF[:10]
    # print (bottomMost)

    # get the most important features... which are the last ones since the data is sorted accendingly
    sortedDF = featImportData.sort_values (by='ImportanceMean')
    topMost  = sortedDF[-10:]

    # Make a graph
    ax = sns.barplot (x='ImportanceMean', y='Feature', data=topMost, palette="Blues_d")
    plt.title (Titles[i])    
    plt.tight_layout ()
  
    # and save it...
    fileName = SourceData[i].split('.')
    fileName = fileName [0] + '.jpg'
    plt.savefig (fileName)
 
    # plt.show ()


