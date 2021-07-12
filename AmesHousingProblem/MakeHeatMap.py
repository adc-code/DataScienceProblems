#
# MakeHeatMap.py
#
# Small utility to make a heat map from the training data 
# 


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import HousingDataUtils


FEAT_CORR_TOL = 0.35


# load the data...
print ('Loading data...')
trainDF, testDF = HousingDataUtils.LoadData ()

# get the correlation matrix
corrMat = trainDF.corr ()

# uncomment for debugging
# print (corrMat)

# Make a heat map
fig, ax = plt.subplots(figsize=(7.5, 7.5))
sns.heatmap (abs(corrMat), annot=False, cbar=False, cmap='viridis', ax=ax, annot_kws={'size':6})
plt.tight_layout ()
plt.show ()

# report the features with little correlation to the sales price
salePriceCorrValues = pd.DataFrame ( { 'Correlation': corrMat['SalePrice'] } )
salePriceCorrValues ['BadCorrelation'] = 0
salePriceCorrValues ['BadCorrelation'] = abs (salePriceCorrValues ['Correlation']) < FEAT_CORR_TOL

poorCorrFeats = salePriceCorrValues [abs(salePriceCorrValues['Correlation']) < FEAT_CORR_TOL].transpose().columns.tolist()
poorCorrFeats.remove ('Id')

print ('Sales correlation values...')
print (salePriceCorrValues)

print ()
print ('Poor Correlation Features...')
print (poorCorrFeats)

