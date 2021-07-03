#
# HousingModel_LinearReg.py
#
# Used to explore the Ames housing dataset with a few different linear
# models and various data conditions...
# 


from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np
import itertools

import HousingDataUtils


FIND_OPTIMAL_PARAMS = False #True
OUTPUT_COEFS        = True

BASE_OUTPUT_NAME    = 'Results/TestPreds_LinReg'
RESULTS_FILE        = 'Results_LinReg.csv'
COEFS_FILE          = 'LinearReg_Coefs.csv'



#
# BuildModel... 
#
def BuildModel (modType):

    # Standard linear regression
    if modType == 'Std':
        model = linear_model.LinearRegression ()

    # Std linear regression with ridge regularization
    elif modType == 'Ridge':
        model = Ridge ()

    # Std linear regression with lasso regularization
    elif modType == 'Lasso':
        model = Lasso ()

    # Std linear regression with elastic regularization
    elif modType == 'ElasticNet':
        model = ElasticNet ()

    return model


#
# MakeFileName
#
def MakeOutputFileName (params):
    
    FileName = BASE_OUTPUT_NAME
    FileName += '_' + params[0]
    FileName += '_' + params[1]
    FileName += '_' + params[2]

    if params[3] == True:
        FileName += '_LogT'
    else:
        FileName += '_LogF'

    # note LV -> little variation
    if params[4] == True:
        FileName += '_LVT'
    else:
        FileName += '_LVF'

    # LC -> little correlation
    if params[5] == True:
        FileName += '_LCT'
    else:
        FileName += '_LCF'

    # FE -> Feature engineering
    if params[6] == True:
        FileName += '_FET'
    else:
        FileName += '_FEF'

    # FD -> Feature engineering dependencies
    if params[7] == True:
        FileName += '_FDT'
    else:
        FileName += '_FDF'

    # DO -> drop outliers
    if params[8] == True:
        FileName += '_DOT'
    else:
        FileName += '_DOF'

    FileName += '.csv' 

    return FileName


def MakeResultsOutput (score, params, errMsg):

    output = str(score) + ','
    for i in range(9):
        output += str (params[i]) + ','
    output += MakeOutputFileName (params) + ','
    output += errMsg + '\n'
   
    return output 


#
# EvaluateModel... this does it all, that is it cleans the data, builds and trains a model, makes a prediction
#    with the test data.
#
def EvaluateModel (params):

    # Get the Train/Test data... depending on the various parameters, 
    XTrain, YTrain, XTestIDs, XTestValues = HousingDataUtils.GetTrainTestData (CatEncoding      = params[1],
                                                                               ScalingType      = params[2],
                                                                               LogSalesPrice    = params[3],
                                                                               DropLilVarFeats  = params[4], 
                                                                               DropLilCorrFeats = params[5],
                                                                               MakeNewFeatures  = params[6], 
                                                                               DropUsedFeatures = params[7],
                                                                               DropOutliers     = params[8])

    # Make the model...
    print ('\n>>> Building and Fitting Model...')
    model = BuildModel (params[0])
    model.fit (XTrain, YTrain)
    print ()

    # Score it...
    modelScore = (cross_val_score (model, XTrain, YTrain, cv=5)).mean()

    print ('>>> Mean cross val score: ', modelScore)
    print ()

    # Output coefficients (betas) and intercept if necessary
    if OUTPUT_COEFS:
        cols = XTrain.columns
        betas = model.coef_
        results = []
        for i in range(len(cols)):
            results.append ( [cols[i], betas[i]] )

        results = sorted (results, reverse=True, key=lambda x: x[1])

        coefsFile = open (COEFS_FILE, 'w')
        coefsFile.write ('Feature,Beta\n')

        print ('>>> Results')
        print (f'  Intercept: {model.intercept_:14.7f}')
        coefsFile.write ('Intercept,' + str(model.intercept_) + '\n')

        print (f'  Idx  {"Feature":25}   Beta')
        for i in range (len(results)):
            print (f'  {i:3}  {results[i][0]:25}  {results[i][1]:14.7f}')
            coefsFile.write (str(results[i][0]) + ',' + str(results[i][1]) + '\n')

        coefsFile.close ()

    # Predict the model
    print ('Predicting the model...')
    predYValues = model.predict (XTestValues)

    # output...
    outputFile = MakeOutputFileName (params)
    results, errMsg = HousingDataUtils.MergeResult (XTestIDs, predYValues, params[3], YTrain.mean())
    results.to_csv (outputFile, index=False)

    if errMsg != '':
        print ('>>> Errors')
        print ('   ', errMsg)

    return (modelScore, errMsg)


if FIND_OPTIMAL_PARAMS == True:

    paramCombos = list (itertools.product( [ 'Std', 'Ridge', 'Lasso', 'ElasticNet' ], # 0 - Model Type
                                           [ 'Label', 'OneHot' ],                     # 1 - Categorical Encoding Type
                                           [ 'Min', 'Std', 'None' ],                  # 2 - Scaling Type
                                           [ True, False ],                           # 3 - Take log of sales prices
                                           [ True, False ],                           # 4 - Drop features with little variation
                                           [ True, False ],                           # 5 - Drop features with little correlation to sales price
                                           [ True, False ],                           # 6 - Make new features; i.e. feature engineering
                                           [ True, False ],                           # 7 - Drop feature engineering dependencies
                                           [ True, False ] ))                         # 8 - Drop outliers

    resultsFile = open (RESULTS_FILE, 'a') 
    resultsFile.write ('Score,ModelType,CatEncoding,ScalingType,LogSales,DropLilVar,DropLilCorr,FeatEng,DropFEDepends,DropOuts,ResultsFile,Errors\n')

    bestScore  = 0
    bestParams = []

    i = 0
    for params in paramCombos:

        print (params)
        score, errMsg = EvaluateModel (params)
        
        if score > bestScore:
            bestScore  = score
            bestParams = params

        resultsFile.write ( MakeResultsOutput(score, params, errMsg) )

        i += 1

    resultsFile.close ()

    print ()
    print ('*** *** *** *** *** *** *** *** ***')
    print ('Best Score: ', bestScore)
    print ('Best Params: ')
    print ('   Model Type: ', bestParams[0])
    print ('   Categorical Encoding Type: ', bestParams[1])
    print ('   Scaling Type: ', bestParams[2])
    print ('   Log of Sales Price: ', bestParams[3])
    print ('   Drop features with little variation: ', bestParams[4])
    print ('   Drop features with little correlation to sales price: ', bestParams[5])
    print ('   Make new features: ', bestParams[6])
    print ('   Drop feature engineering dependencies: ', bestParams[7])
    print ('   Drop outliers: ', bestParams[8])
    print ('-> ', bestParams)
    print ('*** *** *** *** *** *** *** *** ***')
    print ()

else:

    # Evaluate the model using values from one of the better runs...
   
    EvaluateModel ( ('Ridge', 'OneHot', 'None', True, True, False, False, True, True) )


