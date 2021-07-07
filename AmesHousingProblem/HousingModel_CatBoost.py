#
# HousingModel_CatBoost.py
#
# Version that uses CATBoost...
#



from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from catboost import CatBoostRegressor

import pandas as pd
import numpy as np
import itertools

import HousingDataUtils
import BigNums


FIND_OPTIMAL_PARAMS = False #True

EXPLORE_ITERS       = 10
FINETUNE_ITERS      = 5

CROSS_VAL_FOLDS     = 3
MAX_THREADS         = 2
RANDOM_SEED         = 42

OUTPUT_FEAT_IMP_1   = True
OUTPUT_FEAT_IMP_2   = True
FEAT_IMP_FILE_1     = 'FeatureImportance_CatBoost_IntMeth.csv' 
FEAT_IMP_FILE_2     = 'FeatureImportance_CatBoost_Permutations.csv'

BASE_OUTPUT_NAME    = 'Results/TestPreds_CatBoost'
RESULTS_FILE        = 'Results_CatBoost.csv'



#
# MakeFileName
#
def MakeOutputFileName (params):

    FileName = BASE_OUTPUT_NAME

    # 0 - Categorical Encoding Type
    FileName += '_' + str(params[0])

    # 1 - Scaling Type
    FileName += '_' + str(params[1])

    # 2 - Take log of sales prices
    if params[2] == True:
        FileName += '_LogT'
    else:
        FileName += '_LogF'

    # 3 - Drop features with little variation
    if params[3] == True:
        FileName += '_LVT'
    else:
        FileName += '_LVF'

    # 4 - Drop features with little correlation to sales price
    if params[4] == True:
        FileName += '_LCT'
    else:
        FileName += '_LCF'

    # 5 - Make new features; i.e. feature engineering
    if params[5] == True:
        FileName += '_FET'
    else:
        FileName += '_FEF'

    # 6 - Drop feature engineering dependencies
    if params[6] == True:
        FileName += '_FDT'
    else:
        FileName += '_FDF'

    # 7 - Drop outliers    
    if params[7] == True:
        FileName += '_DOT'
    else:
        FileName += '_DOF'

    FileName += '.csv' 

    return FileName


#
# MakeResultsOutput... make an one line entry into the csv file with the current parameters...
#
#def MakeResultsOutput (score, dataParams, modelParams, errMsg):
def MakeResultsOutput (score, dataParams, errMsg):

    output = str(score) + ','
    for param in dataParams:
        output += str (param) + ','

    #output += str(modelParams['n_estimators']) + ','
    #output += str(modelParams['max_depth']) + ','
    #output += str(modelParams['subsample']) + ','
    #output += str(modelParams['colsample_bytree']) + ','

    output += MakeOutputFileName (dataParams) + ','

    output += errMsg + '\n'
   
    return output 


#
# EvaluateModel... this does it all, that is it cleans the data, builds and trains a model, makes a prediction
#    with the test data.
#
def EvaluateModel (params, maxIterations):

    # Get the Train/Test data... depending on the various parameters, 
    XTrain, YTrain, XTestIDs, XTestValues = HousingDataUtils.GetTrainTestData (CatEncoding      = params[0],
                                                                               ScalingType      = params[1],
                                                                               LogSalesPrice    = params[2],
                                                                               DropLilVarFeats  = params[3], 
                                                                               DropLilCorrFeats = params[4],
                                                                               MakeNewFeatures  = params[5], 
                                                                               DropUsedFeatures = params[6],
                                                                               DropOutliers     = params[7])

    # for now just use the 'vanilla' version since it does produce rather good results
    model = CatBoostRegressor()

    # Fit the random search model
    print ()
    print ('>>> Fitting the model...')
    model.fit (XTrain, YTrain)
    print ()

    # get the model score... or loss and mean square error
    modelScore = cross_val_score (model, XTrain, YTrain, cv=CROSS_VAL_FOLDS).mean()
    print ('>>> modelScore = ', modelScore)
    print ()

    if OUTPUT_FEAT_IMP_1:

        print ('\n>>> Getting feature importance results...')

        featImpFile = open (FEAT_IMP_FILE_1, 'w')
        featImpFile.write ('Feature,Importance\n')

        importance = model.feature_importances_
        feats = XTrain.columns

        for i in range (len(importance)):
            print (f'{feats[i]:25} {importance[i]:14.7f}')
            featImpFile.write (feats[i] + ',' + str(importance[i]) + '\n')

        featImpFile.close ()

    # Output feature importance is necessary...
    if OUTPUT_FEAT_IMP_2:

        print ('\n>>> Finding feature importance...')

        featImpFile = open (FEAT_IMP_FILE_2, 'w')
        featImpFile.write ('Feature,ImportanceMean,ImportanceStdDev\n')

        featImportanceResults = HousingDataUtils.FindFeatureImportance (model, XTrain, YTrain, 20, 100)

        for i in range(len(featImportanceResults)):
            print (f'{featImportanceResults[i][0]:25} {featImportanceResults[i][1]:14.7f} {featImportanceResults[i][2]:14.7f}')
            featImpFile.write (feats[i] + ',' + str(featImportanceResults[i][1]) + ',' + str(featImportanceResults[i][2]) + '\n')

        featImpFile.close ()

    # Predict the model
    print ('Predicting the model...')
    predYValues = model.predict (XTestValues)

    # output...
    outputFile = MakeOutputFileName (params)
    results, errMsg = HousingDataUtils.MergeResult (XTestIDs, predYValues, params[2], YTrain.mean())
    results.to_csv (outputFile, index=False)

    if errMsg != '':
        print ('>>> Errors')
        print ('   ', errMsg)

    return (modelScore, errMsg)



if FIND_OPTIMAL_PARAMS == True:

    paramCombos = list (itertools.product( [ 'Label' ],                # 0 - Categorical Encoding Type... [ 'Label', 'OneHot' ]
                                           [ 'Min', 'Std', 'None' ],   # 1 - Scaling Type
                                           [ True, False ],            # 2 - Take log of sales prices
                                           [ True, False ],            # 3 - Drop features with little variation
                                           [ True, False ],            # 4 - Drop features with little correlation to sales price
                                           [ True, False ],            # 5 - Make new features; i.e. feature engineering
                                           [ True, False ],            # 6 - Drop feature engineering dependencies
                                           [ True, False ] ))          # 7 - Drop outliers

    resultsFile = open (RESULTS_FILE, 'a') 
    resultsFile.write ('Score,CatEncoding,ScalingType,LogSales,DropLilVar,DropLilCorr,FeatEng,DropFEDepends,DropOuts,')
    resultsFile.write ('Estimators,MaxDepth,SubSample,ColSample_bytreeBootstrap,ResultsFile,Errors\n')

    bestScore  = 0
    bestDataParams  = []
    bestModelParams = {}

    i = 1
    for params in paramCombos:

        print ('**** Iteration: ')
        print (BigNums.IntToNumFont (i))
        print ('of ', len(paramCombos), '\n')

        print (params)

        #modelScore, bestParams, errMsg = EvaluateModel (params, EXPLORE_ITERS)
        modelScore, errMsg = EvaluateModel (params, EXPLORE_ITERS)
       
        print ('>>> Score: ', modelScore)
        if modelScore > bestScore:
            bestScore       = modelScore
            #bestModelParams = bestParams
            bestDataParams  = params

        #resultsFile.write ( MakeResultsOutput(modelScore, params, bestParams, errMsg) )
        resultsFile.write ( MakeResultsOutput(modelScore, params, errMsg) )

        i += 1

    resultsFile.close ()

    print ()
    print ('*** *** *** *** *** *** *** *** ***')
    print ('Best Score: ', bestScore)
    print ('Best Params: ')
    print ('  data params  -> ', bestDataParams)
    print ('  model params -> ', bestModelParams)
    print ('*** *** *** *** *** *** *** *** ***')
    print ()

else:
    
    # Evaluate the model once using previously found values...
    # data params  -> ('Label', 'Std', True, False, False, True, False, True)
    # model params -> {'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 130, 'bootstrap': False}

    EvaluateModel ( ('Label', 'Std', True, False, False, True, False, True), FINETUNE_ITERS )

