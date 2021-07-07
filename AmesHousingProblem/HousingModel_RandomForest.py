#
# HousingModel_RandomForest.py
#
# Using a random forest, this code tries different data configurations with different 
# model parameters in an attempt to find a combination with an 'ok' score.   
#



from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


import pandas as pd
import numpy as np
import itertools

import HousingDataUtils
import BigNums


FIND_OPTIMAL_PARAMS = False #True

EXPLORE_ITERS       = 8
FINETUNE_ITERS      = 4

CROSS_VAL_FOLDS     = 3
MAX_THREADS         = 2
RANDOM_SEED         = 42

OUTPUT_FEAT_IMP_1   = True
OUTPUT_FEAT_IMP_2   = True
FEAT_IMP_FILE_1     = 'FeatureImportance_RandomForest_InterMeth.csv'     # InterMeth -> internal method
FEAT_IMP_FILE_2     = 'FeatureImportance_RandomForest_Permutations.csv'

BASE_OUTPUT_NAME    = 'Results/TestPreds_RF'
RESULTS_FILE        = 'Results_RandomForest.csv'



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
def MakeResultsOutput (score, dataParams, modelParams, errMsg):

    output = str(score) + ','
    for param in dataParams:
        output += str (param) + ','

    output += str(modelParams['n_estimators']) + ','
    output += str(modelParams['max_depth']) + ','
    output += str(modelParams['min_samples_split']) + ','
    output += str(modelParams['min_samples_leaf']) + ','
    output += str(modelParams['bootstrap']) + ','

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

    n_estimators = [ 200, 400, 600, 800, 1000 ]

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']

    # Maximum number of levels in tree
    max_depth = [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130 ] 
    max_depth.append (None)

    # Minimum number of samples required to split a node
    min_samples_split = [ 2, 5, 10, 15 ]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [ 1, 2, 4, 8, 16 ]

    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators':      n_estimators,
                   'max_features':      max_features,
                   'max_depth':         max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf':  min_samples_leaf,
                   'bootstrap':         bootstrap }


    # Make the model...
    rfr = RandomForestRegressor()

    # Do a randomized search of the various parameters...
    model = RandomizedSearchCV (estimator = rfr, 
                                param_distributions = random_grid, 
                                n_iter = maxIterations, 
                                cv = CROSS_VAL_FOLDS, 
                                verbose=2, 
                                random_state = RANDOM_SEED, 
                                n_jobs = MAX_THREADS)

    # Fit the random search model
    print ()
    print ('>>> Fitting the model...')
    model.fit (XTrain, YTrain)
    print ()

    # get the model score... or loss and mean square error
    modelScore = model.best_score_  
    print ('>>> modelScore = ', modelScore)
    print ()

    # Get the feature importance using the internal method
    if OUTPUT_FEAT_IMP_1:
        featImpFile = open (FEAT_IMP_FILE_1, 'w')
        featImpFile.write ('Feature,Importance\n')

        featureImportance = model.best_estimator_.feature_importances_
        cols = XTrain.columns
        for i in range(len(XTrain.columns)):
            print (f'{cols[i]:25} {(100*featureImportance[i]):8.4f}')
            featImpFile.write (cols[i] + ',' + str(100*featureImportance[i]) + '\n')

        featImpFile.close ()

    # Find the feature importance using the permutation method...
    if OUTPUT_FEAT_IMP_2:
        featImpFile = open (FEAT_IMP_FILE_2, 'w')
        featImpFile.write ('Feature,MeanImportance,StdDevImportance\n')
        
        print ('>>> Finding feature importance...')
        featImportanceResults = HousingDataUtils.FindFeatureImportance (model, XTrain, YTrain, 5, 1)

        for i in range(len(featImportanceResults)):
            print (f'{featImportanceResults[i][0]:25} {(100*featImportanceResults[i][1]):14.7f} {featImportanceResults[i][2]:14.7f}')
            featImpFile.write (cols[i] + ',' + str(featImportanceResults[i][1]) + ',' + str(featImportanceResults[i][2]) + '\n')

        featImpFile.close ()

    # Predict the model
    print ('\n>>>Predicting the model...')
    predYValues = model.best_estimator_.predict (XTestValues)

    # output...
    outputFile = MakeOutputFileName (params)
    print ('\n>>>Writing to file...', outputFile)
    results, errMsg = HousingDataUtils.MergeResult (XTestIDs, predYValues, params[2], YTrain.mean())
    results.to_csv (outputFile, index=False)

    if errMsg != '':
        print ('>>> Errors')
        print ('   ', errMsg)

    return (modelScore, model.best_params_, errMsg)



if FIND_OPTIMAL_PARAMS == True:

    paramCombos = list (itertools.product( [ 'Label', 'OneHot' ],      # 0 - Categorical Encoding Type
                                           [ 'Min', 'Std', 'None' ],   # 1 - Scaling Type
                                           [ True, False ],            # 2 - Take log of sales prices
                                           [ True, False ],            # 3 - Drop features with little variation
                                           [ True, False ],            # 4 - Drop features with little correlation to sales price
                                           [ True, False ],            # 5 - Make new features; i.e. feature engineering
                                           [ True, False ],            # 6 - Drop feature engineering dependencies
                                           [ True, False ] ))          # 7 - Drop outliers

    resultsFile = open (RESULTS_FILE, 'a') 
    resultsFile.write ('Score,CatEncoding,ScalingType,LogSales,DropLilVar,DropLilCorr,FeatEng,DropFEDepends,DropOuts,')
    resultsFile.write ('Estimators,MaxDepth,MinSamplesSplit,MinSamplesLeaf,Bootstrap,ResultsFile,Errors\n')

    bestScore  = 0
    bestDataParams  = []
    bestModelParams = {}

    i = 1
    for params in paramCombos:

        print ('**** Iteration: ')
        print (BigNums.IntToNumFont (i))
        print ('of ', len(paramCombos), '\n')

        print (params)

        modelScore, bestParams, errMsg = EvaluateModel (params, EXPLORE_ITERS)
       
        print ('>>> Score: ', modelScore)
        if modelScore > bestScore:
            bestScore       = modelScore
            bestModelParams = bestParams
            bestDataParams  = params

        resultsFile.write ( MakeResultsOutput(modelScore, bestDataParams, bestModelParams, errMsg) )

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

