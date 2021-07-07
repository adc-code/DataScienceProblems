#
# HousingModel_XGBoost.py
#
# XGBoost version...
#



from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb


import pandas as pd
import numpy as np
import itertools

import HousingDataUtils
import BigNums


FIND_OPTIMAL_PARAMS = False #True

EXPLORE_ITERS       = 10
FINETUNE_ITERS      = 50

CROSS_VAL_FOLDS     = 3
MAX_THREADS         = 2
RANDOM_SEED         = 42

OUTPUT_FEAT_IMP_1   = True
OUTPUT_FEAT_IMP_2   = True
FEAT_IMP_FILE_1     = 'FeatureImportance_XGBoost_IntMeth.csv'
FEAT_IMP_FILE_2     = 'FeatureImportance_XGBoost_Permutations.csv'

BASE_OUTPUT_NAME    = 'Results/TestPreds_XGB'
RESULTS_FILE        = 'Results_XGBoost.csv'



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
    output += str(modelParams['subsample']) + ','
    output += str(modelParams['colsample_bytree']) + ','

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

    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1100, num = 10)]

    # Maximum number of levels in tree
    max_depth = max_depth = [int(x) for x in np.linspace(3, 11, num = 8)]
    max_depth.append (None)

    # Number of features (columns) used in each tree
    colsample_bytree = [ 0.00, 0.20, 0.40, 0.60, 0.80, 1.00 ]

    # Minimum number of samples required at each leaf node
    subsample = [ 0.00, 0.20, 0.40, 0.60, 0.80, 1.00 ]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'subsample': subsample,
                   'colsample_bytree': colsample_bytree}

    # Do a randomized search of the various parameters...
    model = RandomizedSearchCV (estimator = xgb.XGBRegressor(),
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

    if OUTPUT_FEAT_IMP_1:

        print ('\n>>> Getting feature importance results...')
        featImpFile = open (FEAT_IMP_FILE_1, 'w')
        featImpFile.write ('Features,Importance\n')

        importance = model.best_estimator_.feature_importances_
        feats = XTrain.columns

        for i in range (len(importance)):
            print (f'{feats[i]:25} {importance[i]:14.7f}')
            featImpFile.write (feats[i] + ',' + str(importance[i]) + '\n')

        featImpFile.close ()

    # Output feature importance is necessary...
    if OUTPUT_FEAT_IMP_2:

        print ('\n>>> Finding feature importance...')
        featImpFile = open (FEAT_IMP_FILE_2, 'w')
        featImpFile.write ('Features,ImportanceMean,ImportanceStdDev\n')

        featImportanceResults = HousingDataUtils.FindFeatureImportance (model, XTrain, YTrain, 20, 1)

        for i in range(len(featImportanceResults)):
            print (f'{featImportanceResults[i][0]:25} {(100 * featImportanceResults[i][1]):14.7f} {featImportanceResults[i][2]:14.7f}')
            featImpFile.write (featImportanceResults[i][0] + ',' +\
                               str(featImportanceResults[i][1]) + ',' +\
                               str(featImportanceResults[i][2]) + '\n')

        featImpFile.close ()

    # Predict the model
    print ('Predicting the model...')
    predYValues = model.best_estimator_.predict (XTestValues)

    # output...
    outputFile = MakeOutputFileName (params)
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
    # ('Label', 'Std', False, False, False, True, True, True)
    # {'subsample': 0.6, 'n_estimators': 322, 'max_depth': 3, 'colsample_bytree': 0.4}

    # EvaluateModel ( ('Label', 'Std', False, False, False, True, True, True), FINETUNE_ITERS )
    EvaluateModel ( ('Label', 'Std', True, True, True, False, False, False), FINETUNE_ITERS )


