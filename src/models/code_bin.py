#Train and test on PRE segments to classify as responder/nonresponder 
import os 
from sklearn.model_selection import KFold
import pandas as pd 
import numpy as np
from sktime.classification.feature_based import Catch22Classifier
from sklearn.ensemble import RandomForestClassifier
from sktime.transformations.panel.catch22 import Catch22
from sktime.utils.plotting import plot_series, plot_lags,plot_correlations
from sktime.datatypes import check_is_mtype

def create_targets(X, Y):
    Y_OUT=[]
    for i in X.index.unique(level=0):
        Y_OUT.append(int(Y.loc[Y['pid']==i]['RCI_responder'].unique()[0]))
    return Y_OUT

def make_x3d(X,n_periods=1):
    n_instances = len(X.index.get_level_values(0).unique())
    n_timepoints = 358*n_periods  #len(X_PRE.index.get_level_values(1).unique())
    n_columns = X.shape[1]
    X_3d = X.values.reshape(n_instances, n_timepoints, n_columns).swapaxes(1, 2)
    return X_3d

def parse_dataset(df):
    Y= df[['RCI_responder','pid']]

    #drop useless columns, collect target info
    excluded_cols = [#'Unnamed: 0.1',
                    'Unnamed: 0',
                    "X","Unnamed..0",
                    'progsegment',
                    'percentage_change_hama','percentage_change_oasis','percentage_change_bai',
                    'hama_nonclinical_post','oasis_nonclinical_post','bai_nonclinical_post',
                    'OASIS_RCI_responder',
                    'BAI_RCI_responder',
                    'HAMA_RCI_responder',
                    #'HAMA','OASIS','BAI',
                    'RCI_responder',
                    #'pid',
                    'segment_end_date',
                    'local_segment',
                    'local_segment_label',
                    #'local_segment_start_datetime',
                    'local_segment_end_datetime',
                    'phone_data_yield_rapids_ratiovalidyieldedminutes',
                    #'sensing_period',
                    'exp_group']

    df = df.drop(columns=excluded_cols)
    #set timestamp and pid column as index
    df['local_segment_start_datetime'] = pd.to_datetime(df['local_segment_start_datetime'], infer_datetime_format = True)
    df.set_index(['pid','local_segment_start_datetime'],verify_integrity = True,inplace=True,drop=True)

    X_PRE = df.sort_index(level='local_segment_start_datetime').loc[df['sensing_period']=='PRE'].drop(columns=['sensing_period'])
    X_MID = df.sort_index(level='local_segment_start_datetime').loc[df['sensing_period']=='MID'].drop(columns=['sensing_period'])
    X_POST = df.sort_index(level='local_segment_start_datetime').loc[df['sensing_period']=='POST'].drop(columns=['sensing_period'])
    
    X_PREMID = df.sort_index(level='local_segment_start_datetime').loc[(df['sensing_period']=='PRE') | (df['sensing_period']=='MID')].drop(columns=['sensing_period'])
    mid_participants = X_MID.index.get_level_values(0).unique()
    X_PREMID = X_PREMID.loc[list(mid_participants)] ##only select pre participants that also have a mid 

    X_3d_pre = make_x3d(X_PRE)
    X_3d_mid = make_x3d(X_MID)
    X_3d_post = make_x3d(X_POST)
    X_3d_premid = make_x3d(X_PREMID,n_periods=2)

    Y_PRE = create_targets(X_PRE,Y)
    Y_MID = create_targets(X_MID,Y)
    Y_POST = create_targets(X_POST,Y)
    Y_PREMID = create_targets(X_PREMID,Y)

    ret = {'x_pre':X_3d_pre,
            'x_mid':X_3d_mid,
            'x_post':X_3d_post,
            'x_premid':X_3d_premid,
            'y_pre': Y_PRE,
            'y_mid':Y_MID,
            'y_post':Y_POST,
            'y_premid':Y_PREMID
             }
    
    return ret


#check_is_mtype(X_PRE,mtype='pd-multiindex',return_metadata=True)
#check_is_mtype(X_MID,mtype='pd-multiindex',return_metadata=True)
#check_is_mtype(X_POST,mtype='pd-multiindex',return_metadata=True)


#Train and test on PRE segments to classify as responder/nonresponder 
import os 
from sklearn.model_selection import train_test_split
from sktime.classification.kernel_based import Arsenal
from sktime.classification.dictionary_based import MUSE
from sktime.classification.dictionary_based import  TemporalDictionaryEnsemble
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.feature_based import Catch22Classifier
from sktime.classification.feature_based import TSFreshClassifier
from sktime.classification.feature_based import FreshPRINCE

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, roc_auc_score

#from sktime.classification.tests import DummyClassifier

#BOSS,weasel,ProximityForest,TimeSeriesForestClassifier,RandomIntervalSpectralEnsemble,MatrixProfileClassifier             cannot be used  in multivariate 
# HIVECOTEV2 (because of ShapeletTransformClassifier),  is buggy 
# TSFreshClassifier           needs porting to new statsmodels 
# CanonicalIntervalForest has division by zero problem 
#ShapeletTransformClassifier        Cannot unify array(float64, 1d, C) and array(float32, 1d, C) for 'X_n.2', defined at /home/jmocel/.conda/envs/SktimeEnv/lib/python3.9/site-packages/sktime/utils/numba/general.py (39)
# drCIF ValueError: Input contains NaN, infinity or a value too large for dtype('float32').          
#TemporalDictionaryEnsemble,IndividualTDE has pickling problems, to be tried without parallel execution
#SignatureClassifier signatures do not scale well to large input dimensions.


#clf =  TemporalDictionaryEnsemble(n_jobs= -1,random_state = 67)
#clf =  FreshPRINCE(random_state=67,n_jobs=-1) #based on tsfresh features, funny if it worked

ars_multi =  Arsenal(rocket_transform = 'multirocket', time_limit_in_minutes = 4, n_jobs= -1, #use all processors
                     random_state = 67) 
ars_mini =  Arsenal(n_jobs= -1,random_state=67,rocket_transform='minirocket')
muse = MUSE( n_jobs= -1, random_state = 67) 
catch = Catch22Classifier(n_jobs= -1,random_state=67) #TODO debug /home/jmocel/.conda/envs/SktimeEnv/lib/python3.9/site-packages/sktime/transformations/panel/catch22.py
knn_one = KNeighborsTimeSeriesClassifier(n_neighbors=1, n_jobs= -1) 
knn_three = KNeighborsTimeSeriesClassifier(n_neighbors=3, n_jobs= -1) 
shapelet = ShapeletTransformClassifier( n_jobs= -1, random_state = 67) 

models = [muse,ars_mini,knn_one,knn_three,catch] #, shapelet
#ars_multi not working 
#ars_mini,knn_one,knn_three,catch, muse,

for i, clf in enumerate(models): 
    scores_in = []
    auc_scores = []
    precision_scores = []
    recall_scores = []
    print(f'Model: {clf}')
    DATA_PATH = '/home/jmocel/trelium/OPTIMAX-sensing/imputations_0.01_tol'
    #try: 
    if i != 0: #modify to selectively run just few models 
        break
    for (_,_,files) in os.walk(DATA_PATH):
        for file in files:
            if file.endswith('.csv'):
                raw_df = pd.read_csv(os.path.join(DATA_PATH,file))
                df = parse_dataset(raw_df)
                if i==5:
                    X = df['x_pre'] #.astype(np.float32) #need to disable this when running shapelet
                    Y = np.array(df['y_pre']).astype(np.int32)
                elif i==4:
                    X = df['x_pre'].astype(np.float32) #need to disable this when running shapelet
                    Y = np.array(df['y_pre']) #.astype(np.int32)
                elif i>=0 and i<4:
                    X = df['x_pre']#.astype(np.float32) #need to disable this when running shapelet
                    Y = np.array(df['y_pre'])#.astype(np.int32)
                
                print(f'---Dataset {file}---')
                #kf = KFold(n_splits=10)
                X_train, X_test, y_train, y_test = train_test_split(
                                                    X, Y, test_size=0.33, random_state=42)
                clf.fit(X_train, y_train)
                y_pred= clf.predict(X_test)
                #if i==0:
                #    class_summary = classification_report(y_true = y_test, y_pred=y_pred, output_dict=True)
                #else: 
                #    res = classification_report(y_true = y_test, y_pred=y_pred, output_dict=True)
                #    for key in res:
                #        for j in res[key]:
                 #           class_summary[key][j] += res[key]
                scores_in.append(clf.score(X_test,y_test)) #accuracy score
                precision_scores.append(precision_score(y_true = y_test, y_pred=y_pred,labels = [0,1],average='micro'))
                recall_scores.append(recall_score(y_true = y_test, y_pred=y_pred,labels = [0,1],average='micro'))
                auc_scores.append(roc_auc_score(y_true = y_test, y_score=clf.predict_proba(X_test)[:, 1],labels = [0,1],average='micro'))
                print(f'Dataset accuracy: {round(scores_in[-1],4)}')

    overall_accuracy = sum(scores_in) / len(scores_in) 
    print('----------------------------------------------')
    print(f'Overall Accuracy: {round(overall_accuracy,4)}')
    print(f'Overall Precision: {round( sum(precision_scores) / len(precision_scores),4)}')
    print(f'Overall Recall: {round( sum(recall_scores) / len(recall_scores),4)}')
    print(f'Overall AUC: {round( sum(auc_scores) / len(auc_scores),4)}')
    #for key in class_summary:
    #    for j in class_summary[key]:
    #        class_summary[key][j] = class_summary[key][j]/5
    #print(f'Overall classification report:\n{class_summary}')
    
    #except:
    #    print(f'ERROR, could not fit model {clf}')