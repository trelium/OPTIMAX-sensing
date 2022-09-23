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
        Y_OUT.append(int(Y.loc[Y['pid']==i]['avg_perc_change'].unique()[0]))
    return Y_OUT

def make_x3d(X,n_periods=1):
    n_instances = len(X.index.get_level_values(0).unique())
    n_timepoints = 358*n_periods  #len(X_PRE.index.get_level_values(1).unique())
    n_columns = X.shape[1]
    X_3d = X.values.reshape(n_instances, n_timepoints, n_columns).swapaxes(1, 2)
    return X_3d

def parse_dataset(df):
    df['avg_perc_change'] = df['percentage_change_hama']+df['percentage_change_oasis']+df['percentage_change_bai']
    df['avg_perc_change'] = df['avg_perc_change']/3
    Y= df[['avg_perc_change','pid']]

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
from sktime.regression.all import TimeSeriesForestRegressor

from sklearn.metrics import r2_score, mean_squared_error

tsfr = TimeSeriesForestRegressor(n_jobs=-1, random_state=67)
models = [tsfr] #ars_multi not working

for i, clf in enumerate(models): 
    scores_in = []
    rmse_scores = []
    print(f'Model: {clf}')
    DATA_PATH = '/home/jmocel/trelium/OPTIMAX-sensing/imputations_0.01_tol'
    #try: 
    for (_,_,files) in os.walk(DATA_PATH):
        for file in files:
            if file.endswith('.csv'):
                raw_df = pd.read_csv(os.path.join(DATA_PATH,file))
                df = parse_dataset(raw_df)
                X = df['x_pre'][:,4,:].astype(np.float32)
                Y = np.array(df['y_pre']).astype(np.float32)
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
                rmse_scores.append(mean_squared_error(y_true = y_test, y_pred=y_pred))
                print(f'Dataset R2: {round(scores_in[-1],4)}')

    overall_r2 = sum(scores_in) / len(scores_in) 
    print('----------------------------------------------')
    print(f'Overall R2: {round(overall_r2,4)}')
    print(f'Overall RMSE: {round( sum(rmse_scores) / len(rmse_scores),4)}')
    #for key in class_summary:
    #    for j in class_summary[key]:
    #        class_summary[key][j] = class_summary[key][j]/5
    #print(f'Overall classification report:\n{class_summary}')
    
    #except:
    #    print(f'ERROR, could not fit model {clf}')