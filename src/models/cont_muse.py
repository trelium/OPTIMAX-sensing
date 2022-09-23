# -*- coding: utf-8 -*-
"""WEASEL+MUSE classifier.

multivariate dictionary based classifier based on SFA transform, dictionaries
and logistic regression.
"""

__author__ = ["patrickzib", "BINAYKUMAR943"]
__all__ = ["MUSE"]

import math
import warnings

import numpy as np
from numba import njit
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state
from sklearn.ensemble import RandomForestRegressor

from sktime.classification.base import BaseClassifier
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from sktime.transformations.panel.dictionary_based import SFA


class MUSEregressor(BaseClassifier):
    """MUSE (MUltivariate Symbolic Extension).

    Also known as WEASLE-MUSE: implementation of multivariate version of WEASEL,
    referred to as just MUSE from [1].

    Overview: Input n series length m
     WEASEL+MUSE is a multivariate  dictionary classifier that builds a
     bag-of-patterns using SFA for different window lengths and learns a
     logistic regression classifier on this bag.

     There are these primary parameters:
             alphabet_size: alphabet size
             chi2-threshold: used for feature selection to select best words
             anova: select best l/2 fourier coefficients other than first ones
             bigrams: using bigrams of SFA words
             binning_strategy: the binning strategy used to disctrtize into
                               SFA words.

    Parameters
    ----------
    anova: boolean, default=True
        If True, the Fourier coefficient selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.
        Only applicable if labels are given
    bigrams: boolean, default=True
        whether to create bigrams of SFA words
    window_inc: int, default=2
        WEASEL create a BoP model for each window sizes. This is the
        increment used to determine the next window size.
     p_threshold: int, default=0.05 (disabled by default)
        Feature selection is applied based on the chi-squared test.
        This is the p-value threshold to use for chi-squared test on bag-of-words
        (lower means more strict). 1 indicates that the test
        should not be performed.
    use_first_order_differences: boolean, default=True
        If set to True will add the first order differences of each dimension
        to the data.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state: int or None, default=None
        Seed for random, integer

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    classes_ : list
        The classes labels.

    See Also
    --------
    WEASEL

    References
    ----------
    .. [1] Patrick Schäfer and Ulf Leser, "Multivariate time series classification
        with WEASEL+MUSE", in proc 3rd ECML/PKDD Workshop on AALTD}, 2018
        https://arxiv.org/abs/1711.11343

    Notes
    -----
    For the Java version, see
    `MUSE <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/tsml/
    classifiers/multivariate/WEASEL_MUSE.java>`_.

    Examples
    --------
    >>> from sktime.classification.dictionary_based import MUSE
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = MUSE(window_inc=4, use_first_order_differences=False)
    >>> clf.fit(X_train, y_train)
    MUSE(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "X_inner_mtype": "nested_univ",  # MUSE requires nested datafrane
        "classifier_type": "dictionary",
    }

    def __init__(
        self,
        anova=True,
        bigrams=True,
        window_inc=2,
        p_threshold=0.05,
        use_first_order_differences=True,
        n_jobs=1,
        random_state=None,
    ):

        # currently other values than 4 are not supported.
        self.alphabet_size = 4

        # feature selection is applied based on the chi-squared test.
        self.p_threshold = p_threshold
        self.anova = anova
        self.use_first_order_differences = use_first_order_differences

        self.norm_options = [False]
        self.word_lengths = [4, 6]

        self.bigrams = bigrams
        self.binning_strategies = ["equi-width", "equi-depth"]
        self.random_state = random_state

        self.min_window = 6
        self.max_window = 100

        self.window_inc = window_inc
        self.highest_bit = -1
        self.window_sizes = []

        self.col_names = []
        self.highest_dim_bit = 0
        self.highest_bits = []

        self.SFA_transformers = []
        self.clf = None
        self.n_jobs = n_jobs

        super(MUSEregressor, self).__init__()

    def _fit(self, X, y):
        """Build a WEASEL+MUSE classifiers from the training set (X, y).

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.
        """
        y = np.asarray(y)

        # add first order differences in each dimension to TS
        if self.use_first_order_differences:
            X = self._add_first_order_differences(X)

        # Window length parameter space dependent on series length
        self.col_names = X.columns

        rng = check_random_state(self.random_state)

        self.n_dims = len(self.col_names)
        self.highest_dim_bit = (math.ceil(math.log2(self.n_dims))) + 1
        self.highest_bits = np.zeros(self.n_dims)

        if self.n_dims == 1:
            warnings.warn(
                "MUSE Warning: Input series is univariate; MUSE is designed for"
                + " multivariate series. It is recommended WEASEL is used instead."
            )

        self.SFA_transformers = [[] for _ in range(self.n_dims)]

        # the words of all dimensions and all time series
        all_words = [dict() for _ in range(X.shape[0])]

        # On each dimension, perform SFA
        for ind, column in enumerate(self.col_names):
            X_dim = X[[column]]
            X_dim = from_nested_to_3d_numpy(X_dim)
            series_length = X_dim.shape[-1]  # TODO compute minimum over all ts ?

            # increment window size in steps of 'win_inc'
            win_inc = self._compute_window_inc(series_length)

            self.max_window = int(min(series_length, self.max_window))
            if self.min_window > self.max_window:
                raise ValueError(
                    f"Error in MUSE, min_window ="
                    f"{self.min_window} is bigger"
                    f" than max_window ={self.max_window}."
                    f" Try set min_window to be smaller than series length in "
                    f"the constructor, but the classifier may not work at "
                    f"all with very short series"
                )
            self.window_sizes.append(
                list(range(self.min_window, self.max_window, win_inc))
            )

            self.highest_bits[ind] = math.ceil(math.log2(self.max_window)) + 1

            for window_size in self.window_sizes[ind]:
                transformer = SFA(
                    word_length=rng.choice(self.word_lengths),
                    alphabet_size=self.alphabet_size,
                    window_size=window_size,
                    norm=rng.choice(self.norm_options),
                    anova=self.anova,
                    binning_method=rng.choice(self.binning_strategies),
                    bigrams=self.bigrams,
                    remove_repeat_words=False,
                    lower_bounding=False,
                    save_words=False,
                    n_jobs=self._threads_to_use,
                )

                sfa_words = transformer.fit_transform(X_dim, y)

                self.SFA_transformers[ind].append(transformer)
                bag = sfa_words[0]

                # chi-squared test to keep only relevant features
                relevant_features = {}
                apply_chi_squared = self.p_threshold < 1
                if apply_chi_squared:
                    vectorizer = DictVectorizer(sparse=True, dtype=np.int32, sort=False)
                    bag_vec = vectorizer.fit_transform(bag)

                    f_statistics, p = f_regression(bag_vec, y)
                    relevant_features_idx = np.where(p <= self.p_threshold)[0]
                    relevant_features = set(
                        np.array(vectorizer.feature_names_)[relevant_features_idx]
                    )

                # merging bag-of-patterns of different window_sizes
                # to single bag-of-patterns with prefix indicating
                # the used window-length
                highest = np.int32(self.highest_bits[ind])
                for j in range(len(bag)):
                    for (key, value) in bag[j].items():
                        # chi-squared test
                        if (not apply_chi_squared) or (key in relevant_features):
                            # append the prefices to the words to
                            # distinguish between window-sizes
                            word = MUSEregressor._shift_left(
                                key, highest, ind, self.highest_dim_bit, window_size
                            )
                            all_words[j][word] = value

        self.clf = make_pipeline(
            DictVectorizer(sparse=True, sort=False),
            # StandardScaler(with_mean=True, copy=False),
            RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=self._threads_to_use,
            ),
        )

        for words in all_words:
            if len(words) == 0:
                words[-1] = 1

        self.clf.fit(all_words, y)

        return self

    def _predict(self, X) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        bag = self._transform_words(X)
        return self.clf.predict(bag)

    def _predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for n instances in X.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        bag = self._transform_words(X)
        return self.clf.predict_proba(bag)

    def _transform_words(self, X):
        if self.use_first_order_differences:
            X = self._add_first_order_differences(X)

        bag_all_words = [dict() for _ in range(len(X))]

        # On each dimension, perform SFA
        for ind, column in enumerate(self.col_names):
            X_dim = X[[column]]
            X_dim = from_nested_to_3d_numpy(X_dim)

            for i, window_size in enumerate(self.window_sizes[ind]):

                # SFA transform
                sfa_words = self.SFA_transformers[ind][i].transform(X_dim)
                bag = sfa_words[0]

                # merging bag-of-patterns of different window_sizes
                # to single bag-of-patterns with prefix indicating
                # the used window-length
                highest = np.int32(self.highest_bits[ind])
                for j in range(len(bag)):
                    for (key, value) in bag[j].items():
                        # append the prefices to the words to distinguish
                        # between window-sizes
                        word = MUSEregressor._shift_left(
                            key, highest, ind, self.highest_dim_bit, window_size
                        )
                        bag_all_words[j][word] = value

        return bag_all_words

    def _add_first_order_differences(self, X):
        X_copy = X.copy()
        for column in X.columns:
            X_copy[str(column) + "_diff"] = X_copy[column]
            for ts in X[column]:
                ts_diff = ts.diff(1)
                ts.replace(ts_diff)
        return X_copy

    def _compute_window_inc(self, series_length):
        win_inc = self.window_inc
        if series_length < 100:
            win_inc = 1  # less than 100 is ok time-wise
        return win_inc

    def score(self, X, y) -> float:
        """Computes R^2

        Parameters
        ----------
        X : 3D np.array (any number of dimensions, equal length series)
                of shape [n_instances, n_dimensions, series_length]
            or 2D np.array (univariate, equal length series)
                of shape [n_instances, series_length]
            or pd.DataFrame with each column a dimension, each cell a pd.Series
                (any number of dimensions, equal or unequal length series)
            or of any other supported Panel mtype
                for list of mtypes, see datatypes.SCITYPE_REGISTER
                for specifications, see examples/AA_datatypes_and_datasets.ipynb
        y : 1D np.ndarray of int, of shape [n_instances] - class labels (ground truth)
            indices correspond to instance indices in X

        Returns
        -------
        float, R-squared
        """
        from sklearn.metrics import r2_score

        self.check_is_fitted()

        return r2_score(y, self.predict(X))
    
    @staticmethod
    @njit(fastmath=True, cache=True)
    def _shift_left(key, highest, ind, highest_dim_bit, window_size):
        return ((key << highest | ind) << highest_dim_bit) | window_size

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return {"window_inc": 4, "use_first_order_differences": False}


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
                    'HAMA','OASIS','BAI', 
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
musre = MUSEregressor(n_jobs=-1, random_state=67)

models = [musre] #ars_multi not working

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
                X = df['x_pre'].astype(np.float32) #[:,4,:]
                Y = np.array(df['y_pre']).astype(np.float32)
                print(f'---Dataset {file}---')
                #kf = KFold(n_splits=10)
                X_train, X_test, y_train, y_test = train_test_split(
                                                    X, Y, test_size=0.33, random_state=42)
                #clf=MUSEregressor()
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