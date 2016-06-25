import pandas as pd
import csv
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np 
import matplotlib.pyplot as plt
import pylab
import sys
import random
from sklearn import svm, ensemble
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn.cross_validation import train_test_split, KFold
from sklearn.preprocessing import *
from sklearn.feature_selection import RFE
from sklearn.grid_search import ParameterGrid
from multiprocessing import Pool
from functools import partial
from time import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

plt.rcParams["figure.figsize"] = [18.0, 8.0]


clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=0),
    'LR': LogisticRegression(random_state=0, n_jobs=-1),
    'SVM': svm.LinearSVC(random_state=0, dual= False),
    'NB': GaussianNB(),
    'DT': DecisionTreeClassifier(random_state = 0),
    'KNN': KNeighborsClassifier(n_jobs = -1),
    'GB': GradientBoostingClassifier(random_state = 0)

        }

grid = { 
'RF':{'n_estimators': [1,10,100], 'max_depth': [1,5,10,20,50,75], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,5]},
'NB' : {},
'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1], 'penalty': ['l1', 'l2']},
'GB': {'n_estimators': [1,10,50], 'learning_rate' : [0.01,0.05,0.1],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10]},
'KNN' :{'n_neighbors': [1, 3, 5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
       }

def read_data(file_name):
    '''
    Read in data and return a pandas df
    '''
    return pd.read_csv(file_name, header=0)

def print_statistics(data):
    '''
    Given a pandas dataframe, print dataframe statistics, correlation, and missing data.
    '''
    pd.set_option('display.width', 20)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    print '**** column names:  ', "\n", data.columns.values
    print '**** top of the data: ', "\n",  data.head()
    print '**** dataframe shape: ', "\n", data.shape
    print '**** statistics: ', "\n", data.describe(include='all')
    print '**** MODE: ', "\n", data.mode()
    print '**** sum of null values by column: ', "\n", data.isnull().sum()
    print '**** correlation matrix: ', "\n", data.corr()

def print_value_counts(data, col):
    '''
    For a given column in the data, print the counts 
    of the column's values.
    '''
    print pd.value_counts(data[col])

def visualize_all(data):
    '''
    Given a pandas dataframe, save a figure of dataframe column plots.
    '''

    data.hist()
    plt.savefig('all_data_hist.png')

def visualize_by_group_mean(data, cols, group_by_col):

    '''
    Given a dataframe, an array of columns and a column to group by,
    generate a plot of these grouped columns with mean of the group
    '''

    data[cols].groupby(group_by_col).mean().plot()
    file_name = 'viz_by_' + group_by_col + '.png'
    plt.savefig(file_name)

def impute_missing_all(data):
    '''
    Find all columns with missing data and impute with the column's 
    mean.

    To impute specific columns, use impute_missing_column.
    '''

    headers = list(data.columns)
    for name in headers:
        if data[name].isnull().values.any():
            data[name] = data[name].fillna(data[name].mean())

def impute_missing_column(data, columns, method):
    '''
    Given a list of specific data columns, impute missing
    data of those columns with the column's mean, median, or mode.

    This function imputes specific columsn, for imputing all
    columns of the dataset that have missing data, use
    impute_missing_all.
    '''

    for col in columns:
        if method == 'median':
            median = data[col].median()
            data[col] = data[col].fillna(median)
            return median
        elif method == 'mode':
            mode = int(data[col].mode()[0])
            data[col] = data[col].fillna(mode)
            return mode 
        else:
            mean = data[col].mean()
            data[col] = data[col].fillna(mean)
            return mean 


def impute_col_with_val(data, columns, value):
    '''
    Given data, a list of columns, and a value, impute the missing data of 
    given column with the value.

    Good to use to impute test data with training data's mean, median, or mode

    '''
    for col in columns:
        data[col] = data[col].fillna(value)

def log_column(data, column):
    '''
    Log the values of a column.

    Good to use when working with income data

    Returns the name of the new column to include programmatically in list of features
    '''

    log_col = 'log_' + str(column)
    data[log_col] = data[column].apply(lambda x: np.log(x + 1))

    return log_col

def create_bins(data, column, bins, verbose=False):
    '''
    Given a continuous variable, create a new column in the dataframe
    that represents the bin in which the continuous variable falls into.

    If verbose is True, print the value counts of each bin.

    Returns the name of the new column to include programmatically in list of features

    '''
    new_col = 'bins_' + str(column)

    data[new_col] = pd.cut(data[column], bins=bins, include_lowest=True, labels=False)

    if verbose:
        print pd.value_counts(data[new_col])

    return new_col

def convert_to_binary(data, column, zero_string):
    '''
    Given a binary categorical variable, such as a gender column with
    male and female, convert data to 0 for zero_string and 1 otherwise

    Provide the string of the forthcoming 0 value, such as 'male', "MALE", or 'Male"
    '''

    data[column] = data[column].apply(lambda x: 0 if sex == zero_string else 1)

def scale_column(data, column):
    '''
    Given data and a specific column, apply a scale transformation to the column

    Returns the name of the new column to include programmatically in list of features

    '''

    scaled_col = 'scaled_' + str(column)
    data[scaled_col] = StandardScaler().fit_transform(data[column])

    return scaled_col

def model_logistic(training_data, test_data, features, label):

    '''
    With training and testing data and the data's features and label, select the best
    features with recursive feature elimination method, then
    fit a logistic regression model and return predicted values on the test data
    and a list of the best features used.

    '''
    start = time()
    model = LogisticRegression()
    rfe = RFE(model)
    rfe = rfe.fit(training_data[features], training_data[label])
    predicted = rfe.predict(test_data[features])
    best_features = rfe.get_support(indices=True)
    elapsed_time = time() - start
    print 'logistic regression took %s seconds to fit' %elapsed_time
    return predicted, best_features


def evaluate_model(test_data, label, predicted_values):
    '''
    Compare the label of the test data to predicted values
    and return accuracy, precision, recall, and f1 score.

    '''
    accuracy = accuracy_score(test_data[label], predicted_values) 
    precision = precision_score(test_data[label], predicted_values) 
    recall = recall_score(test_data[label], predicted_values) 
    # f1 calculation is F1 = 2 * (precision * recall) / (precision + recall)
    f1 = f1_score(test_data[label], predicted_values) 

    return accuracy, precision, recall, f1

def evaluate_model_threshold(test_data, label, predicted_values, threshold):
    '''
    Compare the label of the test data to predicted values
    and return an accuracy score.
    '''
    accuracy = accuracy_score(test_data[label], predicted_values) 
    precision = precision_score(test_data[label], predicted_values) 
    recall = recall_score(test_data[label], predicted_values) 
    # f1 calculation is F1 = 2 * (precision * recall) / (precision + recall)
    f1 = f1_score(test_data[label], predicted_values) 

    return accuracy, precision, recall, f1

def plot_precision_recall(y_true, y_prob, model_name, model_params):

    '''
    Plot a precision recall curve for one model with its y_prob values.
    '''

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision_curve[:-1]
    recall = recall_curve[:-1]
    plt.clf()
    plt.plot(recall, precision, label='%s' % model_params)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title("Precision Recall Curve for %s" %model_name)
    plt.savefig(model_name)
    plt.legend(loc="lower right")
    #plt.show()

def plot_precision_recall_all_models(y_true, y_prob_dict, file_name):
    '''
    Plot precision-recall curves for models in the y_prob_dict

    y_prob_dict has model names as keys and y_prob values as values

    '''

    plt.clf()

    for model_name, y_prob in y_prob_dict.items():
        print model_name, y_prob
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
        precision = precision_curve[:-1]
        recall = recall_curve[:-1]
        plt.plot(recall, precision, label='%s' %model_name)

    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title("Precision Recall Curves")
    plt.savefig(file_name)
    plt.legend(loc="lower right")
    #plt.show()

def run_imputation(train, test, features):
    '''
    Impute feature columns for train data and use train-set statistics to impute
    test data.
    '''

    # impute TRAIN DATA dependents with mode
    train_dependents_mode = impute_missing_column(train, ['NumberOfDependents'], 'mode')

    # impute TRAIN DATA MonthlyIncome with median
    train_income_median = impute_missing_column(train, ['MonthlyIncome'], 'median')

    assert not train.isnull().values.any()

    # impute test data with train data values
    impute_col_with_val(test, ['NumberOfDependents'], train_dependents_mode)
    impute_col_with_val(test, ['MonthlyIncome'], train_income_median)

    assert not test.isnull().values.any()

def generate_features(train, test, features):
    '''
    Generate features for train and test sets.

    '''

    new_log_col = log_column(train, 'MonthlyIncome')
    age_bins = [0] + range(20, 80, 5) + [120]
    age_bucket = create_bins(train, 'age', age_bins)

    income_bins = range(0, 10000, 1000) + [train['MonthlyIncome'].max()]
    income_bucket = create_bins(train, 'MonthlyIncome', income_bins)

    scaled_income = scale_column(train, 'MonthlyIncome')

    new_log_col = log_column(test, 'MonthlyIncome')
    age_bucket = create_bins(test, 'age', age_bins)
    income_bucket = create_bins(test, 'MonthlyIncome', income_bins)
    scaled_income = scale_column(test, 'MonthlyIncome')

    return new_log_col, age_bucket, income_bucket, scaled_income

def model_loop(models_to_run, df, features, label, n_folds):
    '''
    Given an array of models to run, a train dataset, features, a label, and 
    a number of folds, find the best model and its parameters among all the models and
    the cross product of parameters using k-fold cross validation.

    Returns values for the best model and a dictionary of y-predicted values for each
    model to be used for plotting.

    '''

    best_overall_model = ''
    best_overall_auc = 0
    best_overall_params = ''

    # create params-table csv
    with open('parameters-table.csv', 'wb') as csvfile:
        w = csv.writer(csvfile, delimiter=',')
        w.writerow(['MODEL', 'PARAMETERS', 'avg-fold-AUC', 'AUC-stdv'])

        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            running_model = models_to_run[index]
            parameter_values = grid[running_model]

            top_intra_model_auc = 0

            for p in ParameterGrid(parameter_values):
                auc_per_fold = []

                kf = KFold(len(df), n_folds=n_folds)
                for train_i, test_i in kf: 
                    test = df[:len(test_i)]
                    train = df[:len(train_i)]
                    # run_imputation(train, test, features)

                    # new_log_col, age_bucket, income_bucket, scaled_income = generate_features(train, test, features)
                    # features = features + [new_log_col] + [age_bucket] + [income_bucket] + [scaled_income]

                    clf.set_params(**p)
                    clf.fit(train[features], train[label])

                    if hasattr(clf, 'predict_proba'):
                        y_pred_probs = clf.predict_proba(test[features])[:,1] #second col only for class = 1
                    else:
                        y_pred_probs = clf.decision_function(test[features])

                    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(test[label], y_pred_probs)
                    precision = precision_curve[:-1]
                    recall = recall_curve[:-1]

                    print 'PRECISION', precision 
                    print 'RECALL', recall

                    AUC = auc(recall, precision)
                    auc_per_fold.append(AUC)

                avg_model_auc = np.mean(auc_per_fold)
                auc_stdev = np.std(auc_per_fold)

                # find best parameters within a model and its y_pred to plot
                if avg_model_auc > top_intra_model_auc:
                    top_intra_model_auc = avg_model_auc
                    best_models[running_model] = p

                w.writerow([running_model, clf, avg_model_auc, auc_stdev])

            print 'FINISHED RUNNING MODEL %s' %running_model
            # find best model with params overall
            if avg_model_auc > best_overall_auc:
                best_overall_auc = avg_model_auc
                best_overall_model = running_model
                best_overall_params = clf

    return best_overall_model, best_overall_params, best_overall_auc

def go(training_file, test_file):
    '''
    Run functions for specific data file
    '''
    
    train = read_data(training_file)
    test = read_data(test_file)
    
    #get features, label from create_header():
    features = ["Months since Last Donation",
                "Number of Donations",
                "Total Volume Donated (c.c.)",
                "Months since First Donation"]

    label = "Made Donation in March 2007"

    
    models_to_run=['LR','NB','DT','RF', 'SVM']

    #select folds for k fold validation
    n_folds = 3

    start_loop = time()
    best_overall_model, best_overall_params, best_overall_auc = model_loop(models_to_run, train, features, label, n_folds)
    loop_time_minutes = (time() - start_loop) / 60

    print 'LOOP THRU ALL MODELS TOOK %s MINUTES' % loop_time_minutes
    print 'BEST MODEL %s \n BEST PARAMS %s \n BEST AUC %s \n' % (best_overall_model, best_overall_params, best_overall_auc)
    

    # run_imputation(train, test, features)

    # new_log_col, age_bucket, income_bucket, scaled_income = generate_features(train, test, features)
    # features = features + [new_log_col] + [age_bucket] + [income_bucket] + [scaled_income]

    assert not test.isnull().values.any()

    # create final-table csv
    with open('final-table.csv', 'wb') as csvfile:
        w = csv.writer(csvfile, delimiter=',')
        w.writerow(['MODEL', 'PARAMETERS', 'ACCURACY', 'PRECISION', 'RECALL', 'AUC'])
        params = best_overall_params
        clf = clfs[best_overall_model]  
        clf.set_params(**params)
        clf.fit(train[features], train[label])
        predicted_values = clf.predict(test[features])

        accuracy, precision, recall, f1 = evaluate_model(test, label, predicted_values)

        if hasattr(clf, 'predict_proba'):
            y_pred_probs = clf.predict_proba(test[features])[:,1] #second col only for class = 1
        else:
            y_pred_probs = clf.decision_function(test[features])

        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(test[label], y_pred_probs)
        pr = precision_curve[:-1]
        rec = recall_curve[:-1]
        AUC = auc(rec, pr)

        w.writerow([model, params, accuracy, precision, recall, AUC])


    test['predicted'] = predicted_values
    test.to_csv("/data/pred.csv")

    # plot precision-recall of best model
    #plot_precision_recall(test[label], y_pred_probs, model, params)

if __name__=="__main__":
    instructions = '''Usage: python workflow.py training_file test_file'''

    if(len(sys.argv) != 3):
        print(instructions)
        sys.exit()

    training_file = sys.argv[1]
    test_file = sys.argv[2]

    go(training_file, test_file)

