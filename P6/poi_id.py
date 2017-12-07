#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn.grid_search import GridSearchCV
from time import time
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn.cross_validation import KFold, train_test_split, StratifiedShuffleSplit
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

print "TASK 1: Select Featuers"
#Number of Employees in the Dataset
print 'Number of people in the Enron dataset: {0}'.format(len(data_dict))


#Number of POIs in the Dataset
pois = [x for x, y in data_dict.items() if y['poi']]
print 'Number of POI\'s: {0}'.format(len(pois))
data_dict.items()[0]



##Feature Example for Skilling
#print str(data_dict["SKILLING JEFFREY K"])

##Features in the Enron Dataset
print 'Number of features for each person in the Enron dataset: {0}'.format(len(data_dict.values()[0]))
print '      '


#Features
features_list = ['poi','salary'] # You will need to use more features

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 
                  'from_messages',
                  'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']
financial_features = ['salary', 'deferral_payments', 'total_payments', 
                      'loan_advances', 'bonus', 'restricted_stock_deferred',
                      'deferred_income', 'total_stock_value', 'expenses', 
                      'exercised_stock_options',
                      'other', 'long_term_incentive', 'restricted_stock',
                      'director_fees']


###Missing Values in features
print "Missing Values in each Feature"
def nan_values(data_dict):
    counts = dict.fromkeys(data_dict.itervalues().next().keys(), 0)
    for i in data_dict:
        employee = data_dict[i]
        for j in employee:
            if employee[j] == 'NaN':
                counts[j] += 1
    return counts

valid_values = nan_values(data_dict)
print valid_values
print '     '

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#
#
########################## Task 2: Remove outliers##############################
print "TASK 2: Remove Outliers"    
#Identifying Outliers    

### read in data dictionary, convert to numpy array
#print 'TOTAL'
#data = featureFormat(data_dict, financial_features)

#for point in data:
#    salary = point[0]
#    bonus = point[1]
#    plt.scatter( salary, bonus )

#plt.xlabel("salary")
#plt.ylabel("bonus")
#plt.show()

#Eugene Lockhart Outlier
#print 'EUGENE E LOCKHART'
#print str(data_dict["LOCKHART EUGENE E"])
print '     '

#Travel Agency in the Park Outlier
#print 'Travel Agency In The Park'
#print str(data_dict['THE TRAVEL AGENCY IN THE PARK'])


#removing all 3 outliers
outliers = ['LOCKHART EUGENE E','TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
for outlier in outliers:
    data_dict.pop(outlier, 0)  
print '  '
    
#Update of employee count after removal of OUTLIERS
print "Number of Enron employees after removing outliers:", len(data_dict.keys())
print '        '
#
#
#    
########### Task 3: Create new feature(s)################
print 'TASK 3: Create new feature(s)'

'''
Defined a function to calculate the percentages and insert into data_dict
'''

#Function to compute ratio of two initial features:
def ratio(numerator, denominator):
    if (numerator == 'NaN') or (denominator == 'NaN') or (denominator == 0):
        fraction = 0
    else:
        fraction = float(numerator)/float(denominator)
    return fraction


#Create 2 New Features

'''
Function for the created feature fraction_to_poi
'''

def from_this_person_from_poi_ratio(dict):
    for key in dict:
       from_this_person_from_poi = dict[key]['from_this_person_to_poi']
       from_messages= dict[key]['from_messages']
       fraction_to_poi = ratio(from_this_person_from_poi, from_messages)
       dict[key]['fraction_to_poi'] = fraction_to_poi
       
'''
Function for the created feature fraction_from_poi
'''
def from_poi_to_this_person_ratio(dict):
    for key in dict:
        from_poi_to_this_person_percentage = dict[key]['from_poi_to_this_person']
        to_messages = dict[key]['to_messages']
        fraction_from_poi= ratio(from_poi_to_this_person_percentage, to_messages)
        dict[key]['fraction_from_poi'] = fraction_from_poi

        
#Inserting new features into dataset
 
from_poi_to_this_person_ratio(data_dict)
from_this_person_from_poi_ratio(data_dict)


#Check Dictionary
#for employee in data_dict:
#    for feature, value in data_dict[employee].items():
#        print feature
#    break

### Store to my_dataset for easy export below.
my_dataset = data_dict

#Updated Features List
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 
                 'loan_advances', 'bonus', 'restricted_stock_deferred','deferred_income',
                 'total_stock_value', 'expenses', 'from_poi_to_this_person', 
                 'exercised_stock_options', 'other', 'long_term_incentive', 
                 'shared_receipt_with_poi', 'restricted_stock',
                 'director_fees', 'to_messages', 'from_messages',
                 'from_this_person_to_poi', 'fraction_to_poi', 
                 'fraction_from_poi'] 
#
#
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Scale features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

#Selecting the  best features by using SelectKBest
def get_k_best(data_dict, features_list, k):
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} best features: {1}\n".format(k, k_best_features.keys())
    return k_best_features

print get_k_best(data_dict, features_list, 6)
print '      '


skb = SelectKBest()
gnb = GaussianNB()



### Task 4: Try a varity of classifiers#################################
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

print 'TASK 4: Try a Variety of Classifiers'
print '   '

#Updated Features List (Top 5 Features)
features_list =["poi",'bonus','exercised_stock_options',
                 'salary','fraction_to_poi', 'total_stock_value',
                 'deferred_income']
                
   
data = featureFormat(data_dict, features_list)
#split into labels and features 
labels, features = targetFeatureSplit(data)

#split data into training and testing datasets
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)


#use Decison Tree Classifier algorithm to predict accuracy and time
t0 = time()
clf_dt = DecisionTreeClassifier()
clf_dt.fit(features_train,labels_train)
score = clf_dt.score(features_test,labels_test)
pred= clf_dt.predict(features_test)
print "DecisionTreeClassifier accuracy is", score
print "Decision tree algorithm time:", round(time()-t0, 3), "s"
print '   '

'''
Testing Clssifier Accuracy, Precision and recall for Decision Tree Classifier
'''
from tester import test_classifier
test_classifier(clf_dt, my_dataset, features_list)


# use GaussianNB algorithm to predict accuracy and time
t0 = time()
clf_g = GaussianNB()
clf_g.fit(features_train, labels_train)
pred = clf_g.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
print "GaussianNB accuracy is", accuracy
print "GaussianNB algorithm time:", round(time()-t0, 3), "s"
print '   '


'''
Testing Clssifier Accuracy, Precision and recall for GaussianNB
'''
from tester import test_classifier
test_classifier(clf_g, my_dataset, features_list)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
 
    
#tune decision tree algorithm 
t0 = time()

clf_dt = DecisionTreeClassifier(criterion='gini', class_weight='balanced',
            min_samples_split=3, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
clf_dt = clf_dt.fit(features_train,labels_train)
pred = clf_dt.predict(features_test)
print("done in %0.3fs" % (time() - t0))
print '    '
#

from tester import test_classifier
test_classifier(clf_dt, my_dataset, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf = clf_g
dump_classifier_and_data(clf, my_dataset, features_list)