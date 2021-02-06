#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', \
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', \
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', \
                 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Extract features and labels from dataset for local testing
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Plot various features vs labels or other features
fig1 = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ='3d')
for i in range(len(labels)):
    if labels[i] == 1.0:
        ax.scatter3D(features[i][0], features[i][2], features[i][4], c = 'r', marker = 'x')
    else:
        ax.scatter3D(features[i][0], features[i][2], features[i][4], c = 'b', marker = 'o')

ax.set_xlabel(features_list[1])
ax.set_ylabel(features_list[3])
ax.set_zlabel(features_list[5])


print "Number of POI in data set: ", labels.count(1.0)


### Task 2: Remove outliers
from sklearn.preprocessing import StandardScaler

# Scale data as (x-mean)/std. dev, aka z-score
scaler = StandardScaler()
scaler.fit(features)
scaled_features = scaler.transform(features)

# Plot salary vs bonus
fig2 = plt.figure(figsize = (10, 7))
plt.scatter(scaled_features[:, 0], scaled_features[:, 4])
plt.xlabel('Scaled Salary')
plt.ylabel('Scaled Bonus')

# Remove data points with scaled "salary" more than 3 or less than -3
outlier_indices = []
for data_num in range(len(scaled_features)):
    if scaled_features[data_num][0] > 3 or scaled_features[data_num][0] < -3:
        outlier_indices.append(data_num)

scaled_features = np.delete(scaled_features, outlier_indices, 0)
labels = np.delete(labels, outlier_indices, 0)


# Plot salary vs bonus after removing outliers
fig3 = plt.figure(figsize = (10, 7))
plt.scatter(scaled_features[:, 0], scaled_features[:, 4])
plt.xlabel('Scaled Cleaned Salary')
plt.ylabel('Scaled Cleaned Bonus')


### Plot various features vs labels or other features after scaling and removing outliers
fig4 = plt.figure(figsize = (10, 7))
ax4 = plt.axes(projection ='3d')
for i in range(len(labels)):
    if labels[i] == 1.0:
        ax4.scatter3D(scaled_features[i][0], scaled_features[i][2], scaled_features[i][4], c = 'r', marker = 'x')
    else:
        ax4.scatter3D(scaled_features[i][0], scaled_features[i][2], scaled_features[i][4], c = 'b', marker = 'o')

ax4.set_xlabel('Scaled ' + features_list[1])
ax4.set_ylabel('Scaled ' + features_list[3])
ax4.set_zlabel('Scaled ' + features_list[5])
#plt.show()


###  Split data into training and testing data
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(scaled_features, labels, test_size=0.3, random_state=42)

### Feature selection/transformation
from sklearn.feature_selection import f_classif, SelectPercentile
selector = SelectPercentile(f_classif, percentile=85)
selector.fit(features_train, labels_train)
features_train_transformed = selector.transform(features_train)
features_test_transformed = selector.transform(features_test)

print "Feature Scores: ", selector.scores_

'''
### Make new features using PCA
from sklearn.decomposition import PCA as RandomizedPCA
num_of_features = np.shape(features_train)[1]
print "Num of features: ", num_of_features

n_components = num_of_features - 12 #it was found that last 2 features have negligible explained variance ratio
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(features_train)
features_train_transformed = pca.transform(features_train)
features_test_transformed = pca.transform(features_test)

print "Explained variances: ", pca.explained_variance_ratio_
'''

### Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#clf = GaussianNB()
#clf = SVC(kernel='rbf',  class_weight='balanced', C = 5e3, gamma = 'auto')
#clf = DecisionTreeClassifier()
#clf = RandomForestClassifier(n_estimators = 100)
clf = AdaBoostClassifier()
clf.fit(features_train_transformed, labels_train)
print "Score: ", clf.score(features_test_transformed, labels_test)


'''
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
'''
my_dataset = data_dict

dump_classifier_and_data(clf, my_dataset, features_list)

## What worked
# 1. Scaling features for unit variance
# 2. Remove outliers
# 3. Select 85% top features
# 4. Use AdaBoostClassifier
