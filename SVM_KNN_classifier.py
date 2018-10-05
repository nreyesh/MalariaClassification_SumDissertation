
import numpy as np
import cv2
from matplotlib import pyplot as plt
from Load_data import load_data_2
from Feature_extractor import ObtainFeatures
from sklearn.ensemble import BaggingClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report,f1_score
from sklearn.model_selection import train_test_split, GridSearchCV

## Characteristics of Images
img_rows = 100  # dimensions of image
img_cols = 100
channel = 3  # RGB
nb_train_samples =  27558 #7278  #1280 #27558


train_data_dir = '/home/nicor/Documents/Summer_Project/cell_images'               #   7278 images
#train_data_dir = '/home/nicor/Documents/Summer_Project/cell_images_4B/Train'               #   7278 images

# Loads DataSet
X, Y,lab = load_data_2(train_data_dir,nb_train_samples,img_rows, img_cols)

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
#X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

#X_test, Y_test = load_data_2(test_data_dir,nb_test_samples,img_rows, img_cols)


# Prints the shape of the data
#print('Train set: ',X_train.shape, Y_train.shape,sum(Y_train))
#print('Val set: ',X_val.shape, Y_val.shape)
#print('Test set: ',X_test.shape, Y_test.shape,sum(Y_test))


X = ObtainFeatures(X,len(X))
#X_train = ObtainFeatures(X_train,len(X_train))
#X_test = ObtainFeatures(X_test,len(X_test))

#Train = Features(nb_train_samples,X_train)
#Test = Features(nb_test_samples,X_test)

'''
features = []
for i in range(0,nb_train_samples-1):
    vector = []
    vector.extend(GrayScale(X_train[i]))
    vector.extend(ColourHist(X_train[i],64))
    vector.extend(ColourHist(X_train[i],256))
    vector.extend(ColourHist(X_train[i],576))
    vector.extend(SaturationHist(X_train[i]))

    features.append(vector)
    print('Shape: ', len(vector),len(vector[0]))

features = np.array(features)
'''

#print('Feature shape Train set: ',X_train.shape)
#print('Feature shape Test set: ',X_test.shape,'\n')

'''
#np.savetxt("/home/nicor/Documents/Summer_Project/Train_set.txt", X_train, delimiter=",")
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

## Decision Tree

tree = DecisionTreeClassifier(min_samples_split=0.01,random_state=42)
tree.fit(X_train,Y)
y_pred = tree.predict(X_test)

acc = accuracy_score(Y_test, y_pred)
conf = confusion_matrix(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred, average=None)

print('-------- Prediction Decision Tree --------','\n','Accuracy: ',acc)
print('F1_score: ',f1)
print('Confussion Matrix: \n',conf)

## Bagging

clf = BaggingClassifier(DecisionTreeClassifier(random_state=42,min_samples_split=0.01))
clf.fit(X_train,Y)
y_pred = clf.predict(X_test)

acc = accuracy_score(Y_test, y_pred)
conf = confusion_matrix(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred, average=None)

print('-------- Prediction Bagging --------','\n','Accuracy: ',acc)
print('F1_score: ',f1)
print('Confussion Matrix: \n',conf)

## Training RandomForest

clf = RandomForestClassifier(random_state=1,min_samples_split=0.01)
clf.fit(X_train, Y)
y_pred = clf.predict(X_test)
#y_pred = np.round(y_pred)

acc = accuracy_score(Y_test, y_pred)
conf = confusion_matrix(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred, average=None)

print('-------- Prediction RandomForest --------','\n','Accuracy: ',acc)
print('F1_score: ',f1)
print('Confussion Matrix: \n',conf)

## Training GradientBoosting

clf = GradientBoostingClassifier(learning_rate=0.01,random_state=1)
clf.fit(X_train, Y)
y_pred = clf.predict(X_test)

acc = accuracy_score(Y_test, y_pred)
conf = confusion_matrix(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred, average=None)

print('-------- Prediction GradientBoosting --------','\n','Accuracy: ',acc)
print('F1_score: ',f1)
print('Confussion Matrix: \n',conf)
'''


### Training SVM ###

'''
## Tunning process
svc = svm.SVC(random_state=42,kernel='poly')
# Tuning Parameters
parameters = [{#'kernel': ['rbf'],
                 #   'gamma': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                 #   'C': [0.1, 0.5, 1, 5, 10, 50, 75, 100, 500]},
                #{'kernel': ['poly'],
                    'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                    'C': [0.1, 0.5, 1, 5, 10, 50, 75, 100, 500],
                    'degree': [3,4,5]}
                #{'kernel': ['linear']
                #    , 'C': [0.1, 0.5, 1, 5, 10, 50, 75, 100, 500]}
              ]

#scores = {'f1'}


# Results per parameter

print("# Tuning hyper-parameters for %s" % 'f1')
print()
clf = GridSearchCV(svc, parameters, cv=5, scoring='f1',n_jobs=-1)
clf.fit(X, Y)
#clf.score(X_train, Y_train)
#clf.score(X_test, Y_test)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
'''


## Cross Validation

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

model = svm.SVC(kernel='linear', C=75, random_state=41)

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=41)
#print(skf)
for train_index, test_index in skf.split(X, Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]


    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))

    print(cnf_matrix)





'''
from sklearn.model_selection import cross_validate

scoring = ['f1', 'accuracy','make_scorer','recall_score']
clf = svm.SVC(kernel='linear', C=75, random_state=41)
scores = cross_validate(clf, X, Y, scoring=scoring, cv=5, return_train_score=False)
print('F1_score: ',scores['test_f1'])
print('Accuracy: ',scores['test_accuracy'])
print('Accuracy: ',scores['test_make_scorer'])
print('Accuracy: ',scores['test_recall_score'])
'''

'''
## Final Training
from sklearn.externals import joblib

model = svm.SVC(C=75,kernel="linear")# kernel="linear")
model.fit(X,Y)
#model.fit(X_train, Y_train)
print("Model Trained: OK")

# save the model to disk
filename = 'SVM_linear_final.sav'
joblib.dump(model, filename)
'''

'''
y_pred = model.predict(X_test)
print("Predictions: OK",'\n')

acc = accuracy_score(Y_test, y_pred)
print('-------- Prediction SVM --------','\n','Accuracy: ',acc)

conf = confusion_matrix(Y_test, y_pred)
print('Confussion Matrix: \n',conf)

f1 = f1_score(Y_test, y_pred, average=None)
print('F1_score: ',f1)
'''