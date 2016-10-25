import pandas as pd
from sklearn import cross_validation, ensemble
from sklearn import metrics
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics  import r2_score


X_test = pd.read_csv('C:\Users\sundar d\Desktop\EE379K Labs\Kaggle - Binary Classficiation/test_final.csv')
X_test = X_test.fillna(X_test.mad())
del X_test['id']

X = pd.read_csv('C:\Users\sundar d\Desktop\EE379K Labs\Kaggle - Binary Classficiation/train_final.csv')
y = X['Y']
del X['id']
del X['Y']

"""Tried deleteing duplicating Features but it resulted in a much lower score(0.80188627336)"""
#del X['F2']
#del X['F3']
#del X['F14']
#del X['F25']
#del X_test['F2']
#del X_test['F3']
#del X_test['F14']
#del X_test['F25']

X = X.fillna(X.mad())

"""LOOKING AT THE R_SQUARED VALUES FOR THE FEATURES (only values > 0.0)"""
c = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10',
     'F11','F12','F13','F14','F15','F16','F17','F18',
     'F19','F20', 'F21','F22','F23','F24','F25','F26','F27',]

for f1 in c:
    for f2 in c:
        r2 = r2_score(X[f1],X[f2])
        if r2 >0.0 and f1!=f2:
            print r2, f1, '&', f2
            print '-------'


"""Cross-validation Split (0.16 test size gave best results)"""
X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
X, y,test_size=.2,random_state=41)

"""5th PLACE ORIGINAL MODEL""" 
#model = xgb.XGBClassifier(max_depth=3, n_estimators=313,gamma=0.5,
#                       learning_rate=0.054, reg_alpha=.47, reg_lambda=0.5,subsample=0.91,
#                       min_child_weight=10.4,colsample_bytree=0.6)
#m = model.fit(X_train,y_train)

"""1st PLACE MODEL WITH RAYMOND's PARAMS"""
print "running xgb..."
model2 = xgb.XGBClassifier(max_depth=4, n_estimators=1000, learning_rate=0.01, min_child_weight=5, gamma=.8, 
                          subsample=.4, reg_alpha=.5, colsample_bytree=.4,reg_lambda=.93)
m2 = model2.fit(X_train,y_train)

#print(model2.feature_importances_)
#plt.bar(range(len(model2.feature_importances_)), model2.feature_importances_)
#xgb.plot_importance(model2)
#plt.show()

"""2nd PLACE ORIGINAL MODEL"""
print "running gradient boost..."
model3 = ensemble.GradientBoostingClassifier(n_estimators=319, max_depth=3, 
         subsample=0.92,learning_rate=0.051, loss='exponential',
          max_features=.4,random_state=303)
m3 = model3.fit(X_train,y_train)


"""ExtraTrees MODEL FOR VOTING CLASSIFIER"""
print "running ExtraTrees..."""
model4 = ensemble.ExtraTreesClassifier(n_estimators=1200, criterion='entropy', min_samples_split=100, 
                                       bootstrap=True, oob_score=True)
m4 = model4.fit(X_train, y_train)                                      
#print(m4.feature_importances_)
                                       
"""ADABoost MODEL FOR VOTING CLASSIFIER"""
print "running AdaBoost..."""
model5 = ensemble.AdaBoostClassifier(n_estimators=335, learning_rate=0.054, random_state=39)
m5 = model5.fit(X_train, y_train)

"""Random forest MODEL FOR VOTING CLASSIFIER"""
print "running RandomForest..."""
#model6 = ensemble.RandomForestClassifier(n_estimators=1900, criterion='entropy', min_samples_split=100, 
#                       bootstrap=True, oob_score=True)
#m6 = model6.fit(X_train, y_train)

"""TRYING VOTING CLASSIFIER- William's idea"""
print "running VotingClassifier..."""
eclf1 = ensemble.VotingClassifier(estimators=[('xgb', model2), ('gbc', model3)],#, ('adc', model4), ('gdc', model5)],
    voting='soft').fit(X_train,y_train)

preds = eclf1.predict_proba(X_cv)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_cv, (preds))
roc_auc = metrics.auc(fpr, tpr)
print "AUC(VOTING):", roc_auc

#preds = m2.predict_proba(X_cv)[:,1]
#fpr, tpr, thresholds = metrics.roc_curve(y_cv, (preds))
#roc_auc = metrics.auc(fpr, tpr)
#print "AUC(XGB):", roc_auc

predictions = eclf1.predict_proba(X_test)[:,1]
d = {'Id':range(49999, 99999), 'Y':predictions}
submit = pd.DataFrame(d)
submit.to_csv('Submission.csv', index=False) 



