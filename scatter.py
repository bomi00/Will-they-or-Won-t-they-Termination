import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.special import expit

df = pd.read_csv("/Users/bomiadeleke/Documents/Termination.csv")
#print(df) 

df['EmploymentStatus'].unique()

from sklearn.linear_model import LogisticRegression 

df = df.replace({'EmploymentStatus': {'Voluntarily Terminated': 1,'Terminated for Cause': 1,
                                      'Active': 0}})




y = df['EmploymentStatus']

# print("y is :")
# print(y.sum())


y=y.astype('int')

X = df[['Salary','EngagementSurvey', 'EmpSatisfaction', 'SpecialProjectsCount', 'DaysLateLast30', 'Absences']] 


g = df[['Salary','EngagementSurvey', 'EmpSatisfaction', 'SpecialProjectsCount', 'DaysLateLast30', 'Absences']] 
h = df['EmploymentStatus']


#what train_test_split does is split the dataset randomly. This is where they train and test the data. We use this so we don't have to much training data when implemting the predetion
X_train, X_test, Y_train, Y_test = train_test_split(g,h)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_pred = logreg.predict(X_test)

y_prob = logreg.predict_proba(X_test)
mythreshold = 0.25
y_pred = (y_prob[[1]] >= mythreshold).astype(int)



# X_test = X_test.iloc[::40]
# Y_test = Y_test.iloc[::40]

# X_test = X_test.sort_values(by=['Salary'])

#linear_model.LogisticRegression is logistic regrssion using solvers (a solver is a evaluaters) such as liblinear, newton-cg, sag, saga
#C parameter is basically a balancer when the regression regulate to much 
clf = linear_model.LogisticRegression(C=1e5)
#.fit works as training to model for the modeling process


clf.fit(X_train, Y_train)


#figure creates the figure object 
plt.figure(1, figsize=(4, 3))
#fit workd as a training to a model for the model process
plt.clf()

plt.scatter(X_test['Salary'], Y_test, color='black', zorder=20)
#X_test = np.linspace(-5, 10, 300)
#plt.show()


# #The coef_ contain the coefficients for the prediction of each of the targets. It is also the same as if you trained a model to predict each of the targets separately.
# #clf.intercept includes the intercept of the predition itself and the model 



#a = np.array(X_test[:]) #* clf.coef_
print(X_test.shape)
print(clf.coef_.shape)
print(clf.intercept_.shape)
loss = expit(X_test["Salary"], * clf.coef_ + np.repeat(clf.intercept_,6))
loss = loss.to_numpy()
#loss = loss.ravel()
plt.plot(X_test['Salary'], loss, color='red', linewidth=3)

# plt.tight_layout()
plt.show()