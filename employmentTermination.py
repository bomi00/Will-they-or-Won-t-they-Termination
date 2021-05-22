import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X,y) 

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
#print(y_pred)


# print("y_pred is: ")
# print(y_pred)

print()

y_prob = logreg.predict_proba(X_test)
mythreshold = 0.65
y_pred = (y_prob[[1]] >= mythreshold).astype(int)

# print("y_prob is: ")
# for x in (y_pred):
#     print(x[1])


print(y_prob)

y_pred = [None] * len(y_prob)
for index,x in enumerate(y_prob):
    y_pred[index]=(y_prob[index][0] >= mythreshold).astype(int)


print(y_pred)




# print("X-test is :")
# print(X_test.index)
# print(y.iloc[X_test.index])

# print("X_test shape is :")
# print(X_test.shape)

# print("y_pred is :")
# print(y_pred.sum())





#g_train, g_test, h_train, h_test = train_test_split(g,h)

#new_person = [30,34,4,.....]     #change into dataframe

#new_prediction = logreg.predict(new_person)

#y_pred = logreg.predict(X_test)

#score = logreg.score(X_test,y_test)
#print(y_pred)

#y_test = y_test.astype('int')

#y_test = y_test.reshape(311,6)

#plt.scatter(X_test,y_test)
#plt.show()



g = X_test['Salary']
h = X_test['EmpSatisfaction']
i = X_test['SpecialProjectsCount']
j = X_test['DaysLateLast30']
k = X_test['Absences']

# plt.scatter(g, y_pred)
# plt.ylabel('Y_pred')
# plt.xlabel('Salary')
# plt.show()

# plt.scatter(h, y_pred)
# plt.ylabel('Y_pred')
# plt.xlabel('Employee Satisfaction')
# plt.show()

# plt.scatter(i, y_pred)
# plt.ylabel('Y_pred')
# plt.xlabel('SpecialProjectsCount')
# plt.show()

# plt.scatter(j, y_pred)
# plt.ylabel('Y_pred')
# plt.xlabel('DaysLateLast30')
# plt.show()

# plt.scatter(k, y_pred)
# plt.ylabel('Y_pred')
# plt.xlabel('Absences')
# plt.show()



# plt.hist(g, bins = 10)
# plt.xlabel('Salary')
# plt.ylabel('Frequency Employees')
# plt.show()

# plt.hist(h, bins = 10)
# plt.xlabel('Employee Satifaction')
# plt.ylabel('Frequency of Employees')
# plt.show()

# plt.hist(i, bins = 10)
# plt.xlabel('SpecialProjectsCount')
# plt.ylabel('Frequency of Employees')
# plt.show()

# plt.hist(j, bins = 10)
# plt.xlabel('DaysLateLast30')
# plt.ylabel('Frequency of Employees')
# plt.show()

# plt.hist(k, bins = 10)
# plt.xlabel('Absences')
# plt.ylabel('Frequency of Employees')
# plt.show()
print()
print()
