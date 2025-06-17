import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import joblib
from imblearn.over_sampling import SMOTE

d1=pd.read_csv(r"loan_data.csv")
d2=pd.DataFrame(d1)
#-------------------------------------------pre processing--------------------------------------------------------
d=d2.head(30)
d.isnull().sum()                               #for count of null values in each row
d.isnull().sum()/d.shape[0]*100                         #for calculating percentage of null values in each row
d=d.drop(["person_gender","person_emp_exp","person_home_ownership","person_education","cb_person_cred_hist_length"],axis=1)
d=pd.DataFrame(d)
app=sns.countplot(x='loan_status',data=d)
for i in app.containers:
    app.bar_label(i)
sns.displot(d['person_age'])     #displot used for single column representation
plt.grid(axis='y') 
sns.displot(d['loan_intent'])     #displot used for single column representation
plt.grid(axis='y')
sns.displot(d['previous_loan_defaults_on_file'])     #displot used for single column representation
plt.grid(axis='y')
sns.jointplot(x='person_age',y='person_income',data=d)       #for multiple column representation
plt.grid(axis='both') 

X=d.iloc[:,:8]
Y=d.iloc[:,8] 
X=pd.get_dummies(X,prefix=['loan_intent'],columns=['loan_intent'],dtype=int,sparse=False)
le=LabelEncoder()
X["previous_loan_defaults_on_file"]=le.fit_transform(X["previous_loan_defaults_on_file"])
joblib.dump(le,'label_encoder.joblib')
sc=StandardScaler()
X[["person_income","loan_amnt","credit_score"]]=sc.fit_transform(X[["person_income","loan_amnt","credit_score"]])
joblib.dump(sc,'scaler.joblib')
sm=SMOTE(random_state=1)
resampled_X,resampled_Y=sm.fit_resample(X,Y)
X_train,X_test,Y_train,Y_test=train_test_split(resampled_X,resampled_Y,test_size=0.2,random_state=1)
c_knn= KNeighborsClassifier(n_neighbors=3)
c_knn.fit(X_train,Y_train)
joblib.dump(c_knn,'loan_prediction.joblib')
Y_pred=c_knn.predict(X_test)
accu=metrics.accuracy_score(Y_test,Y_pred)
print(f"accuracy={accu}")
l1=(input("enter the person_age:")).split(",")
l11=[int(i) for i in l1]
l2=(input("enter the person_income:")).split(",")
l22=[int(i) for i in l2]
l3=(input("enter the loan_amnt:")).split(",")
l33=[int(i) for i in l3]
l4=(input("enter the loan_intent:")).upper().split(",")
l44=[(i) for i in l4]
l5=(input("enter the loan_int_rate:")).split(",")
l55=[float(i) for i in l5]
l6=(input("enter the loan_percent_income:")).split(",")
l66=[float(i) for i in l6]
l7=(input("enter the credit_score:")).split(",")
l77=[int(i) for i in l7]
l8=(input("enter the previous_loan_defaults_on_file:")).split(",")
l88=[int(i) for i in l8]
sample_data = {'person_age':l11,'person_income':l22,'loan_amnt': l33,'loan_intent': l44,'loan_int_rate': l55,
               'loan_percent_income': l66,'credit_score': l77,'previous_loan_defaults_on_file': l88}
sample_df = pd.DataFrame(sample_data)
sample_df = pd.get_dummies(sample_df, columns=['loan_intent'], dtype=int, sparse=False)
expected_columns = ['person_age', 'person_income', 'loan_amnt','loan_intent_DEBTCONSOLIDATION','loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE', 'loan_int_rate', 'loan_percent_income', 'credit_score', 'previous_loan_defaults_on_file']
for col in expected_columns:
    if col not in sample_df:
        sample_df[col] = 0
sample_df=sample_df[expected_columns]        
sc = joblib.load('scaler.joblib') 
sample_df[['person_income', 'loan_amnt', 'credit_score']] = sc.transform(sample_df[['person_income', 'loan_amnt', 'credit_score']])
le.fit([0,1])
sample_df['previous_loan_defaults_on_file'] = le.transform(sample_df['previous_loan_defaults_on_file'])
sample=np.array(sample_df)
model = joblib.load('loan_prediction.joblib')
preds=model.predict(sample)
predics=[d.loan_status[p] for p in preds] 
print(f"predictions={predics}")
for i in predics:
    if i == 1:
        print("____________Loan Approved_______________")
    else:
        print("____________Loan Denied_______________")




       

   




       