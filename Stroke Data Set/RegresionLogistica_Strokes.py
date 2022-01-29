import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

x = pd.read_csv("datoslimpios.csv")

y = x['stroke'] 
gender = x['gender'].to_numpy()
age = x['age'].to_numpy()
hypertension = x['hypertension'].to_numpy()
heart_disease = x['heart_disease'].to_numpy()
ever_married = x['ever_married'].to_numpy()
work_type = x['work_type'].to_numpy()
residence_type = x['Residence_type'].to_numpy()
avg_glucose = x['avg_glucose_level'].to_numpy()
bmi = x['bmi'].to_numpy()
smoking_status = x['smoking_status'].to_numpy()


for i in range(gender.shape[0]):
    if(gender[i] == 'Male'):
        gender[i] = 0
    elif(gender[i] == 'Female'):
        gender[i] = 1
    elif(gender[i] == 'Other'):
        gender[i] = 2

for i in range(ever_married.shape[0]):
    if(ever_married[i] == 'Yes'):
        ever_married[i] = 0
    elif(ever_married[i] == 'No'):
        ever_married[i] = 1

for i in range(work_type.shape[0]):
    if(work_type[i] == 'Private'):
        work_type[i] = 0
    elif(work_type[i] == 'Self-employed'):
        work_type[i] = 1
    elif(work_type[i] == 'Govt_job'):
        work_type[i] = 2
    elif(work_type[i] == 'children'):
        work_type[i] = 3
    elif(work_type[i] == 'Never_worked'):
        work_type[i] = 4

for i in range(residence_type.shape[0]):
    if(residence_type[i] == 'Urban'):
        residence_type[i] = 0
    elif(residence_type[i] == 'Rural'):
        residence_type[i] = 1

for i in range(smoking_status.shape[0]):
    if(smoking_status[i] == 'formerly smoked'):
        smoking_status[i] = 0
    elif(smoking_status[i] == 'smokes'):
        smoking_status[i] = 1
    elif(smoking_status[i] == 'never smoked'):
        smoking_status[i] = 2
    elif(smoking_status[i] == 'Unknown'):
        smoking_status[i] = 3


gender_train = gender[:round(gender.shape[0]*0.8)]
age_train = age[:round(age.shape[0]*0.8)]
hypertension_train = hypertension[:round(hypertension.shape[0]*0.8)]
ever_married_train = ever_married[:round(ever_married.shape[0]*0.8)]
work_type_train = work_type[:round(work_type.shape[0]*0.8)]
residence_type_train = residence_type[:round(gender.shape[0]*0.8)]
avg_glucose_train = avg_glucose[:round(avg_glucose.shape[0]*0.8)]
bmi_train = bmi[:round(bmi.shape[0]*0.8)]
smoking_status_train = gender[:round(gender.shape[0]*0.8)]
y = y[:round(y.shape[0]*0.8)]

gender_pred = gender[:round(gender.shape[0]*0.2)]
age_pred = age[:round(age.shape[0]*0.2)]
hypertension_pred = hypertension[:round(hypertension.shape[0]*0.2)]
heart_disease_pred = heart_disease[:round(heart_disease.shape[0]*0.2)]
ever_married_pred = ever_married[:round(ever_married.shape[0]*0.2)]
work_type_pred = work_type[:round(work_type.shape[0]*0.2)]
residence_type_pred = residence_type[:round(gender.shape[0]*0.2)]
avg_glucose_pred = avg_glucose[:round(avg_glucose.shape[0]*0.2)]
bmi_pred = bmi[:round(bmi.shape[0]*0.2)]
smoking_status_pred = gender[:round(gender.shape[0]*0.2)]
y_pred = y[:round(y.shape[0]*0.2)]


X = np.asmatrix(np.column_stack((gender_train, age_train, hypertension_train, ever_married_train, work_type_train, residence_type_train, avg_glucose_train, bmi_train, smoking_status_train)))
X_pred = np.asmatrix(np.column_stack((gender_pred, age_pred, hypertension_pred, ever_married_pred, work_type_pred, residence_type_pred, avg_glucose_pred, bmi_pred, smoking_status_pred)))

#X = np.asmatrix(np.column_stack((gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose, bmi, smoking_status)))


logireg = LogisticRegression(random_state=0).fit(X,y)
pred = logireg.predict(X_pred)

print(mean_squared_error(pred, y_pred))