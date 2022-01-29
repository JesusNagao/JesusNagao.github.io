from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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

X = np.asmatrix(np.column_stack((gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose, bmi, smoking_status)))

x_pred = X[5,:]

neigh = KNeighborsClassifier(n_neighbors=3, weights='distance', p=1.5).fit(X,y)
print(mean_squared_error(y,neigh.predict(X)))


