print("\n" * 80)

#### Presentación ###
print("##############################")
print("#      Avance 2              #")
print("#                            #")
print("#      Alumnos:              #")
print("# Jesus Nagao      A01197330 #")
print("# Marco Briceño    A00824768 #")
print("# Luis Resendez    A01197344 #")
print("##############################")
#####################
print()
print()
#####################

####### TITULO ######
print("STROKES")
print()
# Parte 1
print("Mediante regresión lineal")
print()
#####################

###########################
import pandas as pd
import numpy as np
import os
import graphviz

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA


from graphviz import Source
from sklearn import tree
from sklearn.tree import export_graphviz

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score

from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_squared_error,r2_score
###########################

#####################
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
#####################


###### FIGURAS ######
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
#####################

### ACOMODO DATOS ###
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
#####################

#### TRAINING #######
X = np.asmatrix(np.column_stack((gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose, bmi, smoking_status)))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
#####################


#### Linear Reg #####
Lr_reg = LinearRegression().fit(X_train, y_train)
y_pred = Lr_reg.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)
CD = r2_score(y_test, y_pred)
print("Mean Sqare Error: %.2f" % MSE)
print("Coeficiente de Determinación: %.2f" % CD)
print("")
#####################


# Parte 2
print("(Presione enter para continuar)")
input()
print("Mediante arboles de decisión")
print()
#####################

#####################
tree_clf = tree.DecisionTreeClassifier()#max_depth=2)
tree_clf.fit(X, y)
#####################

####### MODELO ######
export_graphviz(
    tree_clf,
    out_file = os.path.join(IMAGES_PATH, "heart_tree.dot"),
    feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'residence_type', 'avg_glucose', 'bmi', 'smoking_status']
    #class_names = ['Si';'No'],
    #rounded = True,
    #filled = True
    )
Source.from_file(os.path.join(IMAGES_PATH, "heart_tree.dot"))
#####################

###### GRAFICA ######
with open(os.path.join(IMAGES_PATH, "heart_tree.dot")) as f:
    dot_graph = f.read()
dot = graphviz.Source(dot_graph)
dot.format = 'png'
dot.render(filename='heart_tree', cleanup=True)
dot
#####################

####### TRAIN #######
tree_train = tree.DecisionTreeClassifier()
tree_train = tree_train.fit(X_train, y_train)
y_pred = tree_train.predict(X_test)
print("Precisión: ",metrics.accuracy_score(y_test, y_pred))
print("Matriz de Confusión = ")
print(confusion_matrix(y_test, y_pred))
#####################


# Parte 3
print("(Presione enter para continuar)")
input()
print("Mediante regresión logísitca")
print()
#####################

#####################
Lrr = LogisticRegression(multi_class= 'ovr', max_iter=1000).fit(X_train, y_train)
Logistic_pred = Lrr.predict(X_test)

print("Precisión: {:.4f}".format(Lrr.score(X_test, y_test)))
print("Matriz de Confusión = ")
print(confusion_matrix(y_test, Logistic_pred))
#####################


# Parte 4
print("(Presione enter para continuar)")
input()
print("Mediante k-NN")
print()
#####################

#####################
scores = []
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors = k, weights='distance', p=1.5)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
    
knnn =KNeighborsClassifier(n_neighbors = scores.index(max(scores))+1).fit(X_train,y_train)
knn_pred = knnn.predict(X_test)
print("Precisión para k = {}: {:.4f}".format(scores.index(max(scores))+1, max(scores)))
print("Matriz de Confusión = ")
print(confusion_matrix(y_test, knn_pred))
#####################


# Parte 5
print("(Presione enter para continuar)")
input()
print("Mediante Naive bayes")
print()
#####################

#####################
Nf = GaussianNB().fit(X_train, y_train)
Naive_pred = Nf.predict(X_test)
print("Precisión: {:.4f}".format(Nf.score(X_test, y_test)))
print("Matriz de Confusión = ")
print(confusion_matrix(y_test, Naive_pred))
#####################