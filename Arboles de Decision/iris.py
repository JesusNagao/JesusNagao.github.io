import numpy as np
import os
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from graphviz import Source
from sklearn.tree import export_graphviz
from sklearn.metrics import mean_squared_error

   
mpl.rc('axes', labelsize=14) 
mpl.rc('xtick', labelsize=12) 
mpl.rc('ytick', labelsize=12)
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID) 
os.makedirs(IMAGES_PATH, exist_ok=True)


#def main():
    
iris = load_iris()
tree_clf = DecisionTreeClassifier(criterion= "entropy")


X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
tree_clf.fit(X_train,y_train)


def save_fig(fig_id='dtree', tight_layout=True, fig_extension="png", resolution=300):
    
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    

tree_clf_2 = tree_clf.predict(X_test)

export_graphviz(tree_clf, out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"))

Source.from_file(os.path.join(IMAGES_PATH, "iris_tree.dot"))

MSE = mean_squared_error(tree_clf_2, y_test)

print('MSE = ', MSE)