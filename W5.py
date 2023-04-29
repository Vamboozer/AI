import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone 
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt 
%matplotlib inline 

class ThreesandEights:
    """
    Class to store MNIST 3s and 8s data
    """

    def __init__(self, location):

        #import pickle, gzip
        import pickle, requests

        # Load the dataset
        #f = gzip.open(location, 'rb')
        response = requests.get(location)
        dataLocal = pickle.loads(response.content)

        # Split the data set 
        #x_train, y_train, x_test, y_test = pickle.load(f)
        x_train, y_train, x_test, y_test = dataLocal
                
        # Extract only 3's and 8's for training set 
        self.x_train = x_train[np.logical_or(y_train== 3, y_train == 8), :]
        self.y_train = y_train[np.logical_or(y_train== 3, y_train == 8)]
        self.y_train = np.array([1 if y == 8 else -1 for y in self.y_train])
        
        # Shuffle the training data 
        shuff = np.arange(self.x_train.shape[0])
        np.random.shuffle(shuff)
        self.x_train = self.x_train[shuff,:]
        self.y_train = self.y_train[shuff]

        # Extract only 3's and 8's for validation set 
        self.x_test = x_test[np.logical_or(y_test== 3, y_test == 8), :]
        self.y_test = y_test[np.logical_or(y_test== 3, y_test == 8)]
        self.y_test = np.array([1 if y == 8 else -1 for y in self.y_test])
        
        f.close()

def view_digit(ex, label=None, feature=None):
    """
    function to plot digit examples 
    """
    if label: print("true label: {:d}".format(label))
    img = ex.reshape(21,21)
    col = np.dstack((img, img, img))
    if feature is not None: col[feature[0]//21, feature[0]%21, :] = [1, 0, 0]
    plt.imshow(col)
    plt.xticks([]), plt.yticks([])
    
#data = ThreesandEights("data/mnist21x21_3789.pklz")
url = "https://raw.githubusercontent.com/Vamboozer/AI/main/mnist21x21_3789.pkl"
data = ThreesandEights(url)

view_digit(data.x_train[0], data.y_train[0])

# ========== Building an Adaboost Classifier to classify MNIST digits 3 and 8 ============

