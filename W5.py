import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone 
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt 
%matplotlib inline 
import pytest

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

# ========== Problem 1: Building an Adaboost Classifier to classify MNIST digits 3 and 8 ============

# ====== Part A ======

class AdaBoost:
    def __init__(self, n_learners=20, base=DecisionTreeClassifier(max_depth=3), random_state=42):
        """
        Create a new adaboost classifier.
        
        Args:
            n_learners (int, optional): Number of weak learners in classifier.
            base (BaseEstimator, optional): Your general weak learner 
            random_state (int, optional): set random generator.  needed for unit testing. 

        Attributes:
            base (estimator): Your general weak learner 
            n_learners (int): Number of weak learners in classifier.
            alpha (ndarray): Coefficients on weak learners. 
            learners (list): List of weak learner instances. 
        """
        
        np.random.seed(random_state)
        
        self.n_learners = n_learners 
        self.base = base
        self.alpha = np.zeros(self.n_learners)
        self.learners = []
        
    def fit(self, X_train, y_train):
        """
        Train AdaBoost classifier on data. Sets alpha and learners. 
        
        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data 
        """

        # =================================================================
        # TODO 

        # Note: You can create and train a new instantiation 
        # of your sklearn decision tree as follows 
        # you don't have to use sklearn's fit function, 
        # but it is probably the easiest way 

        # w = np.ones(len(y_train))
        # w /= np.sum(w) 
        # for loop
        #   h = clone(self.base)
        #   h.fit(X_train, y_train, sample_weight=w)
        #   ...
        #
        #
        #   ...
        #   Save alpha and learner
        
        # =================================================================
        
        # your code here
        n = X_train.shape[0]
        w = np.full(n, 1/n)
        for i in range(self.n_learners):
            learner = DecisionTreeClassifier(max_depth=1)
            learner.fit(X_train, y_train, sample_weight=w)
            predictions = learner.predict(X_train)
            error = np.sum(w*(predictions!=y_train))/np.sum(w)
            alpha = np.log((1-error)/error)
            self.learners.append(learner)
            self.alpha.append(alpha)
            w = w*np.exp(alpha*(predictions!=y_train))
            w /= np.sum(w)
        
        return self  
            
    def error_rate(self, y_true, y_pred, weights):
        # calculate the weighted error rate
        error = 0
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                error += weights[i]
        return error / np.sum(weights)
        
    def predict(self, X):
        """
        Adaboost prediction for new data X.
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            
        Returns: 
            yhat (ndarray): [n_samples] ndarray of predicted labels {-1,1}
        """

        # =================================================================
        # TODO
        # =================================================================
        yhat = np.zeros(X.shape[0])
        
        # your code here
        for i in range(self.num_learners):
            yhat += self.alpha[i] * self.learners[i].predict(X)
        return np.sign(yhat)
    
    def score(self, X, y):
        """
        Computes prediction accuracy of classifier.  
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            y (ndarray): [n_samples] ndarray of true labels  
            
        Returns: 
            Prediction accuracy (between 0.0 and 1.0).
        """
        
        # your code here
        predictions = self.predict(X)
        return np.sum(predictions == y) / X.shape[0]
    
    def staged_score(self, X, y):
        """
        Computes the ensemble score after each iteration of boosting 
        for monitoring purposes, such as to determine the score on a 
        test set after each boost.
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            y (ndarray): [n_samples] ndarray of true labels  
            
        Returns: 
            scores (ndarary): [n_learners] ndarray of scores 
        """

        scores = []
        
        # your code here
        for i in range(1, self.num_learners + 1):
            predictions = np.zeros(X.shape[0])
            for j in range(i):
                predictions += self.alphas[j] * self.learners[j].predict(X)
            scores.append(np.sum(np.sign(predictions) == y) / X.shape[0])
        
        return np.array(scores) 


# TEST 1 -- for Adaboost error rate function. 
y_true = [-1, 1, 1, -1, 1, -1, -1]
y_pred = [-1, 1, 1, 1, 1, -1, 1]
w = np.ones(len(y_true))
w /= np.sum(w)
clf = AdaBoost() 
err_rate = clf.error_rate(y_true, y_pred, w)
assert pytest.approx(err_rate, 0.01) == 0.2857, "Check the error_rate function."


# TEST 2 -- for Adaboost fit function.
sample_X = np.load('https://raw.githubusercontent.com/Vamboozer/AI/main/X.npy')
sample_y = np.load('https://raw.githubusercontent.com/Vamboozer/AI/main/y.npy')
test_model = AdaBoost(n_learners=5).fit(sample_X,sample_y)
t_alpha = [1.94591015, 2.14179328, 2.48490665, 1.94679667, 2.22627839]
assert pytest.approx(test_model.alpha, 0.01) == t_alpha, "Check the fit function"