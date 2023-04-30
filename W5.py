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
        self.train_scores = []
        
    def fit(self, X_train, y_train):
        """
        Train AdaBoost classifier on data. Sets alpha and learners. 
        
        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data 
        """
        
        # your code here
        w = np.ones(len(y_train)) / len(y_train)

        for i in range(self.n_learners):
            # fit the base learner with the weighted data
            learner = clone(self.base)
            learner.fit(X_train, y_train, sample_weight=w)

            # compute the weighted predictions of the learner
            pred = learner.predict(X_train)
            #err_k = self.error_rate(y_train, pred, w)
            err_k = 1 - np.sum(w * np.equal(y_train, pred)) / np.sum(w)
            alpha_k = 0.5 * np.log((1 - err_k) / err_k)

            # update the weights
            w *= np.exp(-alpha_k * y_train * pred)
            w /= np.sum(w)
            # print(np.sum(w))

            # append the learner and alpha to their respective lists
            self.learners.append(learner)
            self.alpha[i] = alpha_k

            # compute and store the score for this iteration
            score_train = self.score(X_train, y_train)
            self.train_scores.append(score_train)
        
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

        # initialize predictions
        y_pred = np.zeros(len(X))

        # for each learner, compute the weighted prediction and add it to y_pred
        for i in range(len(self.learners)):
            pred = self.learners[i].predict(X)
            y_pred += self.alpha[i] * pred

        # convert predictions to {-1, 1}
        y_pred = np.sign(y_pred)

        return y_pred
    
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
print(t_alpha)
print(test_model.alpha)
assert pytest.approx(test_model.alpha, 0.01) == t_alpha, "Check the fit function"


# create the Adaboost classifier with 150 base decision tree stumps
clf = AdaBoost(base=DecisionTreeClassifier(max_depth=1), n_learners=150)

# fit the classifier to the training data
clf.fit(data.x_train, data.y_train)

# ====== Part B ======

# print out predictions on the training set 

train_predict = clf.predict(data.x_train)

print(train_predict)

max(train_predict)

# ====== Part C ======

# compare to actual labels
print('Training set accuracy:', np.mean(train_predict == data.y_train))

# compute scores for train and test sets
train_scores = clf.staged_score(data.x_train, data.y_train)
test_scores = clf.staged_score(data.x_test, data.y_test)

# compute misclassification errors
train_errors = 1 - np.array(train_scores)
test_errors = 1 - np.array(test_scores)

# plot errors
plt.plot(range(1, clf.n_learners + 1), train_errors, label='train')
plt.plot(range(1, clf.n_learners + 1), test_errors, label='test')
plt.xlabel('Number of Boosting Iterations')
plt.ylabel('Misclassification Error')
plt.title('AdaBoost')
plt.legend()
plt.show()