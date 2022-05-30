import os
import matplotlib.pyplot as plt # For general plotting
from matplotlib import cm
import matplotlib
import numpy as np
from modules import models, prob_utils
from sklearn.decomposition import PCA

from scipy.stats import multivariate_normal # MVN not univariate
from scipy import random, linalg
from sklearn.metrics import confusion_matrix
import csv
import pandas as pd
import warnings
from numpy import genfromtxt
warnings.filterwarnings("ignore")
N=7352
# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#
X = np.loadtxt('X_train.txt',usecols=(28,34,58) ,dtype=float) #read 3 features from X
print(X.shape)

temp= np.loadtxt('X_train.txt' ,dtype=float) #read 3 features from X
pca = PCA(n_components=3)
Xpca = pca.fit_transform(temp)


y = np.loadtxt('y_train.txt', dtype=int)   #read y
print(y.shape)

y=y-1 # change label from 1-6 to 0-5

numOfScore=np.zeros(6) 
for i in range(6):
    numOfScore[i]=np.count_nonzero(y == i)

class_prior=numOfScore/N    #get class prior

print(class_prior)

mean=np.zeros((6,3)) #6 class 3 feature

covs=np.zeros((6,3,3))
for i in range(6):
    covs[i]=np.add(np.cov(X[y==i].T),5*np.identity(3))

means=np.zeros((6,3))
for i in range(6):
    for j in range(3):
        means[i][j]=np.mean(X[y==i,j])

class_cond_likelihoods=np.zeros((6,N))
for i in range (N):
    for j in range(6):
        class_cond_likelihoods[j][i] = multivariate_normal.pdf(X[i], mean=means[j], cov=covs[j])


#print(class_cond_likelihoods)
class_priors = np.diag(class_prior)
class_posteriors = class_priors.dot(class_cond_likelihoods)

decisions = np.argmax(class_posteriors, axis=0)

print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions, y)
print(conf_mat)
correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Mumber of Misclassified Samples: {:d}".format(N - correct_class_samples))
prob_error = 1 - (correct_class_samples / N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))
#################################################################################
#PCA
covsPCA=np.zeros((6,3,3))
for i in range(6):
    covsPCA[i]=np.add(np.cov(Xpca[y==i].T),5*np.identity(3))

meansPCA=np.zeros((6,3))
for i in range(6):
    for j in range(3):
        meansPCA[i][j]=np.mean(Xpca[y==i,j])

class_cond_likelihoodsPCA=np.zeros((6,N))
for i in range (N):
    for j in range(6):
        class_cond_likelihoodsPCA[j][i] = multivariate_normal.pdf(Xpca[i], mean=meansPCA[j], cov=covsPCA[j])


#print(class_cond_likelihoods)
class_posteriorsPCA = class_priors.dot(class_cond_likelihoodsPCA)

decisionsPCA = np.argmax(class_posteriorsPCA, axis=0)

print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_matPCA = confusion_matrix(decisionsPCA, y)
print(conf_matPCA)

correct_class_samplesPCA = np.sum(np.diag(conf_matPCA))
print("Total Mumber of Misclassified Samples: {:d}".format(N - correct_class_samplesPCA))

prob_errorPCA = 1 - (correct_class_samplesPCA / N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_errorPCA))


#########################################################################################
fig1 = plt.figure(figsize=(4, 4), dpi=200)
ax1 = fig1.add_subplot(projection='3d')
for i in range(N):
    ax1.scatter(X[((decisions==i) & (y==i)),0 ], 
              X[((decisions==i) & (y==i)),1 ],
              X[((decisions==i) & (y==i)),2 ],  
              marker='.', c='green', alpha=.6,label="Class"+str(i)+"correct")
    ax1.scatter(X[ ((~(decisions==i)) & (y==i)),0], 
              X[ ((~(decisions==i)) & (y==i)),1], 
              X[ ((~(decisions==i)) & (y==i)),2], 
              marker='.', c='red', alpha=.6,label="Class"+str(i)+"wrong")
#ax1.set_xlim((-5, 5))
#ax1.set_ylim((-5, 5))
ax1.set_title('Q3 PartB')
plt.tight_layout()
#plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.05),
          #ncol=1, fancybox=True, shadow=True,)
plt.savefig('Q3B.jpg')
plt.show()


fig2 = plt.figure(figsize=(4, 4), dpi=200)
ax1 = fig2.add_subplot(projection='3d')
for i in range(N):
    ax1.scatter(Xpca[((decisionsPCA==i) & (y==i)),0 ], 
              Xpca[((decisionsPCA==i) & (y==i)),1 ],
              Xpca[((decisionsPCA==i) & (y==i)),2 ],  
              marker='.', c='green', alpha=.6,label="Class"+str(i)+"correct")
    ax1.scatter(Xpca[ ((~(decisionsPCA==i)) & (y==i)),0], 
              Xpca[ ((~(decisionsPCA==i)) & (y==i)),1], 
              Xpca[ ((~(decisionsPCA==i)) & (y==i)),2], 
              marker='.', c='red', alpha=.6,label="Class"+str(i)+"wrong")
#ax1.set_xlim((-5, 5))
#ax1.set_ylim((-5, 5))
ax1.set_title('Q3 PartB with PCA')
plt.tight_layout()
#plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.05),
          #ncol=1, fancybox=True, shadow=True,)
plt.savefig('Q3BPCA.jpg')
plt.show()