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
N=4898
# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

wine = genfromtxt('winequality-white.csv', delimiter=';', skip_header = 1)
#print(wine.shape)


#score0=wine[wine[:,11]==0]
y=np.zeros(N) #true lable
for i in range(N):
    y[i]=wine[i][11]

y=y-3

numOfScore=np.zeros(7) #score range from 3-9 ,7 classes
for i in range(7):
    temp=wine[y==i]
    numOfScore[i]=len(temp)

class_prior=numOfScore/N



print(numOfScore)

wine=np.delete(wine,11,1)#get rid of last row(score)
#print(wine)
pca = PCA(n_components=3)
wine = pca.fit_transform(wine)

mean=np.zeros((7,3)) 


#test=wine[y==6]
#print(test)
covs=np.zeros((7,3,3))
for i in range(7):
    covs[i]=np.add(np.cov(wine[y==i].T),5*np.identity(3))

means=np.zeros((7,3))
for i in range(7):
    for j in range(3):
        means[i][j]=np.mean(wine[y==i,j])

#print(means)
#print(covs)
class_cond_likelihoods=np.zeros((7,N))
for i in range (N):
    for j in range(7):
        class_cond_likelihoods[j][i] = multivariate_normal.pdf(wine[i], mean=means[j], cov=covs[j])


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
    


fig1 = plt.figure(figsize=(4, 4), dpi=200)
ax1 = fig1.add_subplot(projection='3d')
for i in range(N):
    ax1.scatter(wine[((decisions==i) & (y==i)),0 ], 
              wine[((decisions==i) & (y==i)),1 ],
              wine[((decisions==i) & (y==i)),2 ],  
              marker='.', c='green', alpha=.6,label="Class"+str(i)+"correct")
    ax1.scatter(wine[ ((~(decisions==i)) & (y==i)),0], 
              wine[ ((~(decisions==i)) & (y==i)),1], 
              wine[ ((~(decisions==i)) & (y==i)),2], 
              marker='.', c='red', alpha=.6,label="Class"+str(i)+"wrong")
#ax1.set_xlim((-5, 5))
#ax1.set_ylim((-5, 5))
ax1.set_title('Q3a')
plt.tight_layout()
#plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.05),
          #ncol=1, fancybox=True, shadow=True,)
plt.savefig('Q3A.jpg')
plt.show()


