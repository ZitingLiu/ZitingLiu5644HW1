import os
from telnetlib import XASCII
import matplotlib.pyplot as plt # For general plotting
from matplotlib import cm
import numpy as np
from modules import models, prob_utils

from scipy.stats import multivariate_normal # MVN not univariate
from scipy import random, linalg

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
##==========================================================================================================================================
##Part B

# Number of samples to draw from each distribution
N = 10000

m= np.array([[-0.5, -0.5, 0.5],
                   [1, 1, 1]])
m=np.transpose(m)
#print(m)
c = np.array([[[1, -0.5,0.3],
                [-0.5, 1, -0.5],
                [0.3,-0.5,1]],

                [[1, 0.3,-0.2],
                [0.3, 1, 0.3],
                [-0.2,0.3,1]]])

classPrior=np.array([0.65,0.35])
C=len(classPrior)

gauss_params = prob_utils.GaussianMixturePDFParameters(classPrior,C,m,np.transpose(c))

gauss_params.print_pdf_params()

n=m.shape[0]
#print(n)
X,y = prob_utils.generate_mixture_samples(N, n, gauss_params, False)

#print(samples)
#print("--------------------------------")
#print(y)



#print(samples0)
#print(samples1)
#print(samples)
m=np.transpose(m)
fig = plt.figure(figsize=plt.figaspect(0.5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(X[0, :], X[1, :], X[2, :])
plt.show()

X=np.transpose(X)
W,z=models.perform_lda(X,y)
X=np.transpose(X)
WTX=np.dot(np.transpose(W),X)

#print(WTX)
threshold = np.sort(WTX)


sumOfLabel0=np.sum(y==0)
sumOfLabel1=np.sum(y==1)
#print(str(sumOfLabel0)+"and"+str(sumOfLabel1))

size=threshold.size
p01=np.zeros(size)
p10=np.zeros(size)
p11=np.zeros(size)
perr=np.zeros(size)
label=y==1

for i in range(size):
    decision=WTX>threshold[i]
    p01[i]=np.sum(~decision & label)/sumOfLabel1
    p10[i]=np.sum(decision & ~label)/sumOfLabel0
    p11[i]=np.sum(decision & label)/sumOfLabel1
    perr[i]=p01[i]*classPrior[1]+p10[i]*classPrior[0]

threshold_best = threshold[np.argmin(perr)]

print("Minimum Perr possible: "+str(min(perr)))
print("Threshold that achived minimun probability of error is :"+str(threshold_best))
fig1 = plt.figure(figsize=[4, 3], dpi=200)
ax1 = fig1.add_subplot(111)
ax1.plot(p10, p11, linewidth=1)
#print(p10[np.argmin(perr)])
#print(p11[np.argmin(perr)])
ax1.scatter(p10[np.argmin(perr)], p11[np.argmin(perr)], c='r',
            marker='x', label=r'minimum Perr')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_xlabel(r'$PFP$')
ax1.set_ylabel(r'$PTP$')
ax1.set_title('ROC curve with LDA Q1c')
ax1.legend()
plt.tight_layout()
plt.savefig('Q1c_roc.jpg')
plt.show()

fig2 = plt.figure(figsize=[4, 3], dpi=150)
ax2 = fig2.add_subplot(111)
ax2.plot(threshold, perr, linewidth=1)
ax2.set_xlabel(r'threshold')
ax2.set_ylabel(r'$Perr$')
ax2.set_title('Perr with LDA Q1c')
plt.tight_layout()
plt.savefig('Q1c_Perr.jpg')
plt.show()
