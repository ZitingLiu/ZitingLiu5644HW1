import os
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
samples,y = prob_utils.generate_mixture_samples(N, n, gauss_params, False)

#print(samples)
#print("--------------------------------")
#print(y)



#print(samples0)
#print(samples1)
#print(samples)
m=np.transpose(m)
fig = plt.figure(figsize=plt.figaspect(0.5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(samples[0, :], samples[1, :], samples[2, :])
plt.show()

discriminant_score=np.zeros(N)

for i in range(N):
    discriminant_score[i]=np.log(multivariate_normal.pdf(samples[:,i],mean=m[1],cov=np.identity(3))/
                                    multivariate_normal.pdf(samples[:,i],mean=m[0],cov=np.identity(3)))

#print(discriminant_score)

threshold = np.sort(discriminant_score)


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
    decision=discriminant_score>threshold[i]
    p01[i]=np.sum(~decision & label)/sumOfLabel1
    p10[i]=np.sum(decision & ~label)/sumOfLabel0
    p11[i]=np.sum(decision & label)/sumOfLabel1
    perr[i]=p01[i]*classPrior[1]+p10[i]*classPrior[0]
#print(threshold)

threshold_ideal=np.log(classPrior[0]/classPrior[1])
decision_ideal=discriminant_score>threshold
p01_ideal=sum(~decision_ideal & label)/sumOfLabel0
p10_ideal=sum(decision_ideal & ~label)/sumOfLabel1
p11_ideal=sum(decision_ideal & label)/sumOfLabel1
perr_ideal=p01_ideal*classPrior[0]+p10_ideal*classPrior[1]

threshold_best = threshold[np.argmin(perr)]

print("Minimum Perr possible: "+str(min(perr)))
print("Threshold that achived minimun probability of error is :"+str(threshold_best)+" comparing to Optimal value :"+str(np.log(classPrior[0]/classPrior[1])))
#print(p01)
#print(p10)
#print(p11)
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
ax1.set_title('ROC curve with Mismatch class Q1b')
ax1.legend()
plt.tight_layout()
plt.savefig('Q1b_roc.jpg')
plt.show()

fig2 = plt.figure(figsize=[4, 3], dpi=150)
ax2 = fig2.add_subplot(111)
ax2.plot(threshold, perr, linewidth=1)
ax2.set_xlabel(r'threshold')
ax2.set_ylabel(r'$Perr$')
ax2.set_title('Perr with Mismatch class Q1b')
plt.tight_layout()
plt.savefig('Q1b_Perr.jpg')
plt.show()




