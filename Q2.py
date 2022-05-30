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

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#matplotlib.interactive(True)

N = 10000

m= np.array([[-2, -1],
                [-1, 1],
                [1,1],
                [1,-2]])
m=np.transpose(m)
#print(m)
c = np.array([[[1, 0],
                [0, 1]],

                [[1, 0],
                [0,1]],

                [[1,0],
                [0,1]],

                [[1,0],
                [0,1]]
                
                ])

classPrior=np.array([0.2,0.25,0.25,0.3])
C=len(classPrior)

gauss_params = prob_utils.GaussianMixturePDFParameters(classPrior,C,m,np.transpose(c))

gauss_params.print_pdf_params()

n=m.shape[0]
#print(n)
samples,y = prob_utils.generate_mixture_samples(N, n, gauss_params, False)

#print(samples)
#print(y)
#print(c.shape)
fig0 = plt.figure(figsize=(4, 4), dpi=200)
ax1 = fig0.gca()
color=['blue', 'red', 'yellow', 'green']
for i in range(C):
    ax1.scatter(samples[0, (y==i)],
              samples[1, (y==i)],
              marker='.', c=color[i],alpha=0.6,label="class"+str(i))
ax1.set_xlim((-5, 5))
ax1.set_ylim((-5, 5))
ax1.set_title('Q2 Samples')
plt.tight_layout()
plt.legend(loc='upper right', bbox_to_anchor=(0.9, 1.05),
          ncol=1, shadow=True)
plt.savefig('Q2_Samples.jpg')
plt.show()

class_cond_likelihoods=np.zeros((C,N))
m=np.transpose(m)
for i in range (N):
    for j in range(C):
        class_cond_likelihoods[j][i] = multivariate_normal.pdf(samples[:, i], mean=m[j], cov=c[j])
    
    

#print(class_cond_likelihoods)
class_priors = np.diag(classPrior)
class_posteriors = class_priors.dot(class_cond_likelihoods)

decisions = np.argmax(class_posteriors, axis=0)

print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions, y)
print(conf_mat)
correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Mumber of Misclassified Samples: {:d}".format(N - correct_class_samples))



prob_error = 1 - (correct_class_samples / N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))

marker = ['.', 'o', '^', 's']
fig1 = plt.figure(figsize=(4, 4), dpi=200)
ax1 = fig1.gca()
for i in range(C):
    ax1.scatter(samples[0, ((decisions==i) & (y==i))], 
              samples[1, ((decisions==i) & (y==i))], 
              marker=marker[i], c='green', alpha=.6,label="Class"+str(i)+"correct")
    ax1.scatter(samples[0, ((~(decisions==i)) & (y==i))], 
              samples[1, ((~(decisions==i)) & (y==i))], 
              marker=marker[i], c='red', alpha=.6,label="Class"+str(i)+"wrong")
ax1.set_xlim((-5, 5))
ax1.set_ylim((-5, 5))
ax1.set_title('Q2a')
plt.tight_layout()
#plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.05),
          #ncol=1, fancybox=True, shadow=True,)
plt.savefig('Q2A.jpg')
plt.show()


loss=[[0,1,2,3],
        [1,0,1,2],
        [2,1,0,1],
        [3,2,1,0]]

decision2=np.zeros(N)

for i in range(N):
    posterior=np.zeros((C,1))
    for j in range(C):
        posterior[j]=multivariate_normal.pdf(samples[:,i],mean=m[j],cov=c[j])*classPrior[j]

    decision2[i]=np.argmin(np.dot(loss,posterior))

correct=0
for i in range(C):
    for j in range(N):
        if (decision2[j]==i) & (y[j]==i):
            correct+=1

print(correct/N)



fig2 = plt.figure(figsize=(4, 4), dpi=200)
ax1 = fig2.gca()
for i in range(C):
    ax1.scatter(samples[0, ((decision2==i) & (y==i))], 
              samples[1, ((decision2==i) & (y==i))], 
              marker=marker[i], c='green', alpha=.6,label="Class"+str(i)+"correct")
    ax1.scatter(samples[0, ((~(decision2==i)) & (y==i))], 
              samples[1, ((~(decision2==i)) & (y==i))], 
              marker=marker[i], c='red', alpha=.6,label="Class"+str(i)+"wrong")
ax1.set_xlim((-5, 5))
ax1.set_ylim((-5, 5))
ax1.set_title('Q2b')
plt.tight_layout()
#plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.05),
          #ncol=1, fancybox=True, shadow=True,)
plt.savefig('Q2b.jpg')
plt.show()






