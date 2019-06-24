#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Estimation of Number of Spectrally Distinct Signal Sources in Hyperspectral Imagery Chein-I Chang, Senior Member, IEEE, and Qian Du, Member, IEEE

import numpy as np
import numpy
from scipy import linalg as LA
#from statistics import NormalDist
from scipy.stats import norm
from math import sqrt
import scipy.io as spio
import scipy.io
import scipy.io as sio
import math 
from scipy.linalg import fractional_matrix_power
from scipy.special import erfinv


#ENTER THE MATRIX IN 2D OR 3D FORM 
mat=input('enter the file name to give data in same directory:')
mat_contents = sio.loadmat(mat)
M = mat_contents['total_samples_new']
M=numpy.transpose(M)
print("shape of M original",np.shape(M))



#IF IT IS 3D CONVERTION TAKES PLACE TO 2D HEAR
if (3 == len(np.shape(M))):
    print ("Converting to 2D matrix")
    [XX,YY,bnd] = np.shape(M)
    pxl_no = XX*YY
    M = numpy.transpose(M)(np.reshape(M,pxl_no,bnd))  
    print("shape of M original",np.shape(M))
    
    
    
#ENTER THE VALUES OF FAR
#fardata = int(input("TYPE 1 FOR FAR DATA GIVEN,2 FOR FAR DATA NOT GIVEN: "))
#if (fardata == 1):
#    n_rows = int(input("Enter number of rows: "))
#   n_cols = int(input("Enter number of columns: "))
 
#    far = [[int(input("Enter value for {}. row and {}. column: ".format(r + 1, c + 1))) for c in range(n_cols)] for r in range(n_rows)]

#    print(np.array(far))
#else:
#    far = [pow(10,-1),pow(10,-2),pow(10,-3),pow(10,-4),pow(10,-5)]
far = [pow(10,-1), pow(10,-2), pow(10,-3), pow(10,-4), pow(10,-5)]
q = [0, 0, 0, 0, 0]

numBands = np.shape(M)[0]
N = np.shape(M)[1]
corr=np.matmul(M,(numpy.transpose(M)))/(N-1) 
u = np.mean(M, axis = 1)
cov = (corr-np.matmul(u,numpy.transpose(u)))
#cov=np.dot((numpy.transpose(M-u)),(M-u))/N
#corr=cov+numpy.transpose(u)*u
a = (np.linalg.eigvals(cov))
b = (np.linalg.eigvals(corr))
#a,c = (LA.eig(cov))
#b,d = (LA.eig(corr))
lambdaCov = a[::-1]
lambdaCorr = b[::-1]
    

#FINDING ENDMEMBERS BY HFC METHOD  
def hfc( M,far ):
    #numBands = np.shape(M)[0]
    #N = np.shape(M)[1]
    #corr=np.dot(M,(numpy.transpose(M)))/N 
    #u = np.mean(M, axis = 1)
    #cov = corr-u*numpy.transpose(u)
    #a,c = (LA.eig(cov))
    #b,d = (LA.eig(corr))
    #lambdaCov = a[::-1]
    #lambdaCorr = b[::-1]
    for y in range(0, len(far)):
        numEndmembers = 0
        pf = far[y]
        for x in range(0, numBands-1):
            sigmaSquared = (2/N)*(pow((lambdaCov[x]),2) + pow((lambdaCorr[x]),2))
            #sigmaSquared = (2/N)*((lambdaCov[x])+(lambdaCorr[x]))
            #sigmaSquared = np.var(lambdaCorr[x])+np.var(lambdaCov[x])
            sigma = np.sqrt(sigmaSquared)
            # OR tau = -(NormalDist(mu=0, sigma=sigma).inv_cdf(pf))
            #tau = -norm.ppf(pf, loc=0, scale=sigma)
            #tau = pow((2*sigmaSquared*(np.log(2.5*sigma*pf)/np.log(100))),0.5)
            tau = (sqrt(2))*sigma*erfinv(1-(2*pf))
            if ((lambdaCorr[x]-lambdaCov[x]) > tau):
                numEndmembers = numEndmembers + 1
        q[y]=numEndmembers
        print ("Values of endmembers for : ",pf , q[y])
    return    
    
    
#FINDING ENDMEMBERS BY NWHFC METHOD    
def nwhfc( M,far ):
    #numBands = np.shape(M)[0]
    #N = np.shape(M)[1]
    #corr=numpy.dot(M,(numpy.transpose(M)))/N
    #u = np.mean(M, axis = 1)
    #cov = corr-u*numpy.transpose(u)
    K_Inverse=np.linalg.inv(cov)
    tuta=K_Inverse.diagonal()
    K_noise=np.true_divide(1, tuta)
    K_noise=np.diag(K_noise)
    Y=np.linalg.solve(scipy.linalg.sqrtm(K_noise),(M))
    
    hfc(Y,far)
    return
    
    
#FINDING ENDMEMBERS BY NSP VD METHOD
def nsw( M,far ):
    numBands = np.shape(M)[0]
    N = np.shape(M)[1]
    corr=np.matmul(M,(numpy.transpose(M)))/(N-1)
    u = np.mean(M, axis = 1)
    cov = (corr-np.matmul(u,numpy.transpose(u)))
    K=cov
    
    K_Inverse=np.linalg.inv(K)
    tuta=K_Inverse.diagonal()
    K_noise=np.true_divide(1, tuta)
    K_noise=np.diag(K_noise)
    #Y=np.linalg.solve(scipy.linalg.sqrtm(K_noise),(M))
    #knoise=np.linalg.inv(scipy.linalg.sqrtm(K_noise))
    knoise=pow(K_noise,0.5)
    knoise=np.linalg.inv(knoise)
    #knoise=fractional_matrix_power(K_noise, -0.5)
    cov=numpy.dot(knoise,K)
    cov=numpy.dot(cov,knoise)
    #numBands = np.shape(Y)[0]
    #N = np.shape(Y)[1]
    #corr=np.matmul(Y,(numpy.transpose(Y)))/N-1
    #u = np.mean(Y, axis = 1)
    #cov = corr-np.matmul(u,numpy.transpose(u))
    a = (np.linalg.eigvals(cov))
    lambdaCov = a[::-1]


    for y in range(0, len(far)):
        numEndmembers = 0
        pf = far[y]
        sigma = pow((2/N),0.5)
        # OR tau = -(NormalDist(mu=0, sigma=sigma).inv_cdf(pf))
        #tau = -norm.ppf(pf, loc=1, scale=sigma)
        tau = (sqrt(2))*sigma*erfinv(1-(2*pf))
        
        for x in range(0, numBands-1):
            #sigma = pow(((2/N)*(pow((lambdaCov[x]),2))),0.5)
            #tau = (pow(2,0.5))*sigma*scipy.special.erfinv(1-(2*(pf)))
            if ((lambdaCov[x]) > (1+tau)):
                numEndmembers = numEndmembers + 1
        q[y]=numEndmembers
        print ("Values of endmembers: ",pf , q[y])
    return

   
# SELECTION OF METHOD
method = (input("METHOD = "))
if(method=='hfc'): 
    print ("YOU HAV CHOOSEN HFC METHOD")
    hfc( M,far ) 
elif (method=='nwhfc'):
    print ("YOU HAV CHOOSEN NWHFC METHOD")
    nwhfc( M,far )
elif(method=='nsw'):    
    print ("YOU HAV CHOOSEN NSW METHOD")
    nsw(M,far)


# In[ ]:


#Hyperspectral Subspace Identification  José M. Bioucas-Dias, Member, IEEE, and José M. P. Nascimento, Member, IEEE

import numpy as np
import numpy
from scipy import linalg as LA
from scipy.stats import norm
from math import sqrt
import scipy.io as spio
import scipy.io
import scipy.io as sio
import math 
from scipy.linalg import fractional_matrix_power
from scipy.special import erfinv
from numpy.linalg import eig
import numpy as geek 
from numpy.linalg import inv
from sympy import Matrix, init_printing


#ENTER THE MATRIX 
mat=input('enter the file name to give data in same directory:')
mat_contents = sio.loadmat(mat)
M = mat_contents['total_samples_new']
M =numpy.transpose(M)
print("shape of M original",np.shape(M))



#IF IT IS 3D CONVERTION TAKES PLACE TO 2D HEAR
if (3 == len(np.shape(M))):
    print ("Converting to 2D matrix")
    [XX,YY,bnd] = np.shape(M)
    pxl_no = XX*YY
    new_img = M.reshape(pxl_no,bnd)
    M = new_img.transpose() 
    print("shape of M original",np.shape(M))
    ban=input('enter the file name to give data in same directory:')
    bands = sio.loadmat(ban)
    b = bands['BANDS']
    M= M[b-1,:]
    [XX,YY,bnd] = np.shape(M)
    pxl_no = XX*YY
    M = M.reshape(pxl_no,bnd)
    print("shape of M original",np.shape(M))


#Hyperspectral signal subspace estimation

def subspaceest( M ,w):
    #mat=input('enter the file name to give data in same directory:')
    mat_contents = sio.loadmat('wdata')
    w = mat_contents['w']
    print("shape of rra original",w)
    l=np.shape(M)[0]
    n=np.shape(M)[1]
    ln=np.shape(w)[0]
    nn=np.shape(w)[1]
    x = M-w
    Rx=np.matmul(x,(numpy.transpose(x)))/n
    Ry=np.matmul(M,(numpy.transpose(M)))/n
    Rw=np.matmul(w,(numpy.transpose(w)))/n
    Rw = Rw + (np.sum(np.diag(Rx), axis=0)/l)
    d1=np.shape(Rw)[0]
    d2=np.shape(Rw)[1]
    b,d = (eig(Rx))
    #d, s, b = np.linalg.svd(Rx, full_matrices=True)
   # lambdaCorr = b[::-1]
    #print("shape of w original",lambdaCorr)
    #print("shape of w original",lambdacorrvec)
    py=np.matmul((numpy.transpose(d)),Ry)
    py=np.matmul(py,d)
    py=np.diag(py)
    pn=np.matmul((numpy.transpose(d)),Rw)
    pn=np.matmul(pn,d)
    pn=np.diag(pn)
    cost_F = -py + 2 * pn
    print("shape of rra original",np.shape(cost_F))
    cost_F=cost_F[::-1]
  #  print("shape of rra original",cost_F)
    kf=len(list(filter(lambda x: (x < 0), cost_F)))
    print ("The signal subspace dimension is: ", kf)
    ek=d[:,0:kf]
  #  print ("matrix which columns are the eigenvectors that span the signal subspace : ", ek)
    print("shape of ek transfered",np.shape(ek))
    return


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier



# hyperspectral noise estimation

def noiceest( M ):
    l=np.shape(M)[0]
    n=np.shape(M)[1]
 #   M=pow(M,0.5)  
    small =pow(10,-6)
    w = np.zeros((l, n))
    corr = np.dot(M,(numpy.transpose(M)))#np.dot(M,M.T)/n
    #corinv =pow(corr,-1)#+(small*geek.identity(l))#np.linalg.inv(corr+(small*np.identity(l)))/1000 #
    corinv = np.linalg.inv(corr)
    #corinv = scipy.linalg.inv(corr)#/10000
   # corinv = corinv.round()*10
    #corinv = (np.mat(corr)).I
    print ("noise estimates for every pixel : ", corinv)
    for i in range(0, l):
        Ainv = numpy.subtract(corinv , (np.dot(corinv[i,:],corinv[:,i])/corinv[i,i]))
        RRa = corr[:,i]
        RRa[i]=0
        beta = np.dot(Ainv,(RRa))
        beta[i]=0
        w[i,:] = M[i,:] - np.dot((numpy.transpose(beta)),(M))
  #  x = pow((M-w),2)#np.square(M-w)
 #   w=((pow(x,0.5)*w))*2
    print("shape of w original",(w))
  #  Rw=np.diag(np.diag(np.matmul(w,(numpy.transpose(w)))/n-1))
  #  print ("noise correlation matrix : ", Rw)
    subspaceest(M,w)
    return

noiceest( M )


# In[ ]:


#Empirical Automatic Estimation of the Number of Endmembers in Hyperspectral Images Bin Luo, Jocelyn Chanussot, Fellow, IEEE, Sylvain Douté, and Liangpei Zhang, Senior Member, IEEE

import numpy as np
import numpy
from scipy import linalg as LA
#from statistics import NormalDist
from scipy.stats import norm
from math import sqrt
import scipy.io as spio
import scipy.io
import scipy.io as sio
import math 
from scipy.linalg import fractional_matrix_power
from scipy.special import erfinv
from numpy.linalg import eig
import numpy as geek 


#ENTER THE MATRIX 
mat=input('enter the file name to give data in same directory:')
mat_contents = sio.loadmat(mat)
M = mat_contents['total_samples_new']
M=numpy.transpose(M)
print("shape of M original",np.shape(M))


#IF IT IS 3D CONVERTION TAKES PLACE TO 2D HEAR
if (3 == len(np.shape(M))):
    print ("Converting to 2D matrix")
    [XX,YY,bnd] = np.shape(M)
    pxl_no = XX*YY
    new_img = M.reshape(pxl_no,bnd)
    M = new_img.transpose() 
    print("shape of M original",np.shape(M))
    ban=input('enter the file name to give data in same directory:')
    bands = sio.loadmat(ban)
    b = bands['BANDS']
    M= M[b-1,:]
    [XX,YY,bnd] = np.shape(M)
    pxl_no = XX*YY
    M = M.reshape(pxl_no,bnd)
    print("shape of M original",np.shape(M))
    
#ENTER THE VALUES OF FAR
#fardata = int(input("TYPE 1 FOR FAR DATA GIVEN,2 FOR FAR DATA NOT GIVEN: "))
#if (fardata == 1):
#    n_rows = int(input("Enter number of rows: "))
#   n_cols = int(input("Enter number of columns: "))
 
#    far = [[int(input("Enter value for {}. row and {}. column: ".format(r + 1, c + 1))) for c in range(n_cols)] for r in range(n_rows)]

#    print(np.array(far))
#else:
#    far = [pow(10,-1),pow(10,-2),pow(10,-3),pow(10,-4),pow(10,-5)]
#  far = [pow(10,-1), pow(10,-2), pow(10,-3), pow(10,-4), pow(10,-5)]
#hlogm = [0, 0, 0, 0, 0]


    

#eigenvalue likelihood maximization  
def elm( M,w ):
    numBands = np.shape(M)[0]
    N = np.shape(M)[1]
    M = M-w
    corr=np.dot(M,(numpy.transpose(M)))/(N) 
    u = np.mean(M, axis = 1)
    cov = (corr-np.dot(numpy.transpose(u),u))
#    cov=np.dot((numpy.transpose(M-u)),(M-u))/N
#    corr=cov+numpy.transpose(u)*u
    a = (np.linalg.eigvals(cov))
    b = (np.linalg.eigvals(corr))
#    a,c = (LA.eig(cov))
#    b,d = (LA.eig(corr))
    a=numpy.transpose(a)
    b=numpy.transpose(b)
#    a=cov.eigenvalues()
#    b=corr.eigenvalues()
#    a = a.real*pow(10,9)
#    b = b.real*pow(10,9)
    lambdaCov = a[::-1]
#    lambdaCov = -np.sort(-a)
#    lambdaCov=numpy.transpose(lambdaCov)
#    print("shape of M original",(lambdaCov))
    lambdaCorr = b[::-1]
#    lambdaCorr = -np.sort(-b)
#    lambdaCorr =numpy.transpose(lambdaCorr)
    
    hlog=np.zeros((1,numBands))
    hlog = numpy.transpose(hlog)

    for x in range(0, numBands):
        sigmaSquared = (2/N)*(pow((lambdaCov[x]),2) + pow((lambdaCorr[x]),2))
        #sigmaSquared = (2/N)*((lambdaCov[x])+(lambdaCorr[x]))
        #sigmaSquared = np.var(lambdaCorr[x])+np.var(lambdaCov[x])
        #sigma = np.sqrt(sigmaSquared)
        sigma = pow(sigmaSquared,0.5)
        # OR tau = -(NormalDist(mu=0, sigma=sigma).inv_cdf(pf))
        #tau = -norm.ppf(pf, loc=0, scale=sigma)
        tau = pow((2*sigmaSquared*(np.log(2.5*sigma*pow(10,-3))/np.log(100))),0.5)
        #tau = (sqrt(2))*sigma*erfinv(1-(2*(pow(10,-3))))
        #tau = (lambdaCorr[x]-lambdaCov[x])
        #print("shape of M original",(sigma))
        #print("shape of M original",(tau))
       # h = (1/sigma)*(math.exp(-pow(tau,2)/(2*pow(sigma,2))))
       # hlog[x]=np.log(h)
        hlog[x]=-((pow(tau,2)/(2*pow(sigma,2)))+np.log(sigma))
    hlog = numpy.transpose(hlog)
#    print ("Values of endmembers for : ", hlog)
    maxHlog =np.argmax(hlog, axis=1) #hlog.index(max(hlog))
    print ("Values of endmembers for : ", (maxHlog-1))
    return    

# noice estimation
def noiceest( M ):
    l=np.shape(M)[0]
    n=np.shape(M)[1]
 #   M=pow(M,0.5)  
 #   small =pow(10,-6)
    w = np.zeros((l, n))
    corr = np.dot(M,(numpy.transpose(M)))
    corinv =np.linalg.inv(corr) #pow((corr+(small*geek.identity(l))),-1)#+(small*np.identity(l))
    #print ("noise estimates for every pixel : ", corinv)
    for i in range(0, l):
        Ainv = (corinv - ((corinv[:,i]*corinv[i,:])/corinv[i,i]))
        RRa = corr[:,i]
        RRa[i]=0
        beta = np.matmul(Ainv,(RRa))
        beta[i]=0
        w[i,:] = M[i,:] - np.dot((numpy.transpose(beta)),(M))
  #  x = pow((M-w),2)#np.square(M-w)
 #   w=((pow(x,0.5)*w))*2
 #   print("shape of w original",np.shape(w))
  #  Rw=np.diag(np.diag(np.matmul(w,(numpy.transpose(w)))/n-1))
  #  print ("noise correlation matrix : ", Rw)
    elm( M ,w )
    return

noiceest( M )


# In[ ]:


#Linear Spectral Mixture Analysis Based Approaches to Estimation of Virtual Dimensionality in Hyperspectral Imagery Chein-I Chang, Fellow, IEEE, Wei Xiong, Weimin Liu, Member, IEEE, Mann-Li Chang, Chao-Cheng Wu, and Clayton Chi-Chang Chen

import numpy as np
import numpy
from scipy import linalg as LA
#from statistics import NormalDist
from scipy.stats import norm
from math import sqrt
import scipy.io as spio
import scipy.io
import scipy.io as sio
import math 
from scipy.linalg import fractional_matrix_power
from scipy.special import erfinv
from numpy import zeros, newaxis



mat=input('enter the file name main data to give data in same directory:')
mat_contents = sio.loadmat(mat)
M = mat_contents['total_samples_new']
M=numpy.transpose(M)
HIM = M[:, :, newaxis]
print("shape of M original",np.shape(HIM))

if (1==np.shape(HIM)[2]):
    HIM = np.transpose(HIM);
    [ns,nl,nb]=np.shape(HIM)
print("shape of M original",np.shape(HIM))

maxi = 0

for i in range(0, ns):
    for j in range(0, nl):
        r = (HIM[i,j,:])
        bright = np.dot(np.transpose(r),r)
        #print("shape of M original",bright)
        if (bright>maxi):
            maxi = bright
            posx = i
            posy = j

t0 = HIM[posx,posy,:]
u = np.zeros((1,167))
u = u+t0
u = np.transpose(u)

# LSM METHODS
p=input('enter the asumed number of endmembers:')
p= int(p)
for i in range(0, p):
    uc = u[:,0:i]
    b=(np.dot(uc,np.linalg.pinv(uc)))
    pu = np.identity(nb)-b
    #pu = i-np.dot(np.dot(uc,np.linalg.pinv(np.dot(np.transpose(uc),uc))),np.transpose(uc)) 
    #print("shape of M original",b)
    maxi = 0
    for m in range(0, ns):
        for n in range(0, nl):
            r = (HIM[m,n,:])
            r = np.dot(pu,r)
            bright = np.dot(np.transpose(r),r)
            if (bright>maxi):
                maxi = bright
                posx = m
                posy = n
    print("max norm was found in ",posx+1,posy+1)
    t0 = [HIM[posx,posy,:]]
    t0 = np.transpose(t0)
    if (i==0):
        u=t0
    else:
        u = np.append(u, t0, axis=1)
            
print("shape of M original",u)


# In[ ]:


#Multispectral and Hyperspectral Image Analysis with Convex Cones Agustin Ifarraguerri, Student Member, IEEE, and Chein-I Chang, Senior Member, IEEE

import numpy as np
import numpy
from scipy import linalg as LA
from scipy.linalg import solve
#from statistics import NormalDist
from scipy.stats import norm
from math import sqrt
import scipy.io as spio
import scipy.io
import scipy.io as sio
import math 
from scipy.linalg import fractional_matrix_power
from scipy.special import erfinv


#ENTER THE MATRIX IN 2D OR 3D FORM 
mat=input('enter the file name to give data in same directory:')
mat_contents = sio.loadmat(mat)
M = mat_contents['total_samples_new']
M=numpy.transpose(M)
print("shape of M original",np.shape(M))


#IF IT IS 3D CONVERTION TAKES PLACE TO 2D HEAR
if (3 == len(np.shape(M))):
    print ("Converting to 2D matrix")
    [XX,YY,bnd] = np.shape(M)
    pxl_no = XX*YY
    M = numpy.transpose(M)(np.reshape(M,pxl_no,bnd))  
    print("shape of M original",np.shape(M))
    
#NORMALISATION
for i in range(0, np.shape(M)[1]-1):
    normM = numpy.linalg.norm(M[:,i])
    M[:,i] = M[:,i] / normM
    
    
#EXTRACT EIGEN VECTORS        
numBands = np.shape(M)[0]
N = np.shape(M)[1]
corr=np.matmul(M,(numpy.transpose(M)))#/(N-1) 
b,d = (LA.eigh(corr))
lambdaCorrvec = d[:,numBands-6:numBands]
#print("shape of M original",(b))
#print("shape of M original",(d))
#lambdaCorrval = np.sort(b)[::-1]
#lambdaCorrvec = np.sort(d)[::-1]
#for i in range(0, np.shape(b)[0]):
    #for j in range(0, np.shape(b)[0]):
        #if ((b[i]==lambdaCorrval[j])):
            #lambdaCorrvec[j]=d[i]
        #else:
            #continue
#print("shape of M original",(lambdaCorrvec))
#e = int(input("Enter your value of endmembers: "))
#lambdaCorrvec = lambdaCorrvec[0:6,:]
#lambdaCorrvec = numpy.transpose(lambdaCorrvec)
#print("shape of lambdaCorrvec original", np.shape(lambdaCorrvec))


#ALGORITHOM
endmemberselect=0
p=6
E = np.zeros((numBands,p))
while (endmemberselect < p):
    selector=np.zeros((numBands,1))
    nselected=0
    while (nselected < p-1):
        i=math.trunc(np.random.uniform(0,1)*numBands)
        if not selector[i]:
            selector[i]=1
            nselected=nselected+1
    selector=(np.nonzero(selector))
    selector= selector[0]
    #print("shape of selec original", (selector))
    P=np.zeros((p-1,p-1))
    #
    p1=np.zeros((p-1,1))
    for i in range(0,p-1):
        P[i,:]=lambdaCorrvec[selector[i],1:p]
        p1[i]=lambdaCorrvec[selector[i],0]
    A=np.linalg.solve(P, -p1)
    A = (np.dot(np.linalg.pinv(P),p1))
    b = 1
    A = np.vstack ((A, b) ) 
   # print("shape of selec original", A)
   # print("shape of selec original", lambdaCorrvec)
    x = np.matmul(lambdaCorrvec,A)
   # print("shape of selec original", min(x))
    
    
    if (min(x)>(-0.11)):
        E[:,endmemberselect]=x[:,0]
        endmemberselect = endmemberselect+1
        
        
    print("shape of endmemberselect original", endmemberselect)
print("shape of endmemberselect original", E)


# In[ ]:


#Automatic Target Recognition for Hyperspectral Imagery using High-Order Statistics

import numpy as np
import numpy
from scipy import linalg as LA
from scipy.linalg import solve
#from statistics import NormalDist
from scipy.stats import norm
from math import sqrt
import scipy.io as spio
import scipy.io
import scipy.io as sio
import math 
from scipy.linalg import fractional_matrix_power
from scipy.special import erfinv


#ENTER THE MATRIX IN 2D OR 3D FORM 
#mat=input('enter the file name to give data in same directory:')
mat_contents = sio.loadmat('total_samples_new')
M = mat_contents['total_samples_new']
#M=numpy.transpose(M)
print("shape of M original",np.shape(M))


#IF IT IS 3D CONVERTION TAKES PLACE TO 2D HEAR
if (3 == len(np.shape(M))):
    print ("Converting to 2D matrix")
    [XX,YY,bnd] = np.shape(M)
    pxl_no = XX*YY
    M = numpy.transpose(M)(np.reshape(M,pxl_no,bnd))  
    print("shape of M original",np.shape(M))
    

#Sphering    

numBands = np.shape(M)[0]
N = np.shape(M)[1]    
mu = np.zeros(numBands)
for i in range(0,numBands): 
    for j in range(0,N):
        mu[i] = mu[i] + M[i,j]

x = M - np.transpose(np.array(([mu,])*N))        
print("shape of Mu original",np.shape(x))


cov=np.matmul(x,(numpy.transpose(x)))/(N)
a,c = (LA.eig(cov))
a = np.diag(a)
a = np.sqrt(a)
spemat = np.dot(c,a)
print("shape of Mu original",np.shape(spemat))
y = np.dot(spemat,M)
print("shape of Mu original",np.shape(y))

#Algorithm for Finding Projection Images for ATR
sig = np.zeros((numBands,6))
temp = (numpy.multiply(y,y)).sum(axis=0)
ind = numpy.where(temp == numpy.amax(temp))
ind = ind[0]
sig[:,:1] = y[:,ind]


for i in range(1, 6):
    u = sig
    b=(np.dot(u,np.linalg.pinv(u)))
    pu = np.identity(numBands)-b
    yn = np.dot(pu,y)
    temp = (numpy.multiply(yn,yn)).sum(axis=0)
    ind = numpy.where(temp == numpy.amax(temp))
    ind = ind[0]
    sig[:,i:i+1] = y[:,ind]
print("shape of Mu original",np.shape(sig))
print("shape of Mu original",sig)

