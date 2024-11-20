import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


def ldaLearn(X, y):
    classes = np.unique(y)
    d = X.shape[1]
    k = len(classes)
    means = np.zeros((d, k))
    for idx, cls in enumerate(classes):
        class_samples = X[y.flatten() == cls]
        means[:, idx] = np.mean(class_samples, axis=0)
    covmat = np.cov(X, rowvar=False)
    return means, covmat



def qdaLearn(X, y):
    classes = np.unique(y)
    d = X.shape[1]
    k = len(classes)
    means = np.zeros((d, k))
    covmats = []
    for idx, cls in enumerate(classes):
        class_samples = X[y.flatten() == cls]
        means[:, idx] = np.mean(class_samples, axis=0)
        covmats.append(np.cov(class_samples, rowvar=False))
    return means, covmats



def ldaTest(means, covmat, Xtest, ytest):
    inv_covmat = np.linalg.inv(covmat)
    likelihoods = []
    for mean in means.T:
        diff = Xtest - mean
        likelihood = -0.5 * np.sum(diff @ inv_covmat * diff, axis=1)
        likelihoods.append(likelihood)
    likelihoods = np.array(likelihoods)
    ypred = np.argmax(likelihoods, axis=0) + 1
    acc = np.mean(ypred == ytest.flatten()) * 100
    return acc, ypred.reshape(-1, 1)



def qdaTest(means, covmats, Xtest, ytest):
    k = means.shape[1]
    likelihoods = []
    for cls in range(k):
        mean = means[:, cls]
        covmat = covmats[cls]
        inv_covmat = np.linalg.inv(covmat)
        det_covmat = np.linalg.det(covmat)
        diff = Xtest - mean
        exponent = -0.5 * np.sum(diff @ inv_covmat * diff, axis=1)
        likelihood = exponent - 0.5 * np.log(det_covmat)
        likelihoods.append(likelihood)
    likelihoods = np.array(likelihoods)
    ypred = np.argmax(likelihoods, axis=0) + 1
    acc = np.mean(ypred == ytest.flatten()) * 100
    return acc, ypred.reshape(-1, 1)


def learnOLERegression(X, y):
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w

def learnRidgeRegression(X, y, lambd):
    d = X.shape[1]
    identity = np.eye(d)
    w = np.linalg.inv(X.T @ X + lambd * identity) @ X.T @ y
    return w



def testOLERegression(w, Xtest, ytest):
    y_pred = Xtest @ w
    mse = np.mean((y_pred - ytest) ** 2)
    return mse


def regressionObjVal(w, X, y, lambd):
    w = w.reshape(-1, 1)
    N = X.shape[0]
    error = (1 / (2 * N)) * np.sum((y - X @ w) ** 2) + (lambd / 2) * np.sum(w ** 2)
    error_grad = -(1 / N) * (X.T @ (y - X @ w)) + lambd * w
    return error, error_grad.flatten()


def mapNonLinear(x, p):
    Xp = np.ones((x.shape[0], p + 1))
    for i in range(1, p + 1):
        Xp[:, i] = x[:, 0] ** i
    return Xp


if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'), encoding='latin1')


print("Sample Dataset:")
print("Training Data (X):", X.shape)
print("Training Labels (y):", y.shape)
print("Test Data (Xtest):", Xtest.shape)
print("Test Labels (ytest):", ytest.shape)


if sys.version_info.major == 2:
    X_diabetes, y_diabetes, Xtest_diabetes, ytest_diabetes = pickle.load(open('diabetes.pickle', 'rb'))
else:
    X_diabetes, y_diabetes, Xtest_diabetes, ytest_diabetes = pickle.load(open('diabetes.pickle', 'rb'), encoding='latin1')


print("\nDiabetes Dataset:")
print("Training Data (X):", X_diabetes.shape)
print("Training Labels (y):", y_diabetes.shape)
print("Test Data (Xtest):", Xtest_diabetes.shape)
print("Test Labels (ytest):", ytest_diabetes.shape)



means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))

means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))


x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()



# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')


X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))



# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
mses3_train = np.zeros((k, 1))
mses3 = np.zeros((k, 1))

ole_weights = learnOLERegression(X_i, y)
ridge_weights = []

for i, lambd in enumerate(lambdas):
    w_ridge = learnRidgeRegression(X_i, y, lambd)
    ridge_weights.append(w_ridge)
    mses3_train[i] = testOLERegression(w_ridge, X_i, y)
    mses3[i] = testOLERegression(w_ridge, Xtest_i, ytest)


plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses3_train, label="Train MSE")
plt.title("MSE for Train Data")
plt.xlabel("λ")
plt.ylabel("MSE")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lambdas, mses3, label="Test MSE")
plt.title("MSE for Test Data")
plt.xlabel("λ")
plt.ylabel("MSE")
plt.legend()

plt.tight_layout()
plt.show()


ridge_weights_optimal = ridge_weights[np.argmin(mses3)]
ole_weights_norm = np.linalg.norm(ole_weights)
ridge_weights_norm = np.linalg.norm(ridge_weights_optimal)

print(f"Optimal λ: {lambdas[np.argmin(mses3)]:.2f}")
print(f"Minimum Test MSE: {np.min(mses3):.4f}")
print(f"Magnitude of OLE Weights: {ole_weights_norm:.4f}")
print(f"Magnitude of Optimal Ridge Weights: {ridge_weights_norm:.4f}")




# Problem 4

k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k, 1))
mses4 = np.zeros((k, 1))
opts = {'maxiter': 20}
w_init = np.ones((X_i.shape[1],))

for lambd in lambdas:
    
    args = (X_i, y, lambd)
    result = minimize(regressionObjVal, w_init, jac=True, args=args, method='CG', options=opts)
    w_l = result.x.reshape(-1, 1)
    mses4_train[i] = testOLERegression(w_l, X_i, y)
    mses4[i] = testOLERegression(w_l, Xtest_i, ytest)
    i += 1
fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses4_train, label='Using scipy.minimize')
plt.plot(lambdas, mses3_train, label='Direct minimization (closed form)')
plt.title('MSE for Train Data')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lambdas, mses4, label='Using scipy.minimize')
plt.plot(lambdas, mses3, label='Direct minimization (closed form)')
plt.title('MSE for Test Data')
plt.legend()
plt.show()



# Problem 5

pmax = 7
lambda_opt = 0.06
mses5_train = np.zeros((pmax, 2))
mses5 = np.zeros((pmax, 2))

for p in range(pmax):
    Xd = mapNonLinear(X[:, 2:3], p)
    Xdtest = mapNonLinear(Xtest[:, 2:3], p)
    
    w_d1 = learnRidgeRegression(Xd, y, 0)
    mses5_train[p, 0] = testOLERegression(w_d1, Xd, y)
    mses5[p, 0] = testOLERegression(w_d1, Xdtest, ytest)
    
    w_d2 = learnRidgeRegression(Xd, y, lambda_opt)
    mses5_train[p, 1] = testOLERegression(w_d2, Xd, y)
    mses5[p, 1] = testOLERegression(w_d2, Xdtest, ytest)


fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax), mses5_train)
plt.title('MSE for Train Data')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.legend(('No Regularization', 'Regularization'))

plt.subplot(1, 2, 2)
plt.plot(range(pmax), mses5)
plt.title('MSE for Test Data')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.legend(('No Regularization', 'Regularization'))

plt.show()

