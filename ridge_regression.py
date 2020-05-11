import numpy as np 
import random

class ridge_regression:
    def normalize(self,X):
        """
        Normalize data and add add vector ones
        -----
        Input: 
        X: array-like of shape (n_samples,n_feature)
        -----
        Return:
        X_new: array-like of shape (n_samples,n_feature+1)
        """
        x_max=np.max(X)
        x_min=np.min(X)
        m,n=np.shape(X)
        self.m=m
        self.n=n
        x_minus=x_max-x_min
        X_new=[]
        # Normalize data
        for x in X:
            tmp=(x-x_min)/x_minus
            X_new.append(tmp)
        # add one vector
        x_ones=np.ones((m,1))
        return np.hstack((x_ones,X_new))
    
    def grad(self,X,Y,lamda,w):
        """
        compute gradient
        -----
        Input:
        X: array-like of shape(n_samples,n_feature+1)
        Y: label
        lamda: float
        w: array-like of shape(1,n_feature+1)
        -----
        Return: 
        vector Gradient
        """
        return X.T.dot(X.dot(w)-Y)+lamda*w

    def compute_RSS(self,X,Y,w):
        """
        compute loss function
        -----
        Input:
        X: array-like of shape(n_samples,n_feature+1)
        Y: label
        w: array-like of shape(1,n_feature+1)
        -----
        Return: 
        loss values
        """
        loss= 0.5 /X.shape[0]*np.sum((X.dot(w)-Y)**2)
        return loss
    
    def find_w(self,X,Y,lamda):
        """
        compute w
        """
        X_new=np.array(X)
        Y_new=np.array(Y)
        return np.linalg.inv(X_new.transpose().dot(X_new)+lamda*np.eye(X_new.shape[1])).dot(X_new.transpose()).dot(Y_new)
    
    def find_w_GD(self,lamda,learning_rate,X_train,Y_train,max_epoches=100,batch_size=100,tol=1e-4):
        """ 
        find w with mini-batch GD
        """
        X_train=np.array(X_train)
        Y_train=np.array(Y_train)
        #init w
        w=np.random.randn(X_train.shape[1])
        w_old=np.copy(w)
        n_batch=int(np.ceil(X_train.shape[0]/batch_size))
        for it in range(max_epoches):
            rd=random.sample(range(X_train.shape[0]),X_train.shape[0])
            X_train=X_train[rd]
            Y_train=Y_train[rd]
            for ibatch in range(n_batch):
                X_sub=X_train[ibatch:ibatch+batch_size]
                Y_sub=Y_train[ibatch:ibatch+batch_size]
                w=w-learning_rate*self.grad(X_sub,Y_sub,lamda,w)
                if(np.linalg.norm(w-w_old)<tol):
                    return w
                w_old=np.copy(w)
        return w

    def fit(self,X_train,Y_train):
        def find_best_lamda(best_lamda,min_RSS,list_lamda):
            for lamda in list_lamda:
                current_loss=cross_validation(5,lamda)
                if(current_loss<min_RSS):
                    min_RSS=current_loss
                    best_lamda=lamda
            return best_lamda,min_RSS

        def cross_validation(n_fold,lamda):
            size_fold=int(X_new.shape[0]/n_fold)
            loss=0
            for i in range(n_fold):
                X_validation=X_new[i*size_fold:i*size_fold+size_fold]
                Y_validation=Y_train[i*size_fold:i*size_fold+size_fold]
                X_train_sub=[]
                for k in range(X_new.shape[0]):
                    if(k<i*size_fold or i*size_fold+size_fold<=k):
                        X_train_sub.append(X_new[k])
                
                Y_train_sub= [ y for y in Y_train if y not in Y_validation]
                w=self.find_w(X_train_sub,Y_train_sub,lamda)
                #w=self.find_w_GD(lamda,0.001,X_train_sub,Y_train_sub)
                loss+=self.compute_RSS(X_validation,Y_validation,w)
            return loss/n_fold

        X_new=self.normalize(X_train)
        best_LAMDA,min_RSS=find_best_lamda(10,10000**2,list(range(10)))
        list_lamdas= [ k*0.001 for k in range(max(0,best_LAMDA-1),(best_LAMDA+1)*1000)]
        best_LAMDA,min_RSS=find_best_lamda(best_lamda=list_lamdas[0],min_RSS=min_RSS,list_lamda=list_lamdas)
        
        self.best_lamda=best_LAMDA
        self.w=self.find_w(X_new,Y_train,best_LAMDA)

    def predict(self,X):
        X=self.normalize(X)
        list_pre=[]
        for x in X:
            pre=x.dot(self.w)
            list_pre.append(pre)
        return list_pre

def load_data(pathin):
    with open(pathin,'r') as f:
        data=f.read().splitlines()
    X=[]
    Y=[]
    for doc in data:
        tmp=doc.split()
        X.append(tmp[:15])
        Y.append(tmp[-1])
    return X,Y

pathin="/home/lnq/Desktop/20192/code_lab/Ridge Regression/datasets"
X,Y=load_data(pathin)
X=np.array(X,dtype=np.float64)
Y=np.array(Y,dtype=np.float64)
X_train=np.array(X[:50])
Y_train=np.array(Y[:50])
X_test=np.array(X[50:])
Y_test=np.array(Y[50:])
model=ridge_regression()
model.fit(X_train,Y_train)
pre=model.predict(X_test)
print(model.best_lamda)
print(np.sum(0.5*(pre-Y_test)**2))