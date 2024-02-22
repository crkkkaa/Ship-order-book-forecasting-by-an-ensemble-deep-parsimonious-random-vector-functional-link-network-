# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from math import sqrt
from scipy.special import expit
from sklearn import metrics

class SCN(object):
    name = 'Stochastic Configuration Networks'
    version = '1.0 beta'
    L = 0  
    W = []  
    b = []  
    Beta = []  
    r = np.array([0.9, 0.99, 0.999, 0.9999, 0.99999, 0.99999])
    tol = 1e-4  
    Lambdas = np.array([0.5, 1, 3, 5, 7, 9, 15, 25, 50, 100, 150, 200])
    L_max = 100  
    T_max = 100
    nB = 1  
    verbose = 50  
    COST = 0 
    def __init__(self,
                 L_max=100,
                 T_max=100,
                 tol=1e-4,
                 Lambdas=[0.5, 1, 3, 5, 7, 9, 15, 25, 50, 100, 150, 200],
                 r=[0.9, 0.99, 0.999, 0.9999, 0.99999, 0.99999],
                 nB=1,
                 verbose=50
                 ):
        if isinstance(verbose, int):
            self.verbose = verbose
        if isinstance(L_max,  int):
            self.L_max = L_max
            if L_max > 5000:
                self.verbose = 500  
        if isinstance(T_max, int):
            self.T_max = T_max
        if isinstance(tol, float):
            self.tol = tol
        if isinstance(Lambdas, list):
            self.Lambdas = np.array(Lambdas)
        if isinstance(r, list):
            self.r = np.array(r)
        if isinstance(nB, int):
            self.nB = nB
        
    def printProperties(self):
        print('   Name:{}'.format(self.name))
        print('version:{}'.format(self.version))
        print('      L:{}'.format(self.L))
        print('      W:{}'.format(self.W.shape))
        print('      b:{}'.format(self.b.shape))
        print('   Beta:{}'.format(self.Beta.shape))
        print('      r:{}'.format(self.r))
        print('    tol:{}'.format(self.tol))
        print('Lambdas:{}'.format(self.Lambdas))
        print('  L_max:{}'.format(self.L_max))
        print('  T_max:{}'.format(self.T_max))
        print('     nB:{}'.format(self.nB))
        print('verbose:{}'.format(self.verbose))
        print('   COST:{}'.format(self.COST))
    def inequalityEq(self, eq, gk, r_L):
        ksi = ((eq.conj().T @ gk)**2) / (gk.conj().T @ gk) - \
            (1 - r_L) * (eq.conj().T @ eq)
        return ksi
    def sc_Search(self, X, E0):
        Flag = 0
        WB = []
        bB = []
        d = X.shape[1]
        m = E0.shape[1]
        C = []
        for Lambda in self.Lambdas:         
            WT = Lambda * (2 * np.random.rand(d, self.T_max) - 1)
            bT = Lambda * (2 * np.random.rand(1, self.T_max) - 1)
            HT = expit(X@WT + bT)
            for r_L in self.r:
                for t in range(0, self.T_max):
                    H_t = HT[:, t]
                    ksi_m = np.zeros((1, m), dtype=np.float64)
                    for i_m in range(0, m):
                        eq = E0[:, i_m].reshape(-1, 1)
                        gk = H_t.reshape(-1, 1)
                        ksi_m[0, i_m] = self.inequalityEq(eq, gk, r_L)
                    Ksi_t = np.sum(ksi_m, 0).reshape(-1, 1)
                    if np.min(ksi_m) > 0:
                        if type(C) == list:
                            C = Ksi_t
                        else:
                            C = np.concatenate([C, Ksi_t], axis=1)
                        if type(WB) == list:
                            WB = WT[:, t].reshape(-1, 1)
                        else:
                            WB = np.concatenate(
                                (WB, WT[:, t].reshape(-1, 1)), axis=1)
                        if type(bB) == list:
                            bB = bT[:, t].reshape(-1, 1)
                        else:
                            bB = np.concatenate(
                                (bB, bT[:, t].reshape(-1, 1)), axis=1)
                nC = len(C)
                if nC >= self.nB:
                    break  
                else:
                    continue
            # end r
            if nC >= self.nB:
                break  
            else:
                continue
        if nC >= self.nB:
            I = C.argsort(axis=1)[::-1]
            I_nb = I[0, 0:self.nB]
            WB = WB[:, I_nb]
            bB = bB[:, I_nb]
        if nC == 0 or nC < self.nB:
            Flag = 1
        return [WB, bB, Flag]

    def addNodes(self, w_L, b_L):
        if type(self.W) == list:
            self.W = w_L
        else:
            self.W = np.concatenate((self.W, w_L), axis=1)

        if type(self.b) == list:
            self.b = b_L
        else:
            self.b = np.concatenate((self.b, b_L), axis=1)

        self.L = self.L + 1

    def computeBeta(self, H, T):
        Beta = np.linalg.pinv(H) @ T
        self.Beta = Beta

    def getH(self, X):
        H = self.activationFun(X)
        return H

    def activationFun(self,  X):
        H = expit(X@self.W + self.b)
        return H

    def RMSE(self, E0):
        EN = E0.shape[0]
        Error = sqrt(np.sum(np.sum(E0**2, axis=0)/EN, axis=0))
        return Error

    def upgradeSCN(self, X, T):
        H = self.getH(X)
        self.computeBeta(H, T)
        O = H @self.Beta
        E = T - O
        Error = self.RMSE(E)
        self.COST = Error
        return (O, E, Error)
    def getOutput(self, X):
        H = self.getH(X)
        O = H @ self.Beta
        return O

    def regression(self, X, T):
        E = T
        ErrorList = []
        Error = self.RMSE(E)
        while (self.L < self.L_max) and (Error > self.tol):
            if self.L % self.verbose == 0:
                print('#L:{}\t RMSE:{:.4f} \r'.format(self.L, Error))
            (w_L, b_L, Flag) = self.sc_Search(X, E)
            if Flag == 1:
                break
            self.addNodes(w_L, b_L)
            (otemp, E, Error) = self.upgradeSCN(X, T)
            # log
            if type(ErrorList) == list:
                ErrorList = np.array(Error).reshape(1, 1)
            else:
                ErrorList = np.concatenate(
                    [np.array(ErrorList), np.matlib.repmat(Error, 1, self.nB)], axis=1)
        print('End Searching ...')
        print('#L:{}\t RMSE:{:.4f} \r'.format(self.L, Error))
        print('***************************************')
        self.printProperties()
        return ErrorList

    def getLabel(self, X):
        O = self.getOutput(X)
        (N, p) = O.shape
        ON = np.zeros((N, p))
        ind = np.argmax(O, axis=1)
        if p > 1:
            for i in range(0, N):
                ON[i, ind[i]] = 1
        else:
            for i in range(0, N):
                if O(i) > 0.50:
                    ON[i] = 1
        return ON

    def getAccuracy(self, X, T):
        O = self.getLabel(X)
        rate = metrics.confusion_matrix(T.argmax(axis=1), O.argmax(axis=1))
        return (O, rate)

    def classification(self, X, T):
        E = T
        ErrorList = []
        RateList = []
        Error = self.RMSE(E)
        rate = 0
        while (self.L < self.L_max) and (Error > self.tol):
            if self.L % self.verbose == 0:
                print('#L:{}\tRMSE:{:.4f} \tACC:{:.4f}\r'.format(
                    self.L, Error, rate))
            (w_L, b_L, Flag) = self.sc_Search(X, E)
            if Flag == 1:
                break
            self.addNodes(w_L, b_L)
            (otemp, E, Error) = self.upgradeSCN(X, T)
            O = self.getLabel(X)
            rate = metrics.accuracy_score(T, O)
            if type(ErrorList) == list:
                ErrorList = np.array(Error, dtype=np.float64).reshape(1, 1)
            else:
                ErrorList = np.concatenate([np.array(
                    ErrorList, dtype=np.float64), np.matlib.repmat(Error, 1, self.nB)], axis=1)
                
            if type(RateList) == list:
                RateList = np.array(rate, dtype=np.float64).reshape(1, 1)
            else:
                RateList = np.concatenate([np.array(
                    RateList, dtype=np.float64), np.matlib.repmat(rate, 1, self.nB)], axis=1)

        print('End Searching ...')
        print('#L:{}\tRMSE:{:.4f} \tACC:{:.4f}\r'.format(self.L, Error, rate))
        print('***************************************')
        self.printProperties()
        return ErrorList, RateList
