#
#
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from .d2nutable import *
from .ratiostable import *
from .params import *

class FreqTable(object):
    """ Representing a table of frequencies """

    def __init__(self,path,ufreqs=u.microHertz,coll=0,coln=1,colfreq=2,colerr=3):
        """ Initialise an instance of FreqTable """
        self.path = path    # path to the frequency data
        self.coll = coll    # column degree l
        self.coln = coln    # column radial order n
        self.colfreq = colfreq    # column frequencies
        self.colerr = colerr    # column frequency errors

    def load(self):
        """ Load frequency data from self.path """
        datafreq = np.genfromtxt(self.path)
        self.l = datafreq[:,self.coll]
        self.n = datafreq[:,self.coln]
        self.freq = datafreq[:,self.colfreq]
        if self.colerr != -1:
            self.freqerr = datafreq[:,self.colerr]

    def calc_d2nu(self):
        """ Compute seconde differences from frequency data """
        # Get frequencies for each degree l
        f0 = self.freq[self.l == 0]
        f1 = self.freq[self.l == 1]
        f2 = self.freq[self.l == 2]
        # Limits to computable d2nu in radial orders
        n0 = self.n[self.l == 0]
        n1 = self.n[self.l == 1]
        n2 = self.n[self.l == 2]
        min0 = n0[1]
        max0 = n0[-2]
        min1 = n1[1]
        max1 = n1[-2]
        min2 = n2[1]
        max2 = n2[-2]
        # Check if there is no missing order (True = OK)
        alln0 = np.allclose(n0[1:]-n0[:-1],[1]*(len(n0)-1))
        alln1 = np.allclose(n1[1:]-n1[:-1],[1]*(len(n1)-1))
        alln2 = np.allclose(n2[1:]-n2[:-1],[1]*(len(n2)-1))
        if alln0 == True and alln1 == True and alln2 == True:
            # Compute second differences
            # l = 0
            norder0 = self.n[self.l == 0][1:-1]
            freq0 = self.freq[self.l==0][1:-1]
            d2nu0 = f0[:-2] - 2*f0[1:-1] + f0[2:]
            # l = 1
            norder1 = self.n[self.l == 1][1:-1]
            freq1 = self.freq[self.l == 1][1:-1]
            d2nu1 = f1[:-2] - 2*f1[1:-1] + f1[2:]
            # l = 2
            norder2 = self.n[self.l == 2][1:-1]
            freq2 = self.freq[self.l == 2][1:-1]
            d2nu2 = f2[:-2] - 2*f2[1:-1] + f2[2:]
            # Compute the elements of the covariance matrix
            # Uncomment the commented lines to test if the cov. matrix is correctly
            # computed. You should obtain 1, -4, 6, -4, 1 on one given line/col.
            # l = 0
            err0 = self.freqerr[self.l == 0]
            # err0 = np.ones(len(f0))
            cov0 = err0[:-2]**2+4*err0[1:-1]**2+err0[2:]**2
            cov0p1 = -2*err0[1:-2]**2 - 2*err0[2:-1]**2
            cov0p2 = err0[2:-2]**2
            # l = 1
            err1 = self.freqerr[self.l == 1]
            # err1 = np.ones(len(f1))
            cov1 = err1[:-2]**2+4*err1[1:-1]**2+err1[2:]**2
            cov1p1 = -2*err1[1:-2]**2 - 2*err1[2:-1]**2
            cov1p2 = err1[2:-2]**2
            # l = 2
            err2 = self.freqerr[self.l == 2]
            # err2 = np.ones(len(f2))
            cov2 = err2[:-2]**2+4*err2[1:-1]**2+err2[2:]**2
            cov2p1 = -2*err2[1:-2]**2 - 2*err2[2:-1]**2
            cov2p2 = err2[2:-2]**2
            # Build the cov. matrix itself
            # under the form of 9 blocks for each degree l and their cross
            # coefficients with other l :
            # |d2nu0   | d2nu0-1 | d2nu0-2|
            # |d2nu1-0 | d2nu1   | d2nu1-2|
            # |d2nu2-0 | d2nu2-1 | d2nu2  |
            N = len(d2nu0) + len(d2nu1) + len(d2nu2)
            N0 = len(d2nu0)
            N1 = len(d2nu1)
            N2 = len(d2nu2)
            matcov = np.zeros((N,N))
            # indices for scanning the elements of the matrix
            # initialisation l = 0
            ip_2 = 0
            ip_1 = 0
            ip0 = 0
            ip1 = 0
            ip2 = 0
            for i in np.arange(N0):
                if i != 0 and i != 1 and i != N0-2 and i != N0-1:
                    matcov[i,i-2] = cov0p2[ip_2]
                    matcov[i,i-1] = cov0p1[ip_1]
                    matcov[i,i] = cov0[ip0]
                    matcov[i,i+1] = cov0p1[ip1]
                    matcov[i,i+2] = cov0p2[ip2]
                    ip_2,ip_1,ip0,ip1,ip2 = ip_2+1,ip_1+1,ip0+1,ip1+1,ip2+1
                elif i == 0:
                    matcov[i,i] = cov0[ip0]
                    matcov[i,i+1] = cov0p1[ip1]
                    matcov[i,i+2] = cov0p2[ip2]
                    ip0,ip1,ip2 = ip0+1,ip1+1,ip2+1
                elif i == 1:
                    matcov[i,i-1] = cov0p1[ip_1]
                    matcov[i,i] = cov0[ip0]
                    matcov[i,i+1] = cov0p1[ip1]
                    matcov[i,i+2] = cov0p2[ip2]
                    ip_1,ip0,ip1,ip2 = ip_1+1,ip0+1,ip1+1,ip2+1
                elif i == N0-2:
                    matcov[i,i-2] = cov0p2[ip_2]
                    matcov[i,i-1] = cov0p1[ip_1]
                    matcov[i,i] = cov0[ip0]
                    matcov[i,i+1] = cov0p1[ip1]
                    ip_2,ip_1,ip0,ip1 = ip_2+1,ip_1+1,ip0+1,ip1+1
                elif i == N0-1:
                    matcov[i,i-2] = cov0p2[ip_2]
                    matcov[i,i-1] = cov0p1[ip_1]
                    matcov[i,i] = cov0[ip0]
                    ip_2,ip_1,ip0 = ip_2+1,ip_1+1,ip0+1
            # initialisation l = 1
            ip_2 = 0
            ip_1 = 0
            ip0 = 0
            ip1 = 0
            ip2 = 0
            for i in np.arange(N0,N0+N1):
                if i != N0 and i != N0+1 and i != N0+N1-2 and i != N0+N1-1:
                    matcov[i,i-2] = cov1p2[ip_2]
                    matcov[i,i-1] = cov1p1[ip_1]
                    matcov[i,i] = cov1[ip0]
                    matcov[i,i+1] = cov1p1[ip1]
                    matcov[i,i+2] = cov1p2[ip2]
                    ip_2,ip_1,ip0,ip1,ip2 = ip_2+1,ip_1+1,ip0+1,ip1+1,ip2+1
                elif i == N0:
                    matcov[i,i] = cov1[ip0]
                    matcov[i,i+1] = cov1p1[ip1]
                    matcov[i,i+2] = cov1p2[ip2]
                    ip0,ip1,ip2 = ip0+1,ip1+1,ip2+1
                elif i == N0+1:
                    matcov[i,i-1] = cov1p1[ip_1]
                    matcov[i,i] = cov1[ip0]
                    matcov[i,i+1] = cov1p1[ip1]
                    matcov[i,i+2] = cov1p2[ip2]
                    ip_1,ip0,ip1,ip2 = ip_1+1,ip0+1,ip1+1,ip2+1
                elif i == N0+N1-2:
                    matcov[i,i-2] = cov1p2[ip_2]
                    matcov[i,i-1] = cov1p1[ip_1]
                    matcov[i,i] = cov1[ip0]
                    matcov[i,i+1] = cov1p1[ip1]
                    ip_2,ip_1,ip0,ip1 = ip_2+1,ip_1+1,ip0+1,ip1+1
                elif i == N0+N1-1:
                    matcov[i,i-2] = cov1p2[ip_2]
                    matcov[i,i-1] = cov1p1[ip_1]
                    matcov[i,i] = cov1[ip0]
                    ip_2,ip_1,ip0 = ip_2+1,ip_1+1,ip0+1
            # initialisation l = 2
            ip_2 = 0
            ip_1 = 0
            ip0 = 0
            ip1 = 0
            ip2 = 0
            for i in np.arange(N0+N1,N0+N1+N2):
                if i != N0+N1 and i != N0+N1+1 and i != N0+N1+N2-2 and i != N0+N1+N2-1:
                    matcov[i,i-2] = cov2p2[ip_2]
                    matcov[i,i-1] = cov2p1[ip_1]
                    matcov[i,i] = cov2[ip0]
                    matcov[i,i+1] = cov2p1[ip1]
                    matcov[i,i+2] = cov2p2[ip2]
                    ip_2,ip_1,ip0,ip1,ip2 = ip_2+1,ip_1+1,ip0+1,ip1+1,ip2+1
                elif i == N0+N1:
                    matcov[i,i] = cov2[ip0]
                    matcov[i,i+1] = cov2p1[ip1]
                    matcov[i,i+2] = cov2p2[ip2]
                    ip0,ip1,ip2 = ip0+1,ip1+1,ip2+1
                elif i == N0+N1+1:
                    matcov[i,i-1] = cov2p1[ip_1]
                    matcov[i,i] = cov2[ip0]
                    matcov[i,i+1] = cov2p1[ip1]
                    matcov[i,i+2] = cov2p2[ip2]
                    ip_1,ip0,ip1,ip2 = ip_1+1,ip0+1,ip1+1,ip2+1
                elif i == N0+N1+N2-2:
                    matcov[i,i-2] = cov2p2[ip_2]
                    matcov[i,i-1] = cov2p1[ip_1]
                    matcov[i,i] = cov2[ip0]
                    matcov[i,i+1] = cov2p1[ip1]
                    ip_2,ip_1,ip0,ip1 = ip_2+1,ip_1+1,ip0+1,ip1+1
                elif i == N0+N1+N2-1:
                    matcov[i,i-2] = cov2p2[ip_2]
                    matcov[i,i-1] = cov2p1[ip_1]
                    matcov[i,i] = cov2[ip0]
                    ip_2,ip_1,ip0 = ip_2+1,ip_1+1,ip0+1
            # Concatenate data computed for each degree l
            norder = np.concatenate((norder0,norder1,norder2))
            degree = np.concatenate((np.zeros(N0),np.ones(N1),2*np.ones(N2)))
            freq = np.concatenate((freq0,freq1,freq2))
            d2nu = np.concatenate((d2nu0,d2nu1,d2nu2))
            err = np.concatenate((np.sqrt(cov0),np.sqrt(cov1),np.sqrt(cov2)))
            # Create an instance of D2nuTable to store the results
            self.d2nutable = D2nuTable()
            self.d2nutable.create(degree,norder,freq,d2nu,err,matcov)


    def calc_ratios(self):
        """ Compute ratios rr01/10 and r02 from frequency data """
        nb_draws = 1000 # nb of draws for estimating the cov. matrix
        # Get frequencies for each degree l
        f0 = self.freq[self.l == 0]
        f1 = self.freq[self.l == 1]
        f2 = self.freq[self.l == 2]
        # Limits to computable rr01/10 and r02 in radial orders
        n0 = self.n[self.l == 0]
        n1 = self.n[self.l == 1]
        n2 = self.n[self.l == 2]
        min01 = max(min(n0)+1,min(n1)+1)
        max01 = min(max(n0)-1,max(n1))
        min10 = max(min(n1)+1,min(n0))
        max10 = min(max(n1)-1,max(n0)-1)
        min02 = max(min(n2)+1,min(n1)+1,min(n0))
        max02 = min(max(n2)+1,max(n1),max(n0))
        norder = np.array([min01,max01,min10,max10,min02,max02])
        # Check if there is no missing order (True = OK)
        alln0 = np.allclose(n0[1:]-n0[:-1],[1]*(len(n0)-1))
        alln1 = np.allclose(n1[1:]-n1[:-1],[1]*(len(n1)-1))
        alln2 = np.allclose(n2[1:]-n2[:-1],[1]*(len(n2)-1))
        # Get radial orders at which ratios are computable
        n01 = np.arange(min01,max01+1)
        n10 = np.arange(min10,max10+1)
        n02 = np.arange(min02,max02+1)
        # Get frequencies at which ratios are computable
        freq01 = f0[(n0 >= min01) & (n0 <= max01)]
        freq10 = f1[(n1 >= min10) & (n1 <= max10)]
        freq02 = f0[(n0 >= min02) & (n0 <= max02)]
        # Compute ratios from frequency data
        if alln0 == True and alln1 == True:
            #rr01
            i0_01 = (n0 >= min01-1) & (n0 <= max01+1)
            i1_01 = (n1 >= min01-1) & (n1 <= max01)
            t0 = f0[i0_01]
            t1 = f1[i1_01]
            m = len(n01)
            rr01 = ((1/8.0) * (t0[:m] - 4*t1[:m] + 6*t0[1:m+1] - 4*t1[1:m+1] +
                    t0[2:]))/(t1[1:m+1]-t1[:m])
            #rr10
            i0_10 = (n0 >= min10) & (n0 <= max10+1)
            i1_10 = (n1 >= min10-1) & (n1 <= max10+1)
            tt0 = f0[i0_10]
            tt1 = f1[i1_10]
            m = len(n10)
            rr10 = ((-1/8.0) * (tt1[:m] - 4*tt0[:m] + 6*tt1[1:m+1] - 4*tt0[1:] +
                    tt1[2:]))/(tt0[1:]-tt0[:m])
            if alln2 == True:
                #r02
                i0_02 = (n0 >= min02) & (n0 <= max02)
                i1_02 = (n1 >= min02-1) & (n1 <= max02)
                i2_02 = (n2 >= min02-1) & (n2 <= max02-1)
                t0 = f0[i0_02]
                t1 = f1[i1_02]
                t2 = f2[i2_02]
                m = len(n02)
                r02 = (t0-t2)/(t1[1:]-t1[:m])
            # MC simulation for computing the cov. matrix
            for k in np.arange(nb_draws):
                #rr01
                t0 = f0[i0_01]
                t1 = f1[i1_01]
                c0 = t0
                c1 = t1
                t0 = np.random.normal(t0,self.freqerr[self.l==0][i0_01])
                t1 = np.random.normal(t1,self.freqerr[self.l==1][i1_01])
                m = len(n01)
                rr01_mc = ((1/8.0) * (t0[:m] - 4*t1[:m] + 6*t0[1:m+1] - 4*t1[1:m+1] +
                        t0[2:]))/(t1[1:m+1]-t1[:m])
                #rr10
                tt0 = f0[i0_10]
                tt1 = f1[i1_10]
                t0_temp = t0
                if min10 < (min01-1):
                    add_top = np.random.normal(tt0[0],self.freqerr[self.l==0][i0_10][0])
                    t0_temp = np.insert(t0_temp,0,add_top)
                if min10 > (min01-1):
                    t0_temp = t0_temp[1:]
                if (max10+1) > (max01+1):
                    add_btm = np.random.normal(tt0[-1],self.freqerr[self.l==0][i0_10][-1])
                    t0_temp = np.append(t0_temp,add_btm)
                if (max10+1) < (max01+1):
                    t0_temp = t0_temp[:-1]
                t0 = t0_temp

                t1_temp = t1
                if (min10-1) < (min01-1):
                    add_top = np.random.normal(tt1[0],self.freqerr[self.l==1][i1_10][0])
                    t1_temp = np.insert(t1_temp,0,add_top)
                if (min10-1) > (min01-1):
                    t1_temp = t1_temp[1:]
                if (max10+1) > max01:
                    add_btm = np.random.normal(tt1[-1],self.freqerr[self.l==1][i1_10][-1])
                    t1_temp = np.append(t1_temp,add_btm)
                if (max10+1) < max01:
                    t1_temp = t1_temp[:-1]
                t1 = t1_temp

                m = len(n10)
                rr10_mc = ((-1/8.0) * (t1[:m] - 4*t0[:m] + 6*t1[1:m+1] - 4*t0[1:] +
                        t1[2:]))/(t0[1:]-t0[:m])
                if alln2 == True:
                    #r02
                    ttt0 = f0[i0_02]
                    ttt1 = f1[i1_02]
                    ttt2 = f2[i2_02]
                    ttt0 = np.random.normal(ttt0,self.freqerr[self.l==0][i0_02])
                    ttt1 = np.random.normal(ttt1,self.freqerr[self.l==1][i1_02])
                    ttt2 = np.random.normal(ttt2,self.freqerr[self.l==2][i2_02])
                    m = len(n02)
                    r02_mc = (ttt0-ttt2)/(ttt1[1:]-ttt1[:m])
                # Store the results of the MC simulation
                if k == 0:
                    cov01 = np.transpose(rr01_mc)
                    cov10 = np.transpose(rr10_mc)
                    if alln2 == True:
                        cov02 = np.transpose(r02_mc)
                else:
                    cov01 = np.vstack((cov01,rr01_mc))
                    cov10 = np.vstack((cov10,rr10_mc))
                    if alln2 == True:
                        cov02 = np.vstack((cov02,r02_mc))
            # Concatenate data computed for degree l = 0 and l = 1
            n010 = np.concatenate((n01,n10))
            l010 = np.concatenate((np.zeros(len(n01)),np.ones(len(n10))))
            freq010 = np.concatenate((freq01,freq10))
            rr010 = np.concatenate((rr01,rr10))
            cov010 = np.transpose(np.hstack((cov01,cov10)))
            matcov010 = np.cov(cov010)
            err010 = np.sqrt(np.diag(matcov010))
            self.rr010table = rr010Table()
            self.rr010table.create(l010,n010,freq010,rr010,err010,matcov010)
            if alln2 == True:
                l02 = np.zeros(len(n02))
                cov02 = np.transpose(cov02)
                matcov02 = np.cov(cov02)
                err02 = np.sqrt(np.diag(matcov02))
                self.r02table = r02Table()
                self.r02table.create(l02,n02,freq02,r02,err02,matcov02)
