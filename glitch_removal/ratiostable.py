#
#
import numpy as np
import matplotlib.pyplot as plt
from .params import *

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

class rr010Table(object):
    """ Class to represent the ratios rr01/10 """

    def __init__(self):
        """ Initialise an instance of rr010Table """

    def create(self,l,n,freq,rr010,err,matcov):
        """ """
        self.l = l
        self.n = n
        self.freq = freq
        self.rr010 = rr010
        self.err = err
        self.matcov = matcov

    def load(self):
        """ Load the ratios rr01/10 from a file """

    # Methods for data handling

    def cut(self,sigma):
        """ Remove observational points using sigma-clipping """
        # clip data points with uncertainties above sigma * median uncertainty
        median_sigma = np.median(self.err)
        i_kept = np.argwhere(self.err <= sigma*median_sigma)[:,0]
        l_kept, n_kept, freq_kept, rr010_kept, err_kept = \
                self.l[i_kept], self.n[i_kept], self.freq[i_kept], \
                self.rr010[i_kept],self.err[i_kept]
        matcov_kept = self.matcov[i_kept,:][:,i_kept]
        # sort data points by increasing mode frequency
        i_sort = np.argsort(freq_kept)
        l_kept, n_kept, freq_kept, rr010_kept, err_kept = \
                l_kept[i_sort], n_kept[i_sort], freq_kept[i_sort], \
                rr010_kept[i_sort], err_kept[i_sort]
        matcov_kept = matcov_kept[i_sort,:][:,i_sort]
        self.cut_table = rr010Table()
        self.cut_table.create(l_kept,n_kept,freq_kept,rr010_kept,err_kept,matcov_kept)

    def cut_range(self,freqmin,freqmax):
        """ Keep only data points between freqmin and freqmax """
        i_kept = np.argwhere((self.freq >= freqmin) & (self.freq <= freqmax))[:,0]
        l_kept, n_kept, freq_kept, rr010_kept, err_kept = \
                self.l[i_kept], self.n[i_kept], self.freq[i_kept], \
                self.rr010[i_kept],self.err[i_kept]
        matcov_kept = self.matcov[i_kept,:][:,i_kept]
        # sort data points by increasing mode frequency
        i_sort = np.argsort(freq_kept)
        l_kept, n_kept, freq_kept, rr010_kept, err_kept = \
                l_kept[i_sort], n_kept[i_sort], freq_kept[i_sort], \
                rr010_kept[i_sort], err_kept[i_sort]
        matcov_kept = matcov_kept[i_sort,:][:,i_sort]
        self.cut_range_table = rr010Table()
        self.cut_range_table.create(l_kept,n_kept,freq_kept,rr010_kept,err_kept,matcov_kept)


    # Methods related to the covariance matrix

    def matcov_condnumber(self):
        """ Get the condition number of self.matcov """
        self.cond = np.linalg.cond(self.matcov)
        print(f"Condition number is {self.cond}")

    def matcov_invert(self):
        """ Invert the covariance matrix. """
        if fit_params["nsvd"] != -1:
            u,s,v = np.linalg.svd(self.matcov)
            self.inv_matcov = np.linalg.pinv(self.matcov,
                    rcond=s[fit_params["nsvd"]-1]/s.max())
        else:
            self.inv_matcov = np.linalg.inv(self.matcov)


    # Plotting methods

    def plot(self):
        """ Plot the ratios rr010 """
        plt.figure()
        plt.xlabel(r'Frequency ($\mu$Hz)')
        plt.ylabel(r'${\it {rr}_{010}}$')
        markers = ['o','^']
        for l in np.arange(2):
            i_l = self.l==l
            plt.errorbar(self.freq[i_l],self.rr010[i_l],yerr=self.err[i_l],
                    fmt=markers[l],mfc=colors[l],mec='none',ecolor='#c4c4c4',
                    label=r'$\ell = {}$'.format(l))
        plt.legend()
        plt.show()

    def plot_matcov(self):
        """ Plot the covariance matrix """
        plt.figure()
        plt.imshow(self.matcov)
        cb = plt.colorbar()
        plt.show()

    def plot_matcov_svd(self):
        """ Plot the singular values of self.matcov """
        u, s, v = np.linalg.svd(self.matcov)
        plt.figure()
        plt.plot(np.arange(1,len(s)+1),s,'o-')
        plt.show()





class r02Table(object):
    """ Class to represent the ratios r02 """

    def __init__(self):
        """ Initialise an instance of r02Table """

    def create(self,l,n,freq,r02,err,matcov):
        """ """
        self.l = l
        self.n = n
        self.freq = freq
        self.r02 = r02
        self.err = err
        self.matcov = matcov

    def load(self):
        """ Load the ratios r02 from a file """

    # Methods for data handling

    def cut(self,sigma):
        """ Remove observational points using sigma-clipping """
        median_sigma = np.median(self.err)
        i_kept = np.argwhere(self.err <= sigma*median_sigma)[:,0]
        l_kept, n_kept, freq_kept, r02_kept, err_kept = \
                self.l[i_kept], self.n[i_kept], self.freq[i_kept], \
                self.r02[i_kept],self.err[i_kept]
        matcov_kept = self.matcov[i_kept,:][:,i_kept]
        i_sort = np.argsort(freq_kept)
        l_kept, n_kept, freq_kept, r02_kept, err_kept = \
                l_kept[i_sort], n_kept[i_sort], freq_kept[i_sort], \
                r02_kept[i_sort], err_kept[i_sort]
        matcov_kept = matcov_kept[i_sort,:][:,i_sort]
        self.cut_table = r02Table()
        self.cut_table.create(l_kept,n_kept,freq_kept,r02_kept,err_kept,matcov_kept)

    def cut_range(self,freqmin,freqmax):
        """ Keep only data points between freqmin and freqmax """
        i_kept = np.argwhere((self.freq >= freqmin) & (self.freq <= freqmax))[:,0]
        l_kept, n_kept, freq_kept, r02_kept, err_kept = \
                self.l[i_kept], self.n[i_kept], self.freq[i_kept], \
                self.r02[i_kept],self.err[i_kept]
        matcov_kept = self.matcov[i_kept,:][:,i_kept]
        # sort data points by increasing mode frequency
        i_sort = np.argsort(freq_kept)
        l_kept, n_kept, freq_kept, r02_kept, err_kept = \
                l_kept[i_sort], n_kept[i_sort], freq_kept[i_sort], \
                r02_kept[i_sort], err_kept[i_sort]
        matcov_kept = matcov_kept[i_sort,:][:,i_sort]
        self.cut_range_table = r02Table()
        self.cut_range_table.create(l_kept,n_kept,freq_kept,r02_kept,err_kept,matcov_kept)

    # Methods related to the covariance matrix

    def matcov_condnumber(self):
        """ Get the condition number of self.matcov """
        self.cond = np.linalg.cond(self.matcov)
        print(f"Condition number is {self.cond}")

    def matcov_invert(self):
        """ Invert the covariance matrix. """
        if fit_params["nsvd"] != -1:
            u,s,v = np.linalg.svd(self.matcov)
            self.inv_matcov = np.linalg.pinv(self.matcov,
                    rcond=s[fit_params["nsvd"]-1]/s.max())
        else:
            self.inv_matcov = np.linalg.inv(self.matcov)

    # Plotting methods

    def plot(self):
        """ Plot the ratios r02 """
        plt.figure()
        plt.xlabel(r'Frequency ($\mu$Hz)')
        plt.ylabel(r'${\it r_{02}}$')
        plt.errorbar(self.freq,self.r02,yerr=self.err,fmt='s',mfc=colors[2],
                mec='none',ecolor='#c4c4c4')
        plt.show()

    def plot_matcov_svd(self):
        """ Plot the singular values of self.matcov """
        u, s, v = np.linalg.svd(self.matcov)
        plt.figure()
        plt.plot(np.arange(1,len(s)+1),s,'o-')
        plt.show()


    def plot_matcov(self):
        """ Plot the covariance matrix """
        plt.figure()
        plt.imshow(self.matcov)
        plt.xticks(ticks=np.arange(len(self.n)),labels=self.n.astype(int))
        plt.yticks(ticks=np.arange(len(self.n)),labels=self.n.astype(int))
        plt.xlabel(r'$\it r_{02} (n)$')
        plt.ylabel(r'$\it r_{02} (n)$')
        cb = plt.colorbar()
        plt.show()
