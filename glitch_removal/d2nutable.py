#
#
import numpy as np
import matplotlib.pyplot as plt
from .params import *


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

class D2nuTable(object):
    """ Class to represent the second frequency differences D2nu """

    def __init__(self):
        """ Initialise an instance of D2nuTable """

    def create(self,l,n,freq,d2nu,err,matcov):
        """ """
        self.l = l
        self.n = n
        self.freq = freq
        self.d2nu = d2nu
        self.err = err
        self.matcov = matcov

    def load(self):
        """ Load the second differences D2nu from a file """

    # Methods for data handling

    def cut(self,sigma):
        """ Remove observational points using sigma-clipping """
        median_sigma = np.median(self.err)
        i_kept = np.argwhere(self.err <= sigma*median_sigma)[:,0]
        l_kept, n_kept, freq_kept, d2nu_kept, err_kept = \
                self.l[i_kept], self.n[i_kept], self.freq[i_kept],\
                self.d2nu[i_kept],self.err[i_kept]
        matcov_kept = self.matcov[i_kept,:][:,i_kept]
        i_sort = np.argsort(freq_kept)
        l_kept, n_kept, freq_kept, rr010_kept, err_kept = \
                l_kept[i_sort], n_kept[i_sort], freq_kept[i_sort],\
                d2nu_kept[i_sort], err_kept[i_sort]
        matcov_kept = matcov_kept[i_sort,:][:,i_sort]
        self.cut_table = D2nuTable()
        self.cut_table.create(l_kept,n_kept,freq_kept,d2nu_kept,err_kept,matcov_kept)

    def cut_range(self,freqmin,freqmax):
        """ Keep only data points between freqmin and freqmax """
        i_kept = np.argwhere((self.freq >= freqmin) & (self.freq <= freqmax))[:,0]
        l_kept, n_kept, freq_kept, d2nu_kept, err_kept = \
                self.l[i_kept], self.n[i_kept], self.freq[i_kept], \
                self.d2nu[i_kept],self.err[i_kept]
        matcov_kept = self.matcov[i_kept,:][:,i_kept]
        # sort data points by increasing mode frequency
        i_sort = np.argsort(freq_kept)
        l_kept, n_kept, freq_kept, d2nu_kept, err_kept = \
                l_kept[i_sort], n_kept[i_sort], freq_kept[i_sort], \
                d2nu_kept[i_sort], err_kept[i_sort]
        matcov_kept = matcov_kept[i_sort,:][:,i_sort]
        self.cut_range_table = D2nuTable()
        self.cut_range_table.create(l_kept,n_kept,freq_kept,d2nu_kept,err_kept,matcov_kept)

    def cut_l(self,l):
        """ Keep only data points that have their angular degree != l. """
        i_kept = np.argwhere(self.l != l)[:,0]
        l_kept, n_kept, freq_kept, d2nu_kept, err_kept = \
                self.l[i_kept], self.n[i_kept], self.freq[i_kept], \
                self.d2nu[i_kept],self.err[i_kept]
        matcov_kept = self.matcov[i_kept,:][:,i_kept]
    # Methods related to the covariance matrix
        # sort data points by increasing mode frequency
        i_sort = np.argsort(freq_kept)
        l_kept, n_kept, freq_kept, d2nu_kept, err_kept = \
                l_kept[i_sort], n_kept[i_sort], freq_kept[i_sort], \
                d2nu_kept[i_sort], err_kept[i_sort]
        matcov_kept = matcov_kept[i_sort,:][:,i_sort]
        self.cut_l_table = D2nuTable()
        self.cut_l_table.create(l_kept,n_kept,freq_kept,d2nu_kept,err_kept,matcov_kept)

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
        """ Plot the second differences D2nu """
        plt.figure()
        plt.xlabel(r'Frequency ($\mu$Hz)')
        plt.ylabel(r'$\Delta_2 \nu$ $(\mu Hz)$')
        markers = ['o','^','s']
        for l in np.arange(3):
            i_l = self.l == l
            plt.errorbar(self.freq[i_l],self.d2nu[i_l],yerr=self.err[i_l],
                    fmt=markers[l],mfc=colors[l],mec='none',ecolor='#c4c4c4',
                    label=r'$\ell = {}$'.format(l))
        plt.legend()
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
        cb = plt.colorbar()
        plt.show()
