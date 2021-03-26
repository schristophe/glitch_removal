#
#
import numpy as np
import matplotlib.pyplot as plt
from .params import *


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

class D2nuTable(object):
    """ Class to represent the second frequency differences D2nu.

    Attributes:
        l (np.array): Harmonic degrees.
        n (np.array): Radial orders.
        freq (np.array): Mode frequencies.
        d2nu (np.array): Second differences.
        err (np.array): Errors on second differences.
        matcov (np.array): Covariance matrix.
        cut_table (D2nuTable):
            The initial dataset of d2nu but data points
            with large uncertainties are excluded.
        cut_range_table (D2nuTable):
            The initial dataset of d2nu but only data points
            in a given frequency range are kept.
        cut_l_table (D2nuTable):
            The initial dataset of d2nu but data points which
            angular degree is l are excluded (usually l=2 because they have the
            largest uncertainties).
        cond (float): Condition number of the covariance matrix.
        inv_matcov (np.array): Inverse covariance matrix.
    """

    def __init__(self):
        """ Initialises an instance of D2nuTable. """

    def create(self, l, n, freq, d2nu, err, matcov):
        """ Creates an instance of D2nuTable by directly passing the data.

        Args:
            l (np.array): Angular degrees.
            n (np.array): Radial orders.
            freq (np.array): Mode frequencies.
            d2nu (np.array): Second differences.
            err (np.array): Errors on second differences.
            matcov (np.array): Covariance matrix.
        """
        self.l = l
        self.n = n
        self.freq = freq
        self.d2nu = d2nu
        self.err = err
        self.matcov = matcov

    def load(self):
        """ Loads the second differences D2nu from a file.

        Not implemented yet.
        """

    # Methods for data handling

    def cut(self, sigma):
        """ Removes observational points for which the D2nu uncertainty is
        above sigma * median(err).

        The resulting D2nuTable object is stored in a new attribute called
        'cut_table'.

        Args:
            sigma (float):
                Defines the maximum tolerated d2nu uncertainty in cut_table
                (i.e. sigma * median(err)).
        """
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

    def cut_range(self, freqmin, freqmax):
        """ Keeps data points between freqmin and freqmax.

        The resulting D2nuTable object is stored in a new attribute called
        'cut_range_table'.

        Args:
            freqmin (float): Minimum frequency.
            freqmax (float): Maximum frequency.
        """
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

    def cut_l(self, l):
        """ Keeps data points that have their angular degree != l.

        The resulting D2nuTable object is stored in a new attribute called
        'cut_l_table'.

        Args:
            l (int): Angular degree of the data points to exclude.
        """
        i_kept = np.argwhere(self.l != l)[:,0]
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
        self.cut_l_table = D2nuTable()
        self.cut_l_table.create(l_kept,n_kept,freq_kept,d2nu_kept,err_kept,matcov_kept)

    # Methods related to the covariance matrix

    def matcov_condnumber(self):
        """ Gets the condition number of the covariance matrix. """
        self.cond = np.linalg.cond(self.matcov)
        print(f"Condition number is {self.cond}")

    def matcov_invert(self):
        """ Inverts the covariance matrix.

        If fit_params["nsvd"] is different from -1, the inversion of the
        covariance matrix is regularised using the truncated singular value
        decomposition (SVD) method, fit_params["nsvd"] being the number of
        singular values to keep.
        """
        if fit_params["nsvd"] != -1:
            u,s,v = np.linalg.svd(self.matcov)
            self.inv_matcov = np.linalg.pinv(self.matcov,
                    rcond=s[fit_params["nsvd"]-1]/s.max())
        else:
            self.inv_matcov = np.linalg.inv(self.matcov)

    # Plotting methods

    def plot(self):
        """ Plots the second differences D2nu. """
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
        """ Plots the singular values of the covariance matrix. """
        u, s, v = np.linalg.svd(self.matcov)
        plt.figure()
        plt.plot(np.arange(1,len(s)+1),s,'o-')
        plt.show()

    def plot_matcov(self):
        """ Plots the covariance matrix. """
        plt.figure()
        plt.imshow(self.matcov)
        cb = plt.colorbar()
        plt.show()
