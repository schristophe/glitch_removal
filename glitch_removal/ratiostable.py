#
#
import numpy as np
import matplotlib.pyplot as plt
from .params import *

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

class rr010Table(object):
    """ Class to represent the ratios rr01/10.

    Attributes:
        l (np.array): Angular degrees.
        n (np.array): Radial orders.
        freq (np.array): Mode frequencies.
        rr010 (np.array): Frequency ratios rr010.
        err (np.array): Errors on frequency ratios rr010.
        matcov (np.array): Covariance matrix.
        cut_table (rr010Table):
            The initial dataset of rr010 but data points with large
            uncertainties are excluded.
        cut_range_table (rr010Table):
            The initial dataset of rr010 but only data points in a given
            frequency range are kept.
        cond (float): Condition number of the covariance matrix.
        inv_matcov (np.array): Inverse covariance matrix.
    """

    def __init__(self):
        """ Initialises an instance of rr010Table. """

    def create(self, l, n, freq, rr010, err, matcov):
        """ Creates an instance of rr010Table by directly passing the data.

        Args:
            l (np.array): Angular degrees.
            n (np.array): Radial orders.
            freq (np.array): Mode frequencies.
            rr010 (np.array): Frequency ratios rr010.
            err (np.array): Errors on frequency ratios rr010.
            matcov (np.array): Covariance matrix.
        """
        self.l = l
        self.n = n
        self.freq = freq
        self.rr010 = rr010
        self.err = err
        self.matcov = matcov

    def load(self):
        """ Loads the ratios rr01/10 from a file.

        Not implemented yet.
        """

    # Methods for data handling

    def cut(self, sigma):
        """ Removes observational points for which the rr010 uncertainty is
        above sigma * median(err).

        The resulting rr010Table object is stored in a new attribute called
        'cut_table'.

        Args:
            sigma (float):
                Defines the maximum tolerated rr010 uncertainty in cut_table
                (i.e. sigma * median(err)).
        """
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

    def cut_range(self, freqmin, freqmax):
        """ Keeps data points between freqmin and freqmax.

        The resulting rr010Table object is stored in a new attribute called
        'cut_range_table'.

        Args:
            freqmin (float): Minimum frequency.
            freqmax (float): Maximum frequency.
        """
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
        """ Plots the ratios rr010. """
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
        """ Plots the covariance matrix. """
        plt.figure()
        plt.imshow(self.matcov)
        cb = plt.colorbar()
        plt.show()

    def plot_matcov_svd(self):
        """ Plots the singular values of the covariance matrix. """
        u, s, v = np.linalg.svd(self.matcov)
        plt.figure()
        plt.plot(np.arange(1,len(s)+1),s,'o-')
        plt.show()





class r02Table(object):
    """ Class to represent the ratios r02.

    Attributes:
        l (np.array): Angular degrees.
        n (np.array): Radial orders.
        freq (np.array): Mode frequencies.
        r02 (np.array): Frequency ratios r02.
        err (np.array): Errors on frequency ratios r02.
        matcov (np.array): Covariance matrix.
        cut_table (r02Table):
            The initial dataset of r02 but data points with large uncertainties
            are excluded.
        cut_range_table (r02Table):
            The initial dataset of r02 but only data points in a given frequency
            range are kept.
        cond (float): Condition number of the covariance matrix.
        inv_matcov (np.array): Inverse covariance matrix.
    """

    def __init__(self):
        """ Initialises an instance of r02Table. """

    def create(self, l, n, freq, r02, err, matcov):
        """ Creates an instance of rr010Table by directly passing the data.

        Args:
            l (np.array): Angular degrees.
            n (np.array): Radial orders.
            freq (np.array): Mode frequencies.
            rr010 (np.array): Frequency ratios r02.
            err (np.array): Errors on frequency ratios r02.
            matcov (np.array): Covariance matrix.
        """
        self.l = l
        self.n = n
        self.freq = freq
        self.r02 = r02
        self.err = err
        self.matcov = matcov

    def load(self):
        """ Loads the ratios r02 from a file.

        Not implemented yet.
        """

    # Methods for data handling

    def cut(self, sigma):
        """ Removes observational points for which the r02 uncertainty is above
        sigma * median(err).

        The resulting r02Table object is stored in a new attribute called
        'cut_table'.

        Args:
            sigma (float):
                Defines the maximum tolerated r02 uncertainty in cut_table
                (i.e. sigma * median(err)).
        """
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

    def cut_range(self, freqmin, freqmax):
        """ Keeps only data points between freqmin and freqmax.

        The resulting r02Table object is stored in a new attribute called
        'cut_range_table'.

        Args:
            freqmin (float): Minimum frequency.
            freqmax (float): Maximum frequency.
        """
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
        """ Plots the ratios r02. """
        plt.figure()
        plt.xlabel(r'Frequency ($\mu$Hz)')
        plt.ylabel(r'${\it r_{02}}$')
        plt.errorbar(self.freq,self.r02,yerr=self.err,fmt='s',mfc=colors[2],
                mec='none',ecolor='#c4c4c4')
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
        plt.xticks(ticks=np.arange(len(self.n)),labels=self.n.astype(int))
        plt.yticks(ticks=np.arange(len(self.n)),labels=self.n.astype(int))
        plt.xlabel(r'$\it r_{02} (n)$')
        plt.ylabel(r'$\it r_{02} (n)$')
        cb = plt.colorbar()
        plt.show()
