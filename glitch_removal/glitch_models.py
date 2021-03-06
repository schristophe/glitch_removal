#
#
import numpy as np
import scipy.optimize as op
import emcee
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import corner
import os

from .params import *

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

class GlitchModel(object):
    """ Class to represent a model for acoustic glitches.

    Implemented glitch models are:
        * d2nu_verma
        * d2nu_mazumdar
        * d2nu_basu
        * rr010_const_amp
        * rr010_freqinv_amp
        * rr010_freqinvsq_amp
        * rr010_freqinvpoly_amp
        * rr010_freqinvsq_amp_polyper

    Attributes:
        model (function):
            Function to be fitted to the ratios rr010 or the second differences.
            Must be one of the models implemented in glitch_removal.
        data (D2nuTable or rr010Table):
            Data to be fitted. Must be either the ratios rr010 or the second
            differences.
        prob (function): Posterior probability function.
        bds (tuple): Boundaries for the model parameters.
        ig0 (np.array): Initial guess for the model parameters.
        sampler (emcee.ensemble.EnsembleSampler): Ensemble MCMC sampler.
        samples (np.array): Samples of the posterior probability.
        mod_params_mcmc (np.array):
            Final parameter values and their uncertainties.

    """

    def __init__(self,model,data):
        """ Initialises an instance of GlitchModel.

        Args:
            model (function):
                Function to be fitted to the ratios or second differences. Must
                be one of the models implemented in glitch_removal.
            data (D2nuTable or rr010Table):
                Data to be fitted. Must be either the ratios rr010 or the second
                differences.
        """
        self.model = model
        self.data = data
        update_params(self.model,self.data)
        self.prob = select_prob(self.model)

    def initial_guess(self):
        """ Finds an initial guess and appropriate boundaries for the glitch
            parameters. """
        # minimum detectable value for the acoustic location of the glitch
        T_min = 0.5*(self.data.freq.max() - self.data.freq.min())**-1
        # total acoustic radius
        T0 = (2*star_params["delta_nu"])**-1
        if self.model == d2nu_verma:
            a_init, V = np.polyfit(self.data.freq,self.data.d2nu,1,cov=True)
            sig_a_init = np.sqrt(np.diag(V))
            res_osc = self.data.d2nu - np.polyval(a_init,self.data.freq)
            tau_random_He = 0.5*(0.5*T_min+0.3*T0) + np.random.randn() * 0.5 * (0.3*T0-0.5*T_min)
            tau_random_CE = 0.5*T0 + np.random.randn() * 0.2 * T0
            c2_random = 5e-7
            ig0 = [a_init[1],a_init[0],0.1*res_osc.max(),\
                    tau_random_CE,np.pi,10*res_osc.max(),\
                    c2_random,tau_random_He,np.pi]
            self.bds = ((a_init[1]-5*sig_a_init[1],a_init[1]+5*sig_a_init[1]),\
                    (a_init[0]-5*sig_a_init[0],a_init[0]+5*sig_a_init[0]),\
                    (1e-2,1),
                    (0.3*T0, 0.7*T0),\
                    (-2*np.pi,4*np.pi),
                    (0.1*res_osc.max(),100*res_osc.max()),\
                    (1e-7,4e-6),\
                    (0.5*T_min, 0.3*T0),
                    (-2*np.pi,4*np.pi))
            nll = lambda *args: -lnlikelihood_d2nu(*args)
        elif self.model == d2nu_mazumdar:
            a_init, V = np.polyfit(self.data.freq,self.data.d2nu,0,cov=True)
            sig_a_init = np.sqrt(np.diag(V))
            res_osc = self.data.d2nu - np.polyval(a_init,self.data.freq)
            tau_random_He = 0.5*(0.5*T_min+0.3*T0) + np.random.randn() * 0.5 * (0.3*T0-0.5*T_min)
            tau_random_CE = 0.5*T0 + np.random.randn() * 0.2 * T0
            c2_random = 5e-7
            ig0 = [a_init[0],0.1*res_osc.max(),\
                    tau_random_CE,np.pi,10*res_osc.max(),\
                    c2_random,tau_random_He,np.pi]
            self.bds = ((a_init[0]-5*sig_a_init[0],a_init[0]+5*sig_a_init[0]),\
                    (1e-2,1),
                    (0.3*T0, 0.7*T0),\
                    (-2*np.pi,4*np.pi),
                    (0.1*res_osc.max(),100*res_osc.max()),\
                    (1e-7,4e-6),\
                    (0.5*T_min, 0.3*T0),
                    (-2*np.pi,4*np.pi))
            nll = lambda *args: -lnlikelihood_d2nu(*args)
        elif self.model == d2nu_basu:
            smooth_component_basu = lambda x, a1, a2, a3 : d2nu_basu(x,[a1,a2,a3,0,0,0,0,0,0,0,0])
            a_init, V = op.curve_fit(smooth_component_basu,self.data.freq,self.data.d2nu,p0 = [0.05,-5e-5,-3e5])
            sig_a_init = np.sqrt(np.diag(V))
            res_osc = self.data.d2nu - smooth_component_basu(self.data.freq,a_init[0],a_init[1],a_init[2])
            tau_random_He = 0.5*(T_min+0.5*T0) + np.random.randn() * 0.5 * (0.5*T0-T_min)
            tau_random_CE = 0.5*(2*T_min+T0) + np.random.randn() * 0.5 * (T0-2*T_min)
            ig0 = [a_init[0],a_init[1],a_init[2],\
                    res_osc.max(),res_osc.max()*star_params["numax"]**2,\
                    tau_random_He,np.pi,\
                    0.1*res_osc.max(),0.1*res_osc.max()*star_params["numax"]**2,\
                    tau_random_CE,np.pi]
            self.bds = ((a_init[0]-2*sig_a_init[0],a_init[0]+2*sig_a_init[0]),\
                    (a_init[1]-2*sig_a_init[1],a_init[1]+2*sig_a_init[1]),\
                    (a_init[2]-3*sig_a_init[2],a_init[2]+3*sig_a_init[2]),\
                    (0.1*res_osc.max(),2*res_osc.max()),\
                    (0.1*res_osc.max()*star_params["numax"]**2,2*res_osc.max()*star_params["numax"]**2),\
                    (T_min, 0.5*T0),\
                    (-2*np.pi,4*np.pi),\
                    (0,res_osc.max()),\
                    (0,res_osc.max()*star_params["numax"]**2),\
                    (2*T_min, T0),\
                    (-2*np.pi,4*np.pi))
            nll = lambda *args: -lnlikelihood_d2nu(*args)
        # elif self.model == d2nu_houdek:
        #     pass
        elif self.model == rr010_const_amp:
            smooth_component_rr010_const_amp = lambda x, c0, c1, c2 : rr010_const_amp(x,[c0,c1,c2,0,0,0])
            c_init, V = op.curve_fit(smooth_component_rr010_const_amp,\
                    self.data.freq,self.data.rr010)
            sig_c_init = np.sqrt(np.diag(V))
            res_osc = self.data.rr010 - smooth_component_rr010_const_amp(self.data.freq,c_init[0],c_init[1],c_init[2])
            T_random = 0.5*(T_min+T0) + np.random.randn() * 0.5 * (T0-T_min)
            ig0 = [c_init[0],c_init[1],c_init[2],res_osc.max(),T_random,np.pi]
            self.bds = ((c_init[0]-3*sig_c_init[0],c_init[0]+3*sig_c_init[0]),\
                    (c_init[1]-5*sig_c_init[1],c_init[1]+5*sig_c_init[1]),\
                    (c_init[2]-10*sig_c_init[2],c_init[2]+10*sig_c_init[2]),\
                    (1e-5,3*res_osc.max()),\
                    (2*T_min,T0-2*T_min),\
                    (-2*np.pi,4*np.pi))
            nll = lambda *args: -lnlikelihood_rr010(*args)
        elif self.model == rr010_freqinv_amp:
            smooth_component_rr010_freqinv_amp = lambda x, c0, c1, c2 : rr010_freqinv_amp(x,[c0,c1,c2,0,0,0])
            c_init, V = op.curve_fit(smooth_component_rr010_freqinv_amp,\
                    self.data.freq,self.data.rr010)
            sig_c_init = np.sqrt(np.diag(V))
            res_osc = self.data.rr010 - smooth_component_rr010_freqinv_amp(self.data.freq,c_init[0],c_init[1],c_init[2])
            T_random = 0.5*(T_min+T0) + np.random.randn() * 0.5 * (T0-T_min)
            ig0 = [c_init[0],c_init[1],c_init[2],res_osc.max(),T_random,np.pi]
            self.bds = ((c_init[0]-5*sig_c_init[0],c_init[0]+5*sig_c_init[0]),\
                    (c_init[1]-10*sig_c_init[1],c_init[1]+10*sig_c_init[1]),\
                    (c_init[2]-20*sig_c_init[2],c_init[2]+20*sig_c_init[2]),\
                    (1e-5,3*res_osc.max()),\
                    (T_min,T0-2*T_min),\
                    (-2*np.pi,4*np.pi))
            nll = lambda *args: -lnlikelihood_rr010(*args)
        elif self.model == rr010_freqinvsq_amp:
            smooth_component_rr010_freqinvsq_amp = lambda x, c0, c1, c2 : rr010_freqinvsq_amp(x,[c0,c1,c2,0,0,0])
            c_init, V = op.curve_fit(smooth_component_rr010_freqinvsq_amp,\
                    self.data.freq,self.data.rr010)
            sig_c_init = np.sqrt(np.diag(V))
            res_osc = self.data.rr010 - smooth_component_rr010_freqinvsq_amp(self.data.freq,c_init[0],c_init[1],c_init[2])
            T_random = 0.5*(T_min+T0) + np.random.randn() * 0.5 * (T0-T_min)
            ig0 = [c_init[0],c_init[1],c_init[2],res_osc.max(),T_random,np.pi]
            self.bds = ((c_init[0]-3*sig_c_init[0],c_init[0]+3*sig_c_init[0]),\
                    (c_init[1]-5*sig_c_init[1],c_init[1]+5*sig_c_init[1]),\
                    (c_init[2]-10*sig_c_init[2],c_init[2]+10*sig_c_init[2]),\
                    (5e-5,3*res_osc.max()),\
                    (T_min,T0-2*T_min),\
                    (-2*np.pi,4*np.pi))
            nll = lambda *args: -lnlikelihood_rr010(*args)
        elif self.model == rr010_freqinvpoly_amp:
            smooth_component_rr010_freqinvpoly_amp = lambda x, c0, c1, c2 : rr010_freqinvpoly_amp(x,[c0,c1,c2,0,0,0,0])
            c_init, V = op.curve_fit(smooth_component_rr010_freqinvpoly_amp,\
                    self.data.freq,self.data.rr010)
            sig_c_init = np.sqrt(np.diag(V))
            res_osc = self.data.rr010 - smooth_component_rr010_freqinvpoly_amp(self.data.freq, c_init[0], c_init[1], c_init[2])
            T_random = 0.5*(2*T_min+T0) + np.random.randn() * 0.5 * (T0-2*T_min)
            ig0 = [c_init[0],c_init[1],c_init[2],res_osc.max(),res_osc.max(),T_random,np.pi]
            self.bds = ((c_init[0]-3*sig_c_init[0],c_init[0]+3*sig_c_init[0]),\
                    (c_init[1]-5*sig_c_init[1],c_init[1]+5*sig_c_init[1]),\
                    (c_init[2]-10*sig_c_init[2],c_init[2]+10*sig_c_init[2]),\
                    (1e-5,3*res_osc.max()),\
                    (1e-5,3*res_osc.max()),\
                    (2*T_min,T0-2*T_min),\
                    (-2*np.pi,4*np.pi))
            nll = lambda *args: -lnlikelihood_rr010(*args)
        elif self.model == rr010_freqinvsq_amp_polyper:
            smooth_component_rr010_freqinvsq_amp_polyper = lambda x, c0, c1, c2 : rr010_freqinvpoly_amp(x,[c0,c1,c2,0,0,0,0,0,0])
            c_init, V = op.curve_fit(smooth_component_rr010_freqinvsq_amp_polyper,\
                    self.data.freq,self.data.rr010)
            sig_c_init = np.sqrt(np.diag(V))
            res_osc = self.data.rr010 - smooth_component_rr010_freqinvsq_amp_polyper(self.data.freq, c_init[0], c_init[1], c_init[2])
            T_random_0 = 0.5*(T_min+T0) + np.random.randn() * 0.5 * (T0-T_min)
            T_random_1 = 0.5*(T_min+T0) + np.random.randn() * 0.5 * (T0-T_min)
            ig0 = [c_init[0],c_init[1],c_init[2],res_osc.max(),T_random_0,np.pi,0.1*res_osc.max(),T_random_1,np.pi]
            self.bds = ((c_init[0]-3*sig_c_init[0],c_init[0]+3*sig_c_init[0]),\
                    (c_init[1]-5*sig_c_init[1],c_init[1]+5*sig_c_init[1]),\
                    (c_init[2]-10*sig_c_init[2],c_init[2]+10*sig_c_init[2]),\
                    (1e-5,3*res_osc.max()),\
                    (T_min,T0-T_min),\
                    (-2*np.pi,4*np.pi),\
                    (1e-5,res_osc.max()),\
                    (T_min,T0-T_min),\
                    (-2*np.pi,4*np.pi))
            nll = lambda *args: -lnlikelihood_rr010(*args)
        self.ig0 = op.minimize(nll,ig0,args=(self.model,self.data),bounds=self.bds)
        self.ig0 = self.ig0["x"]


    def setup_smooth_component(self):
        """ Ensures the fitted coefficients of the polynomial in rr010 models
        will not be correlated by setting the value of beta, gamma1 and gamma2.
        See Appendix B of Deheuvels et al. 2016 for more information. """
        fit_params["beta"], fit_params["gamma1"], fit_params["gamma2"] = \
                op.fsolve(f2solve_smooth_component,x0=[star_params["numax"]]*3,args=(self.data.freq,self.data.inv_matcov))


    def run_mcmc(self):
        """ Runs the MCMC sampling of the posterior probability function. """
        pos = [self.ig0 + 1e-4*self.ig0*np.random.randn(emcee_params["ndim"]) for i in range(emcee_params["nwalkers"])]
        sampler = emcee.EnsembleSampler(emcee_params["nwalkers"], emcee_params["ndim"],self.prob, args=(self.model,self.data,self.bds))
        sampler.run_mcmc(pos,emcee_params["nsteps"])
        # Process MCMC chains
        if self.model == d2nu_verma:
            sampler.chain[:,:,[4,8]] = np.mod(sampler.chain[:,:,[4,8]],2*np.pi) # fold the phases on the [0,2pi] interval
            sampler.chain[:,:,[3,7]] = 1e6 * sampler.chain[:,:,[3,7]]   # convert acoustic depth in seconds
        elif self.model == d2nu_mazumdar:
            sampler.chain[:,:,[3,7]] = np.mod(sampler.chain[:,:,[3,7]],2*np.pi) # fold the phases on the [0,2pi] interval
            sampler.chain[:,:,[2,6]] = 1e6 * sampler.chain[:,:,[2,6]]   # convert acoustic depth in seconds
        elif self.model == d2nu_basu:
            sampler.chain[:,:,[6,10]] = np.mod(sampler.chain[:,:,[6,10]],2*np.pi)
            sampler.chain[:,:,[5,9]] = 1e6 * sampler.chain[:,:,[5,9]]
        # elif self.model == d2nu_houdek:
        #     sampler.chain[:,:,[]] = np.mod(sampler.chain[:,:,[]],2*np.pi)
        elif self.model == rr010_const_amp or self.model == rr010_freqinv_amp or \
                self.model == rr010_freqinvsq_amp:
            sampler.chain[:,:,5] = np.mod(sampler.chain[:,:,5],2*np.pi)
            sampler.chain[:,:,4] = 1e6 * sampler.chain[:,:,4]
        elif self.model == rr010_freqinvpoly_amp:
            sampler.chain[:,:,6] = np.mod(sampler.chain[:,:,6],2*np.pi)
            sampler.chain[:,:,5] = 1e6 * sampler.chain[:,:,5]
        elif self.model == rr010_freqinvsq_amp_polyper:
            sampler.chain[:,:,[5,8]] = np.mod(sampler.chain[:,:,[5,8]],2*np.pi) # fold the phases on the [0,2pi] interval
            sampler.chain[:,:,[4,7]] = 1e6 * sampler.chain[:,:,[4,7]]   # convert acoustic depth in seconds
        self.sampler = sampler
        # Discard burn-in steps
        self.samples = sampler.chain[:, emcee_params["nburns"]:, :].reshape((-1, emcee_params["ndim"]))
        # Estimate of the model parameters from the 16th, 50th and 84th
        # percentile of the marginal posterior distributions
        mod_params_mcmc = np.array([])
        p = np.percentile(self.samples, [50, 84, 16],axis=0)
        p[2,:] = p[2,:] - p[0,:]
        p[1,:] = p[1,:] - p[0,:]
        self.mod_params_mcmc = np.transpose(p)

    def log_and_plot(self, what2save=[]):
        """ Saves a log file with model parameter estimates and plot/save the
            corner plot, the evolution of the walkers position and the
            result of the sampling in the observed plane. """
        if save_params["directory"] == '':
            directory = ''
        elif not os.path.isdir(save_params["directory"]):
            os.mkdir(save_params["directory"])
            directory = save_params["directory"].split("/")[0]+'/' # making sure there is a slash
        else:
            directory = save_params["directory"].split("/")[0]+'/' # making sure there is a slash
        if self.model == d2nu_verma:
            model_name = 'd2nu_verma'
            # Log file
            logfile = open(directory+save_params["nameplate"]+'_'+model_name+'.log',"w")
            logfile.write('Glitch fitting with glitch_removal\n'+
                    'Glitch model: '+model_name+'\n'+
                    'a0 + a1*freq +' +\
                    '(b0/freq**2)*sin(4*pi*freq*tau_CE+phi_CE) + ' + \
                    'c0 * exp(-c2*freq**2) * sin(4*pi*freq*tau_He+phi_He)\n\n')
            logfile.write('Results\n'+'_____________\n'+\
                    'a0\t'+str('%10.4e' % self.mod_params_mcmc[0][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[0][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[0][1])+'\n'+\
                    'a1\t'+str('%10.4e' % self.mod_params_mcmc[1][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[1][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[1][1])+'\n'+\
                    'b0\t'+str('%10.4e' % self.mod_params_mcmc[2][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[2][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[2][1])+'\n'+\
                    'tau_CE\t'+str('%10.2f' % self.mod_params_mcmc[3][0])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[3][2])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[3][1])+'\n'+\
                    'phi_CE\t'+str('%10.4f' % self.mod_params_mcmc[4][0])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[4][2])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[4][1])+'\n'+\
                    'c0\t'+str('%10.4e' % self.mod_params_mcmc[5][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[5][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[5][1])+'\n'+\
                    'c2\t'+str('%10.4e' % self.mod_params_mcmc[6][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[6][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[6][1])+'\n'+\
                    'tau_He\t'+str('%10.2f' % self.mod_params_mcmc[7][0])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[7][2])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[7][1])+'\n'+\
                    'phi_He\t'+str('%10.4f' % self.mod_params_mcmc[8][0])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[8][2])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[8][1])+'\n\n')
            logfile.write('emcee parameters\n'+'_____________\n'+\
                    'nwalkers\t'+str('%d' % emcee_params["nwalkers"])+'\n'+\
                    'nburns\t'+str('%d' % emcee_params["nburns"])+'\n'+\
                    'nsteps\t'+str('%d' % emcee_params["nsteps"])+'\n\n')
            logfile.write('fit parameters\n'+'_____________\n'+\
                    'freqref\t'+str('%10.2f' % fit_params["freqref"])+'\n'+\
                    'nsvd\t'+str('%d' % fit_params["nsvd"])+'\n\n')
            logfile.write('star parameters\n'+'_____________\n'+\
                    'delta_nu\t'+str('%10.2f' % star_params["delta_nu"])+'\n'+\
                    'numax\t'+str('%10.2f' % star_params["numax"]))
            logfile.close()
            # Corner plot
            labels = ['$a_0$','$a_1$','$b_0$',r'$\tau_{\rm CE}$',\
                    r'$\phi_{\rm CE}$','$c_0$','$c_2$',r'$\tau_{\rm He}$',\
                    r'$\phi_{\rm He}$']
            fig_corner = corner.corner(self.samples,labels=labels,quantiles=[0.16, 0.5, 0.84], show_titles=True)
            # Walkers position
            walkers_redux = np.percentile(self.sampler.chain, [1, 50, 99],axis=0)
            nsteps_tab = np.arange(1,emcee_params["nsteps"]+1)
            fig_walkers = plt.figure(figsize=(10,25))
            gs = gridspec.GridSpec(9,1)
            ax1 = fig_walkers.add_subplot(gs[0,0])
            ax2 = fig_walkers.add_subplot(gs[1,0])
            ax3 = fig_walkers.add_subplot(gs[2,0])
            ax4 = fig_walkers.add_subplot(gs[3,0])
            ax5 = fig_walkers.add_subplot(gs[4,0])
            ax6 = fig_walkers.add_subplot(gs[5,0])
            ax7 = fig_walkers.add_subplot(gs[6,0])
            ax8 = fig_walkers.add_subplot(gs[7,0])
            ax9 = fig_walkers.add_subplot(gs[8,0])
            #
            ax1.fill_between(nsteps_tab,walkers_redux[0,:,0],walkers_redux[2,:,0],color='#b5b5b5',alpha=0.5)
            ax1.plot(nsteps_tab,walkers_redux[1,:,0],lw=2)
            ax1.set_ylabel(labels[0])
            #
            ax2.fill_between(nsteps_tab,walkers_redux[0,:,1],walkers_redux[2,:,1],color='#b5b5b5',alpha=0.5)
            ax2.plot(nsteps_tab,walkers_redux[1,:,1],lw=2)
            ax2.set_ylabel(labels[1])
            #
            ax3.fill_between(nsteps_tab,walkers_redux[0,:,2],walkers_redux[2,:,2],color='#b5b5b5',alpha=0.5)
            ax3.plot(nsteps_tab,walkers_redux[1,:,2],lw=2)
            ax3.set_ylabel(labels[2])
            #
            ax4.fill_between(nsteps_tab,walkers_redux[0,:,3],walkers_redux[2,:,3],color='#b5b5b5',alpha=0.5)
            ax4.plot(nsteps_tab,walkers_redux[1,:,3],lw=2)
            ax4.set_ylabel(labels[3])
            #
            ax5.fill_between(nsteps_tab,walkers_redux[0,:,4],walkers_redux[2,:,4],color='#b5b5b5',alpha=0.5)
            ax5.plot(nsteps_tab,walkers_redux[1,:,4],lw=2)
            ax5.set_ylabel(labels[4])
            #
            ax6.fill_between(nsteps_tab,walkers_redux[0,:,5],walkers_redux[2,:,5],color='#b5b5b5',alpha=0.5)
            ax6.plot(nsteps_tab,walkers_redux[1,:,5],lw=2)
            ax6.set_ylabel(labels[5])
            #
            ax7.fill_between(nsteps_tab,walkers_redux[0,:,6],walkers_redux[2,:,6],color='#b5b5b5',alpha=0.5)
            ax7.plot(nsteps_tab,walkers_redux[1,:,6],lw=2)
            ax7.set_ylabel(labels[6])
            #
            ax8.fill_between(nsteps_tab,walkers_redux[0,:,7],walkers_redux[2,:,7],color='#b5b5b5',alpha=0.5)
            ax8.plot(nsteps_tab,walkers_redux[1,:,7],lw=2)
            ax8.set_ylabel(labels[7])
            #
            ax9.fill_between(nsteps_tab,walkers_redux[0,:,8],walkers_redux[2,:,8],color='#b5b5b5',alpha=0.5)
            ax9.plot(nsteps_tab,walkers_redux[1,:,8],lw=2)
            ax9.set_ylabel(labels[8])
            ax9.set_xlabel(r'$n_{\rm steps}$')
            # Results in the observed plane
            freq_array = np.linspace(self.data.freq.min(),self.data.freq.max(),100)
            fig_result = plt.figure()
            gs = gridspec.GridSpec(3,1)
            ax1 = fig_result.add_subplot(gs[:2,0])
            markers = ['o','^','s']
            for l in np.arange(3):
                i_l = self.data.l == l
                ax1.errorbar(self.data.freq[i_l],self.data.d2nu[i_l],
                        yerr=self.data.err[i_l],fmt=markers[l],
                        mfc=colors[l],mec='none',ecolor='#c4c4c4',
                        label=r'$\ell = {}$'.format(l))
            plt.legend()
            for mod_params in self.samples[np.random.randint(len(self.samples), size=100)]:
                mod_params[[3,7]] = 1e-6 * mod_params[[3,7]]
                ax1.plot(freq_array,d2nu_verma(freq_array,mod_params),c='#999999',alpha=0.1)
            best_params = self.mod_params_mcmc[:,0]
            best_params[[3,7]] = 1e-6 * best_params[[3,7]]
            phase_d2nu_verma = lambda x, phi, psi : d2nu_verma(x,\
                    [best_params[0],best_params[1],best_params[2],best_params[3],\
                    phi,best_params[5],best_params[6],best_params[7],psi])
            phase_opt, V = op.curve_fit(phase_d2nu_verma,\
                    self.data.freq,self.data.d2nu,bounds=((0,0),(2*np.pi,2*np.pi)))
            best_params[[4,8]] = phase_opt
            ax1.plot(freq_array,d2nu_verma(freq_array,best_params),color=colors[3])
            ax1.set_xlabel(r'Frequency ($\mu$Hz)')
            ax1.set_ylabel(r'$\Delta_2 \nu$ $(\mu Hz)$')
            ax2 = fig_result.add_subplot(gs[2,0])
            for l in np.arange(3):
                i_l = self.data.l == l
                ax2.errorbar(self.data.freq[i_l],self.data.d2nu[i_l]-d2nu_verma(self.data.freq[i_l],best_params),
                        yerr=self.data.err[i_l],fmt=markers[l],
                        mfc=colors[l],mec='none',ecolor='#c4c4c4',
                        label=r'$\ell = {}$'.format(l))
            ax2.set_xlabel(r'Frequency ($\mu$Hz)')
            ax2.set_ylabel(r'$\delta\Delta_2 \nu$ $(\mu Hz)$')
            res = self.data.d2nu - d2nu_verma(self.data.freq,best_params)
            ax2.set_ylim([0.8*res.min(), 1.2*res.max()])
        elif self.model == d2nu_mazumdar:
            model_name = 'd2nu_mazumdar'
            # Log file
            logfile = open(directory+save_params["nameplate"]+'_'+model_name+'.log',"w")
            logfile.write('Glitch fitting with glitch_removal\n'+
                    'Glitch model: '+model_name+'\n'+
                    'a0 + ' +\
                    '(b0/freq**2)*sin(4*pi*freq*tau_CE+phi_CE) + ' + \
                    'c0 * exp(-c2*freq**2) * sin(4*pi*freq*tau_He+phi_He)\n\n')
            logfile.write('Results\n'+'_____________\n'+\
                    'a0\t'+str('%10.4e' % self.mod_params_mcmc[0][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[0][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[0][1])+'\n'+\
                    'b0\t'+str('%10.4e' % self.mod_params_mcmc[1][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[1][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[1][1])+'\n'+\
                    'tau_CE\t'+str('%10.2f' % self.mod_params_mcmc[2][0])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[2][2])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[2][1])+'\n'+\
                    'phi_CE\t'+str('%10.4f' % self.mod_params_mcmc[3][0])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[3][2])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[3][1])+'\n'+\
                    'c0\t'+str('%10.4e' % self.mod_params_mcmc[4][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[4][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[4][1])+'\n'+\
                    'c2\t'+str('%10.4e' % self.mod_params_mcmc[5][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[5][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[5][1])+'\n'+\
                    'tau_He\t'+str('%10.2f' % self.mod_params_mcmc[6][0])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[6][2])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[6][1])+'\n'+\
                    'phi_He\t'+str('%10.4f' % self.mod_params_mcmc[7][0])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[7][2])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[7][1])+'\n\n')
            logfile.write('emcee parameters\n'+'_____________\n'+\
                    'nwalkers\t'+str('%d' % emcee_params["nwalkers"])+'\n'+\
                    'nburns\t'+str('%d' % emcee_params["nburns"])+'\n'+\
                    'nsteps\t'+str('%d' % emcee_params["nsteps"])+'\n\n')
            logfile.write('fit parameters\n'+'_____________\n'+\
                    'freqref\t'+str('%10.2f' % fit_params["freqref"])+'\n'+\
                    'nsvd\t'+str('%d' % fit_params["nsvd"])+'\n\n')
            logfile.write('star parameters\n'+'_____________\n'+\
                    'delta_nu\t'+str('%10.2f' % star_params["delta_nu"])+'\n'+\
                    'numax\t'+str('%10.2f' % star_params["numax"]))
            logfile.close()
            # Corner plot
            labels = ['$a_0$','$b_0$',r'$\tau_{\rm CE}$',\
                    r'$\phi_{\rm CE}$','$c_0$','$c_2$',r'$\tau_{\rm He}$',\
                    r'$\phi_{\rm He}$']
            fig_corner = corner.corner(self.samples,labels=labels,quantiles=[0.16, 0.5, 0.84], show_titles=True)
            # Walkers position
            walkers_redux = np.percentile(self.sampler.chain, [1, 50, 99],axis=0)
            nsteps_tab = np.arange(1,emcee_params["nsteps"]+1)
            fig_walkers = plt.figure(figsize=(10,25))
            gs = gridspec.GridSpec(8,1)
            ax1 = fig_walkers.add_subplot(gs[0,0])
            ax2 = fig_walkers.add_subplot(gs[1,0])
            ax3 = fig_walkers.add_subplot(gs[2,0])
            ax4 = fig_walkers.add_subplot(gs[3,0])
            ax5 = fig_walkers.add_subplot(gs[4,0])
            ax6 = fig_walkers.add_subplot(gs[5,0])
            ax7 = fig_walkers.add_subplot(gs[6,0])
            ax8 = fig_walkers.add_subplot(gs[7,0])
            #
            ax1.fill_between(nsteps_tab,walkers_redux[0,:,0],walkers_redux[2,:,0],color='#b5b5b5',alpha=0.5)
            ax1.plot(nsteps_tab,walkers_redux[1,:,0],lw=2)
            ax1.set_ylabel(labels[0])
            #
            ax2.fill_between(nsteps_tab,walkers_redux[0,:,1],walkers_redux[2,:,1],color='#b5b5b5',alpha=0.5)
            ax2.plot(nsteps_tab,walkers_redux[1,:,1],lw=2)
            ax2.set_ylabel(labels[1])
            #
            ax3.fill_between(nsteps_tab,walkers_redux[0,:,2],walkers_redux[2,:,2],color='#b5b5b5',alpha=0.5)
            ax3.plot(nsteps_tab,walkers_redux[1,:,2],lw=2)
            ax3.set_ylabel(labels[2])
            #
            ax4.fill_between(nsteps_tab,walkers_redux[0,:,3],walkers_redux[2,:,3],color='#b5b5b5',alpha=0.5)
            ax4.plot(nsteps_tab,walkers_redux[1,:,3],lw=2)
            ax4.set_ylabel(labels[3])
            #
            ax5.fill_between(nsteps_tab,walkers_redux[0,:,4],walkers_redux[2,:,4],color='#b5b5b5',alpha=0.5)
            ax5.plot(nsteps_tab,walkers_redux[1,:,4],lw=2)
            ax5.set_ylabel(labels[4])
            #
            ax6.fill_between(nsteps_tab,walkers_redux[0,:,5],walkers_redux[2,:,5],color='#b5b5b5',alpha=0.5)
            ax6.plot(nsteps_tab,walkers_redux[1,:,5],lw=2)
            ax6.set_ylabel(labels[5])
            #
            ax7.fill_between(nsteps_tab,walkers_redux[0,:,6],walkers_redux[2,:,6],color='#b5b5b5',alpha=0.5)
            ax7.plot(nsteps_tab,walkers_redux[1,:,6],lw=2)
            ax7.set_ylabel(labels[6])
            #
            ax8.fill_between(nsteps_tab,walkers_redux[0,:,7],walkers_redux[2,:,7],color='#b5b5b5',alpha=0.5)
            ax8.plot(nsteps_tab,walkers_redux[1,:,7],lw=2)
            ax8.set_ylabel(labels[7])
            #
            ax8.set_xlabel(r'$n_{\rm steps}$')
            # Results in the observed plane
            freq_array = np.linspace(self.data.freq.min(),self.data.freq.max(),100)
            fig_result = plt.figure()
            gs = gridspec.GridSpec(3,1)
            ax1 = fig_result.add_subplot(gs[:2,0])
            markers = ['o','^','s']
            for l in np.arange(3):
                i_l = self.data.l == l
                ax1.errorbar(self.data.freq[i_l],self.data.d2nu[i_l],
                        yerr=self.data.err[i_l],fmt=markers[l],
                        mfc=colors[l],mec='none',ecolor='#c4c4c4',
                        label=r'$\ell = {}$'.format(l))
            plt.legend()
            for mod_params in self.samples[np.random.randint(len(self.samples), size=100)]:
                mod_params[[2,6]] = 1e-6 * mod_params[[2,6]]
                ax1.plot(freq_array,d2nu_mazumdar(freq_array,mod_params),c='#999999',alpha=0.1)
            best_params = self.mod_params_mcmc[:,0]
            best_params[[2,6]] = 1e-6 * best_params[[2,6]]
            phase_d2nu_mazumdar = lambda x, phi, psi : d2nu_mazumdar(x,\
                    [best_params[0],best_params[1],best_params[2],\
                    phi,best_params[4],best_params[5],best_params[6],psi])
            phase_opt, V = op.curve_fit(phase_d2nu_mazumdar,\
                    self.data.freq,self.data.d2nu,bounds=((0,0),(2*np.pi,2*np.pi)))
            best_params[[3,7]] = phase_opt
            ax1.plot(freq_array,d2nu_mazumdar(freq_array,best_params),color=colors[3])
            ax1.set_xlabel(r'Frequency ($\mu$Hz)')
            ax1.set_ylabel(r'$\Delta_2 \nu$ $(\mu Hz)$')
            ax2 = fig_result.add_subplot(gs[2,0])
            for l in np.arange(3):
                i_l = self.data.l == l
                ax2.errorbar(self.data.freq[i_l],self.data.d2nu[i_l]-d2nu_mazumdar(self.data.freq[i_l],best_params),
                        yerr=self.data.err[i_l],fmt=markers[l],
                        mfc=colors[l],mec='none',ecolor='#c4c4c4',
                        label=r'$\ell = {}$'.format(l))
            ax2.set_xlabel(r'Frequency ($\mu$Hz)')
            ax2.set_ylabel(r'$\delta\Delta_2 \nu$ $(\mu Hz)$')
            res = self.data.d2nu - d2nu_mazumdar(self.data.freq,best_params)
            ax2.set_ylim([0.8*res.min(), 1.2*res.max()])
        elif self.model == d2nu_basu:
            model_name = 'd2nu_basu'
            # Log file
            logfile = open(directory+save_params["nameplate"]+'_'+model_name+'.log',"w")
            logfile.write('Glitch fitting with glitch_removal\n'+
                    'Glitch model: '+model_name+'\n'+
                    'a1 + a2*freq + a3/freq**2 + ' +\
                    '(b1 + b2/freq**2)*sin(4*pi*freq*tau_He+phi_He) + ' + \
                    '(c1 + c2/freq**2) * sin(4*pi*freq*tau_CE+phi_CE)\n\n')
            logfile.write('Results\n'+'_____________\n'+\
                    'a1\t'+str('%10.4e' % self.mod_params_mcmc[0][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[0][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[0][1])+'\n'+\
                    'a2\t'+str('%10.4e' % self.mod_params_mcmc[1][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[1][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[1][1])+'\n'+\
                    'a3\t'+str('%10.4e' % self.mod_params_mcmc[2][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[2][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[2][1])+'\n'+\
                    'b1\t'+str('%10.4e' % self.mod_params_mcmc[3][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[3][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[3][1])+'\n'+\
                    'b2\t'+str('%10.4e' % self.mod_params_mcmc[4][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[4][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[4][1])+'\n'+\
                    'tau_He\t'+str('%10.2f' % self.mod_params_mcmc[5][0])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[5][2])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[5][1])+'\n'+\
                    'phi_He\t'+str('%10.4f' % self.mod_params_mcmc[6][0])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[6][2])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[6][1])+'\n'+\
                    'c1\t'+str('%10.4e' % self.mod_params_mcmc[7][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[7][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[7][1])+'\n'+\
                    'c2\t'+str('%10.4e' % self.mod_params_mcmc[8][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[8][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[8][1])+'\n'+\
                    'tau_CE\t'+str('%10.2f' % self.mod_params_mcmc[9][0])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[9][2])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[9][1])+'\n'+\
                    'phi_CE\t'+str('%10.4f' % self.mod_params_mcmc[10][0])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[10][2])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[10][1])+'\n\n')
            logfile.write('emcee parameters\n'+'_____________\n'+\
                    'nwalkers\t'+str('%d' % emcee_params["nwalkers"])+'\n'+\
                    'nburns\t'+str('%d' % emcee_params["nburns"])+'\n'+\
                    'nsteps\t'+str('%d' % emcee_params["nsteps"])+'\n\n')
            logfile.write('fit parameters\n'+'_____________\n'+\
                    'freqref\t'+str('%10.2f' % fit_params["freqref"])+'\n'+\
                    'nsvd\t'+str('%d' % fit_params["nsvd"])+'\n\n')
            logfile.write('star parameters\n'+'_____________\n'+\
                    'delta_nu\t'+str('%10.2f' % star_params["delta_nu"])+'\n'+\
                    'numax\t'+str('%10.2f' % star_params["numax"]))
            logfile.close()
            # Corner plot
            labels = ['$a_1$','$a_2$','$a_3$','$b_1$','$b_2$',\
                    r'$\tau_{\rm He}$',r'$\phi_{\rm He}$',\
                    '$c_1$','$c_2$',r'$\tau_{\rm CE}$',r'$\phi_{\rm CE}$']
            fig_corner = corner.corner(self.samples,labels=labels,quantiles=[0.16, 0.5, 0.84], show_titles=True)
            # Walkers position
            walkers_redux = np.percentile(self.sampler.chain, [1, 50, 99],axis=0)
            nsteps_tab = np.arange(1,emcee_params["nsteps"]+1)
            fig_walkers = plt.figure(figsize=(10,25))
            gs = gridspec.GridSpec(11,1)
            ax1 = fig_walkers.add_subplot(gs[0,0])
            ax2 = fig_walkers.add_subplot(gs[1,0])
            ax3 = fig_walkers.add_subplot(gs[2,0])
            ax4 = fig_walkers.add_subplot(gs[3,0])
            ax5 = fig_walkers.add_subplot(gs[4,0])
            ax6 = fig_walkers.add_subplot(gs[5,0])
            ax7 = fig_walkers.add_subplot(gs[6,0])
            ax8 = fig_walkers.add_subplot(gs[7,0])
            ax9 = fig_walkers.add_subplot(gs[8,0])
            ax10 = fig_walkers.add_subplot(gs[9,0])
            ax11 = fig_walkers.add_subplot(gs[10,0])
            #
            ax1.fill_between(nsteps_tab,walkers_redux[0,:,0],walkers_redux[2,:,0],color='#b5b5b5',alpha=0.5)
            ax1.plot(nsteps_tab,walkers_redux[1,:,0],lw=2)
            ax1.set_ylabel(labels[0])
            #
            ax2.fill_between(nsteps_tab,walkers_redux[0,:,1],walkers_redux[2,:,1],color='#b5b5b5',alpha=0.5)
            ax2.plot(nsteps_tab,walkers_redux[1,:,1],lw=2)
            ax2.set_ylabel(labels[1])
            #
            ax3.fill_between(nsteps_tab,walkers_redux[0,:,2],walkers_redux[2,:,2],color='#b5b5b5',alpha=0.5)
            ax3.plot(nsteps_tab,walkers_redux[1,:,2],lw=2)
            ax3.set_ylabel(labels[2])
            #
            ax4.fill_between(nsteps_tab,walkers_redux[0,:,3],walkers_redux[2,:,3],color='#b5b5b5',alpha=0.5)
            ax4.plot(nsteps_tab,walkers_redux[1,:,3],lw=2)
            ax4.set_ylabel(labels[3])
            #
            ax5.fill_between(nsteps_tab,walkers_redux[0,:,4],walkers_redux[2,:,4],color='#b5b5b5',alpha=0.5)
            ax5.plot(nsteps_tab,walkers_redux[1,:,4],lw=2)
            ax5.set_ylabel(labels[4])
            #
            ax6.fill_between(nsteps_tab,walkers_redux[0,:,5],walkers_redux[2,:,5],color='#b5b5b5',alpha=0.5)
            ax6.plot(nsteps_tab,walkers_redux[1,:,5],lw=2)
            ax6.set_ylabel(labels[5])
            #
            ax7.fill_between(nsteps_tab,walkers_redux[0,:,6],walkers_redux[2,:,6],color='#b5b5b5',alpha=0.5)
            ax7.plot(nsteps_tab,walkers_redux[1,:,6],lw=2)
            ax7.set_ylabel(labels[6])
            #
            ax8.fill_between(nsteps_tab,walkers_redux[0,:,7],walkers_redux[2,:,7],color='#b5b5b5',alpha=0.5)
            ax8.plot(nsteps_tab,walkers_redux[1,:,7],lw=2)
            ax8.set_ylabel(labels[7])
            #
            ax9.fill_between(nsteps_tab,walkers_redux[0,:,8],walkers_redux[2,:,8],color='#b5b5b5',alpha=0.5)
            ax9.plot(nsteps_tab,walkers_redux[1,:,8],lw=2)
            ax9.set_ylabel(labels[8])
            #
            ax10.fill_between(nsteps_tab,walkers_redux[0,:,9],walkers_redux[2,:,9],color='#b5b5b5',alpha=0.5)
            ax10.plot(nsteps_tab,walkers_redux[1,:,9],lw=2)
            ax10.set_ylabel(labels[9])
            #
            ax11.fill_between(nsteps_tab,walkers_redux[0,:,10],walkers_redux[2,:,10],color='#b5b5b5',alpha=0.5)
            ax11.plot(nsteps_tab,walkers_redux[1,:,10],lw=2)
            ax11.set_ylabel(labels[10])
            ax11.set_xlabel(r'$n_{\rm steps}$')
            # Results in the observed plane
            freq_array = np.linspace(self.data.freq.min(),self.data.freq.max(),100)
            fig_result = plt.figure()
            gs = gridspec.GridSpec(3,1)
            ax1 = fig_result.add_subplot(gs[:2,0])
            markers = ['o','^','s']
            for l in np.arange(3):
                i_l = self.data.l == l
                ax1.errorbar(self.data.freq[i_l],self.data.d2nu[i_l],
                        yerr=self.data.err[i_l],fmt=markers[l],
                        mfc=colors[l],mec='none',ecolor='#c4c4c4',
                        label=r'$\ell = {}$'.format(l))
            plt.legend()
            for mod_params in self.samples[np.random.randint(len(self.samples), size=100)]:
                mod_params[[5,9]] = 1e-6 * mod_params[[5,9]]
                ax1.plot(freq_array,d2nu_basu(freq_array,mod_params),c='#999999',alpha=0.1)
            best_params = self.mod_params_mcmc[:,0]
            best_params[[5,9]] = 1e-6 * best_params[[5,9]]
            ax1.plot(freq_array,d2nu_basu(freq_array,best_params),color=colors[3])
            ax1.set_xlabel(r'Frequency ($\mu$Hz)')
            ax1.set_ylabel(r'$\Delta_2 \nu$ $(\mu Hz)$')
            ax2 = fig_result.add_subplot(gs[2,0])
            for l in np.arange(3):
                i_l = self.data.l == l
                ax2.errorbar(self.data.freq[i_l],self.data.d2nu[i_l]-d2nu_basu(self.data.freq[i_l],best_params),
                        yerr=self.data.err[i_l],fmt=markers[l],
                        mfc=colors[l],mec='none',ecolor='#c4c4c4',
                        label=r'$\ell = {}$'.format(l))
            ax2.set_xlabel(r'Frequency ($\mu$Hz)')
            ax2.set_ylabel(r'$\delta\Delta_2 \nu$ $(\mu Hz)$')
        # elif self.model == d2nu_houdek:
        elif self.model == rr010_const_amp:
            model_name = 'rr010_const_amp'
            # Log file
            logfile = open(directory+save_params["nameplate"]+'_'+model_name+'.log',"w")
            logfile.write('Glitch fitting with glitch_removal\n'+
                    'Glitch model: '+model_name+'\n'+
                    'c0 + c1*(freq - beta) + c2*(freq - gamma1)*(freq - gamma2) + ' +\
                    'amp*sin(4*pi*freq*T_CE+phi_CE)\n\n')
            logfile.write('Results\n'+'_____________\n'+\
                    'c0\t'+str('%10.4e' % self.mod_params_mcmc[0][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[0][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[0][1])+'\n'+\
                    'c1\t'+str('%10.4e' % self.mod_params_mcmc[1][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[1][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[1][1])+'\n'+\
                    'c2\t'+str('%10.4e' % self.mod_params_mcmc[2][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[2][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[2][1])+'\n'+\
                    'amp\t'+str('%10.4e' % self.mod_params_mcmc[3][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[3][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[3][1])+'\n'+\
                    'T_CE'+str('%10.2f' % self.mod_params_mcmc[4][0])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[4][2])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[4][1])+'\n'+\
                    'phi_CE'+str('%10.4f' % self.mod_params_mcmc[5][0])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[5][2])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[5][1])+'\n\n')
            logfile.write('emcee parameters\n'+'_____________\n'+\
                    'nwalkers\t'+str('%d' % emcee_params["nwalkers"])+'\n'+\
                    'nburns\t'+str('%d' % emcee_params["nburns"])+'\n'+\
                    'nsteps\t'+str('%d' % emcee_params["nsteps"])+'\n\n')
            logfile.write('fit parameters\n'+'_____________\n'+\
                    'freqref\t'+str('%10.2f' % fit_params["freqref"])+'\n'+\
                    'nsvd\t'+str('%d' % fit_params["nsvd"])+'\n'+\
                    'beta\t'+str("%10.2f" % fit_params["beta"])+'\n'+\
                    'gamma1\t'+str("%10.2f" % fit_params["gamma1"])+'\n'+\
                    'gamma2\t'+str("%10.2f" % fit_params["gamma2"])+'\n\n')
            logfile.write('star parameters\n'+'_____________\n'+\
                    'delta_nu\t'+str('%10.2f' % star_params["delta_nu"])+'\n'+\
                    'numax\t'+str('%10.2f' % star_params["numax"]))
            logfile.close()
            # Corner plot
            labels = ['$c_0$','$c_1$','$c_2$',\
                    '$A$',r'$T_{\rm CE}$',r'$\phi_{\rm CE}$']
            fig_corner = corner.corner(self.samples,labels=labels,quantiles=[0.16, 0.5, 0.84], show_titles=True)
            # Walkers position
            walkers_redux = np.percentile(self.sampler.chain, [1, 50, 99],axis=0)
            nsteps_tab = np.arange(1,emcee_params["nsteps"]+1)
            fig_walkers = plt.figure(figsize=(10,25))
            gs = gridspec.GridSpec(6,1)
            ax1 = fig_walkers.add_subplot(gs[0,0])
            ax2 = fig_walkers.add_subplot(gs[1,0])
            ax3 = fig_walkers.add_subplot(gs[2,0])
            ax4 = fig_walkers.add_subplot(gs[3,0])
            ax5 = fig_walkers.add_subplot(gs[4,0])
            ax6 = fig_walkers.add_subplot(gs[5,0])
            #
            ax1.fill_between(nsteps_tab,walkers_redux[0,:,0],walkers_redux[2,:,0],color='#b5b5b5',alpha=0.5)
            ax1.plot(nsteps_tab,walkers_redux[1,:,0],lw=2)
            ax1.set_ylabel(labels[0])
            #
            ax2.fill_between(nsteps_tab,walkers_redux[0,:,1],walkers_redux[2,:,1],color='#b5b5b5',alpha=0.5)
            ax2.plot(nsteps_tab,walkers_redux[1,:,1],lw=2)
            ax2.set_ylabel(labels[1])
            #
            ax3.fill_between(nsteps_tab,walkers_redux[0,:,2],walkers_redux[2,:,2],color='#b5b5b5',alpha=0.5)
            ax3.plot(nsteps_tab,walkers_redux[1,:,2],lw=2)
            ax3.set_ylabel(labels[2])
            #
            ax4.fill_between(nsteps_tab,walkers_redux[0,:,3],walkers_redux[2,:,3],color='#b5b5b5',alpha=0.5)
            ax4.plot(nsteps_tab,walkers_redux[1,:,3],lw=2)
            ax4.set_ylabel(labels[3])
            #
            ax5.fill_between(nsteps_tab,walkers_redux[0,:,4],walkers_redux[2,:,4],color='#b5b5b5',alpha=0.5)
            ax5.plot(nsteps_tab,walkers_redux[1,:,4],lw=2)
            ax5.set_ylabel(labels[4])
            #
            ax6.fill_between(nsteps_tab,walkers_redux[0,:,5],walkers_redux[2,:,5],color='#b5b5b5',alpha=0.5)
            ax6.plot(nsteps_tab,walkers_redux[1,:,5],lw=2)
            ax6.set_ylabel(labels[5])
            ax6.set_xlabel(r'$n_{\rm steps}$')
            # Results in the observed plane
            freq_array = np.linspace(self.data.freq.min(),self.data.freq.max(),100)
            fig_result = plt.figure()
            gs = gridspec.GridSpec(3,1)
            ax1 = fig_result.add_subplot(gs[:2,0])
            markers = ['o','^']
            for l in np.arange(2):
                i_l = self.data.l == l
                ax1.errorbar(self.data.freq[i_l],self.data.rr010[i_l],
                        yerr=self.data.err[i_l],fmt=markers[l],
                        mfc=colors[l],mec='none',ecolor='#c4c4c4',
                        label=r'$\ell = {}$'.format(l))
            plt.legend()
            for mod_params in self.samples[np.random.randint(len(self.samples), size=100)]:
                mod_params[4] = 1e-6 * mod_params[4]
                ax1.plot(freq_array,rr010_const_amp(freq_array,mod_params),c='#999999',alpha=0.1)
            best_params = self.mod_params_mcmc[:,0]
            best_params[4] = 1e-6 * best_params[4]
            phase_rr010_const_amp = lambda x, phi : rr010_const_amp(x,\
                    [best_params[0],best_params[1],best_params[2],best_params[3],\
                    best_params[4],phi])
            phase_opt, V = op.curve_fit(phase_rr010_const_amp,\
                    self.data.freq,self.data.rr010,bounds=(0,2*np.pi))
            best_params[5] = phase_opt
            ax1.plot(freq_array,rr010_const_amp(freq_array,best_params),color=colors[3])
            ax1.set_xlabel(r'Frequency ($\mu$Hz)')
            ax1.set_ylabel(r'${rr}_{010}$')
            ax2 = fig_result.add_subplot(gs[2,0])
            for l in np.arange(2):
                i_l = self.data.l == l
                ax2.errorbar(self.data.freq[i_l],self.data.rr010[i_l]-rr010_const_amp(self.data.freq[i_l],best_params),
                        yerr=self.data.err[i_l],fmt=markers[l],
                        mfc=colors[l],mec='none',ecolor='#c4c4c4',
                        label=r'$\ell = {}$'.format(l))
            ax2.set_xlabel(r'Frequency ($\mu$Hz)')
            ax2.set_ylabel(r'$\delta{rr}_{010}$')
        elif self.model == rr010_freqinv_amp:
            model_name = 'rr010_freqinv_amp'
            # Log file
            logfile = open(directory+save_params["nameplate"]+'_'+model_name+'.log',"w")
            logfile.write('Glitch fitting with glitch_removal\n'+
                    'Glitch model: '+model_name+'\n'+
                    'c0 + c1*(freq - beta) + c2*(freq - gamma1)*(freq - gamma2) + ' +\
                    'amp*(freqref/freq)*sin(4*pi*freq*T_CE+phi_CE)\n\n')
            logfile.write('Results\n'+'_____________\n'+\
                    'c0\t'+str('%10.4e' % self.mod_params_mcmc[0][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[0][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[0][1])+'\n'+\
                    'c1\t'+str('%10.4e' % self.mod_params_mcmc[1][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[1][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[1][1])+'\n'+\
                    'c2\t'+str('%10.4e' % self.mod_params_mcmc[2][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[2][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[2][1])+'\n'+\
                    'amp\t'+str('%10.4e' % self.mod_params_mcmc[3][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[3][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[3][1])+'\n'+\
                    'T_CE'+str('%10.2f' % self.mod_params_mcmc[4][0])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[4][2])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[4][1])+'\n'+\
                    'phi_CE'+str('%10.4f' % self.mod_params_mcmc[5][0])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[5][2])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[5][1])+'\n\n')
            logfile.write('emcee parameters\n'+'_____________\n'+\
                    'nwalkers\t'+str('%d' % emcee_params["nwalkers"])+'\n'+\
                    'nburns\t'+str('%d' % emcee_params["nburns"])+'\n'+\
                    'nsteps\t'+str('%d' % emcee_params["nsteps"])+'\n\n')
            logfile.write('fit parameters\n'+'_____________\n'+\
                    'freqref\t'+str('%10.2f' % fit_params["freqref"])+'\n'+\
                    'nsvd\t'+str('%d' % fit_params["nsvd"])+'\n'+\
                    'beta\t'+str("%10.2f" % fit_params["beta"])+'\n'+\
                    'gamma1\t'+str("%10.2f" % fit_params["gamma1"])+'\n'+\
                    'gamma2\t'+str("%10.2f" % fit_params["gamma2"])+'\n\n')
            logfile.write('star parameters\n'+'_____________\n'+\
                    'delta_nu\t'+str('%10.2f' % star_params["delta_nu"])+'\n'+\
                    'numax\t'+str('%10.2f' % star_params["numax"]))
            logfile.close()
            # Corner plot
            labels = ['$c_0$','$c_1$','$c_2$',\
                    '$A$',r'$T_{\rm CE}$',r'$\phi_{\rm CE}$']
            fig_corner = corner.corner(self.samples,labels=labels,quantiles=[0.16, 0.5, 0.84], show_titles=True)
            # Walkers position
            walkers_redux = np.percentile(self.sampler.chain, [1, 50, 99],axis=0)
            nsteps_tab = np.arange(1,emcee_params["nsteps"]+1)
            fig_walkers = plt.figure(figsize=(10,25))
            gs = gridspec.GridSpec(6,1)
            ax1 = fig_walkers.add_subplot(gs[0,0])
            ax2 = fig_walkers.add_subplot(gs[1,0])
            ax3 = fig_walkers.add_subplot(gs[2,0])
            ax4 = fig_walkers.add_subplot(gs[3,0])
            ax5 = fig_walkers.add_subplot(gs[4,0])
            ax6 = fig_walkers.add_subplot(gs[5,0])
            #
            ax1.fill_between(nsteps_tab,walkers_redux[0,:,0],walkers_redux[2,:,0],color='#b5b5b5',alpha=0.5)
            ax1.plot(nsteps_tab,walkers_redux[1,:,0],lw=2)
            ax1.set_ylabel(labels[0])
            #
            ax2.fill_between(nsteps_tab,walkers_redux[0,:,1],walkers_redux[2,:,1],color='#b5b5b5',alpha=0.5)
            ax2.plot(nsteps_tab,walkers_redux[1,:,1],lw=2)
            ax2.set_ylabel(labels[1])
            #
            ax3.fill_between(nsteps_tab,walkers_redux[0,:,2],walkers_redux[2,:,2],color='#b5b5b5',alpha=0.5)
            ax3.plot(nsteps_tab,walkers_redux[1,:,2],lw=2)
            ax3.set_ylabel(labels[2])
            #
            ax4.fill_between(nsteps_tab,walkers_redux[0,:,3],walkers_redux[2,:,3],color='#b5b5b5',alpha=0.5)
            ax4.plot(nsteps_tab,walkers_redux[1,:,3],lw=2)
            ax4.set_ylabel(labels[3])
            #
            ax5.fill_between(nsteps_tab,walkers_redux[0,:,4],walkers_redux[2,:,4],color='#b5b5b5',alpha=0.5)
            ax5.plot(nsteps_tab,walkers_redux[1,:,4],lw=2)
            ax5.set_ylabel(labels[4])
            #
            ax6.fill_between(nsteps_tab,walkers_redux[0,:,5],walkers_redux[2,:,5],color='#b5b5b5',alpha=0.5)
            ax6.plot(nsteps_tab,walkers_redux[1,:,5],lw=2)
            ax6.set_ylabel(labels[5])
            ax6.set_xlabel(r'$n_{\rm steps}$')
            # Results in the observed plane
            freq_array = np.linspace(self.data.freq.min(),self.data.freq.max(),100)
            fig_result = plt.figure()
            gs = gridspec.GridSpec(3,1)
            ax1 = fig_result.add_subplot(gs[:2,0])
            markers = ['o','^']
            for l in np.arange(2):
                i_l = self.data.l == l
                ax1.errorbar(self.data.freq[i_l],self.data.rr010[i_l],
                        yerr=self.data.err[i_l],fmt=markers[l],
                        mfc=colors[l],mec='none',ecolor='#c4c4c4',
                        label=r'$\ell = {}$'.format(l))
            plt.legend()
            for mod_params in self.samples[np.random.randint(len(self.samples), size=100)]:
                mod_params[4] = 1e-6 * mod_params[4]
                ax1.plot(freq_array,rr010_freqinv_amp(freq_array,mod_params),c='#999999',alpha=0.1)
            best_params = self.mod_params_mcmc[:,0]
            best_params[4] = 1e-6 * best_params[4]  # because mode frequencies are in muHz
            phase_rr010_freqinv_amp = lambda x, phi : rr010_freqinv_amp(x,\
                    [best_params[0],best_params[1],best_params[2],best_params[3],\
                    best_params[4],phi])
            phase_opt, V = op.curve_fit(phase_rr010_freqinv_amp,\
                    self.data.freq,self.data.rr010,bounds=(0,2*np.pi))
            best_params[5] = phase_opt
            ax1.plot(freq_array,rr010_freqinv_amp(freq_array,best_params),color=colors[3])
            ax1.set_xlabel(r'Frequency ($\mu$Hz)')
            ax1.set_ylabel(r'${rr}_{010}$')
            ax2 = fig_result.add_subplot(gs[2,0])
            for l in np.arange(2):
                i_l = self.data.l == l
                ax2.errorbar(self.data.freq[i_l],self.data.rr010[i_l]-rr010_freqinv_amp(self.data.freq[i_l],best_params),
                        yerr=self.data.err[i_l],fmt=markers[l],
                        mfc=colors[l],mec='none',ecolor='#c4c4c4',
                        label=r'$\ell = {}$'.format(l))
            ax2.set_xlabel(r'Frequency ($\mu$Hz)')
            ax2.set_ylabel(r'$\delta{rr}_{010}$')
        elif self.model == rr010_freqinvsq_amp:
            model_name = 'rr010_freqinvsq_amp'
            # Log file
            logfile = open(directory+save_params["nameplate"]+'_'+model_name+'.log',"w")
            logfile.write('Glitch fitting with glitch_removal\n'+
                    'Glitch model: '+model_name+'\n'+
                    'c0 + c1*(freq - beta) + c2*(freq - gamma1)*(freq - gamma2) + ' +\
                    'amp*(freqref/freq)**2*sin(4*pi*freq*T_CE+phi_CE)\n\n')
            logfile.write('Results\n'+'_____________\n'+\
                    'c0\t'+str('%10.4e' % self.mod_params_mcmc[0][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[0][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[0][1])+'\n'+\
                    'c1\t'+str('%10.4e' % self.mod_params_mcmc[1][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[1][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[1][1])+'\n'+\
                    'c2\t'+str('%10.4e' % self.mod_params_mcmc[2][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[2][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[2][1])+'\n'+\
                    'amp\t'+str('%10.4e' % self.mod_params_mcmc[3][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[3][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[3][1])+'\n'+\
                    'T_CE'+str('%10.2f' % self.mod_params_mcmc[4][0])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[4][2])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[4][1])+'\n'+\
                    'phi_CE'+str('%10.4f' % self.mod_params_mcmc[5][0])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[5][2])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[5][1])+'\n\n')
            logfile.write('emcee parameters\n'+'_____________\n'+\
                    'nwalkers\t'+str('%d' % emcee_params["nwalkers"])+'\n'+\
                    'nburns\t'+str('%d' % emcee_params["nburns"])+'\n'+\
                    'nsteps\t'+str('%d' % emcee_params["nsteps"])+'\n\n')
            logfile.write('fit parameters\n'+'_____________\n'+\
                    'freqref\t'+str('%10.2f' % fit_params["freqref"])+'\n'+\
                    'nsvd\t'+str('%d' % fit_params["nsvd"])+'\n'+\
                    'beta\t'+str("%10.2f" % fit_params["beta"])+'\n'+\
                    'gamma1\t'+str("%10.2f" % fit_params["gamma1"])+'\n'+\
                    'gamma2\t'+str("%10.2f" % fit_params["gamma2"])+'\n\n')
            logfile.write('star parameters\n'+'_____________\n'+\
                    'delta_nu\t'+str('%10.2f' % star_params["delta_nu"])+'\n'+\
                    'numax\t'+str('%10.2f' % star_params["numax"]))
            logfile.close()
            # Corner plot
            labels = ['$c_0$','$c_1$','$c_2$',\
                    '$A$',r'$T_{\rm CE}$',r'$\phi_{\rm CE}$']
            fig_corner = corner.corner(self.samples,labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
            # Walkers position
            walkers_redux = np.percentile(self.sampler.chain, [1, 50, 99],axis=0)
            nsteps_tab = np.arange(1,emcee_params["nsteps"]+1)
            fig_walkers = plt.figure(figsize=(10,25))
            gs = gridspec.GridSpec(6,1)
            ax1 = fig_walkers.add_subplot(gs[0,0])
            ax2 = fig_walkers.add_subplot(gs[1,0])
            ax3 = fig_walkers.add_subplot(gs[2,0])
            ax4 = fig_walkers.add_subplot(gs[3,0])
            ax5 = fig_walkers.add_subplot(gs[4,0])
            ax6 = fig_walkers.add_subplot(gs[5,0])
            #
            ax1.fill_between(nsteps_tab,walkers_redux[0,:,0],walkers_redux[2,:,0],color='#b5b5b5',alpha=0.5)
            ax1.plot(nsteps_tab,walkers_redux[1,:,0],lw=2)
            ax1.set_ylabel(labels[0])
            #
            ax2.fill_between(nsteps_tab,walkers_redux[0,:,1],walkers_redux[2,:,1],color='#b5b5b5',alpha=0.5)
            ax2.plot(nsteps_tab,walkers_redux[1,:,1],lw=2)
            ax2.set_ylabel(labels[1])
            #
            ax3.fill_between(nsteps_tab,walkers_redux[0,:,2],walkers_redux[2,:,2],color='#b5b5b5',alpha=0.5)
            ax3.plot(nsteps_tab,walkers_redux[1,:,2],lw=2)
            ax3.set_ylabel(labels[2])
            #
            ax4.fill_between(nsteps_tab,walkers_redux[0,:,3],walkers_redux[2,:,3],color='#b5b5b5',alpha=0.5)
            ax4.plot(nsteps_tab,walkers_redux[1,:,3],lw=2)
            ax4.set_ylabel(labels[3])
            #
            ax5.fill_between(nsteps_tab,walkers_redux[0,:,4],walkers_redux[2,:,4],color='#b5b5b5',alpha=0.5)
            ax5.plot(nsteps_tab,walkers_redux[1,:,4],lw=2)
            ax5.set_ylabel(labels[4])
            #
            ax6.fill_between(nsteps_tab,walkers_redux[0,:,5],walkers_redux[2,:,5],color='#b5b5b5',alpha=0.5)
            ax6.plot(nsteps_tab,walkers_redux[1,:,5],lw=2)
            ax6.set_ylabel(labels[5])
            ax6.set_xlabel(r'$n_{\rm steps}$')
            # Results in the observed plane
            freq_array = np.linspace(self.data.freq.min(),self.data.freq.max(),100)
            fig_result = plt.figure()
            gs = gridspec.GridSpec(3,1)
            ax1 = fig_result.add_subplot(gs[:2,0])
            markers = ['o','^']
            for l in np.arange(2):
                i_l = self.data.l == l
                ax1.errorbar(self.data.freq[i_l],self.data.rr010[i_l],
                        yerr=self.data.err[i_l],fmt=markers[l],
                        mfc=colors[l],mec='none',ecolor='#c4c4c4',
                        label=r'$\ell = {}$'.format(l))
            plt.legend()
            for mod_params in self.samples[np.random.randint(len(self.samples), size=100)]:
                mod_params[4] = 1e-6 * mod_params[4]
                ax1.plot(freq_array,rr010_freqinvsq_amp(freq_array,mod_params),c='#999999',alpha=0.1)
            best_params = self.mod_params_mcmc[:,0]
            best_params[4] = 1e-6 * best_params[4]
            phase_rr010_freqinvsq_amp = lambda x, phi : rr010_freqinvsq_amp(x,\
                    [best_params[0],best_params[1],best_params[2],best_params[3],\
                    best_params[4],phi])
            phase_opt, V = op.curve_fit(phase_rr010_freqinvsq_amp,\
                    self.data.freq,self.data.rr010,bounds=(0,2*np.pi))
            best_params[5] = phase_opt
            ax1.plot(freq_array,rr010_freqinvsq_amp(freq_array,best_params),color=colors[3])
            ax1.set_xlabel(r'Frequency ($\mu$Hz)')
            ax1.set_ylabel(r'${rr}_{010}$')
            ax2 = fig_result.add_subplot(gs[2,0])
            for l in np.arange(2):
                i_l = self.data.l == l
                ax2.errorbar(self.data.freq[i_l],self.data.rr010[i_l]-rr010_freqinvsq_amp(self.data.freq[i_l],best_params),
                        yerr=self.data.err[i_l],fmt=markers[l],
                        mfc=colors[l],mec='none',ecolor='#c4c4c4',
                        label=r'$\ell = {}$'.format(l))
            ax2.set_xlabel(r'Frequency ($\mu$Hz)')
            ax2.set_ylabel(r'$\delta{rr}_{010}$')
        elif self.model == rr010_freqinvpoly_amp:
            model_name = 'rr010_freqinvpoly_amp'
            # Log file
            logfile = open(directory+save_params["nameplate"]+'_'+model_name+'.log',"w")
            logfile.write('Glitch fitting with glitch_removal\n'+
                    'Glitch model: '+model_name+'\n'+
                    'c0 + c1*(freq - beta) + c2*(freq - gamma1)*(freq - gamma2) + ' +\
                    '(amp0*(freqref/freq)+amp1*(freqref/freq)**2) * '+
                    'sin(4*pi*freq*T_CE+phi_CE)'+'\n\n')
            logfile.write('Results\n'+'_____________\n'+\
                    'c0\t'+str('%10.4e' % self.mod_params_mcmc[0][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[0][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[0][1])+'\n'+\
                    'c1\t'+str('%10.4e' % self.mod_params_mcmc[1][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[1][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[1][1])+'\n'+\
                    'c2\t'+str('%10.4e' % self.mod_params_mcmc[2][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[2][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[2][1])+'\n'+\
                    'amp0\t'+str('%10.4e' % self.mod_params_mcmc[3][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[3][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[3][1])+'\n'+\
                    'amp1\t'+str('%10.4e' % self.mod_params_mcmc[4][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[4][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[4][1])+'\n'+\
                    'T_CE'+str('%10.2f' % self.mod_params_mcmc[5][0])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[5][2])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[5][1])+'\n'+\
                    'phi1'+str('%10.4f' % self.mod_params_mcmc[6][0])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[6][2])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[6][1])+'\n\n')
            logfile.write('emcee parameters\n'+'_____________\n'+\
                    'nwalkers\t'+str('%d' % emcee_params["nwalkers"])+'\n'+\
                    'nburns\t'+str('%d' % emcee_params["nburns"])+'\n'+\
                    'nsteps\t'+str('%d' % emcee_params["nsteps"])+'\n\n')
            logfile.write('fit parameters\n'+'_____________\n'+\
                    'freqref\t'+str('%10.2f' % fit_params["freqref"])+'\n'+\
                    'nsvd\t'+str('%d' % fit_params["nsvd"])+'\n'+\
                    'beta\t'+str("%10.2f" % fit_params["beta"])+'\n'+\
                    'gamma1\t'+str("%10.2f" % fit_params["gamma1"])+'\n'+\
                    'gamma2\t'+str("%10.2f" % fit_params["gamma2"])+'\n\n')
            logfile.write('star parameters\n'+'_____________\n'+\
                    'delta_nu\t'+str('%10.2f' % star_params["delta_nu"])+'\n'+\
                    'numax\t'+str('%10.2f' % star_params["numax"]))
            logfile.close()
            # Corner plot
            labels = ['$c_0$','$c_1$','$c_2$',\
                    '$A_0$','$A_1$',r'$T_{\rm CE}$',r'$\phi_{\rm CE}$']
            fig_corner = corner.corner(self.samples,labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
            # Walkers position
            walkers_redux = np.percentile(self.sampler.chain, [1, 50, 99],axis=0)
            nsteps_tab = np.arange(1,emcee_params["nsteps"]+1)
            fig_walkers = plt.figure(figsize=(10,25))
            gs = gridspec.GridSpec(7,1)
            ax1 = fig_walkers.add_subplot(gs[0,0])
            ax2 = fig_walkers.add_subplot(gs[1,0])
            ax3 = fig_walkers.add_subplot(gs[2,0])
            ax4 = fig_walkers.add_subplot(gs[3,0])
            ax5 = fig_walkers.add_subplot(gs[4,0])
            ax6 = fig_walkers.add_subplot(gs[5,0])
            ax7 = fig_walkers.add_subplot(gs[6,0])
            #
            ax1.fill_between(nsteps_tab,walkers_redux[0,:,0],walkers_redux[2,:,0],color='#b5b5b5',alpha=0.5)
            ax1.plot(nsteps_tab,walkers_redux[1,:,0],lw=2)
            ax1.set_ylabel(labels[0])
            #
            ax2.fill_between(nsteps_tab,walkers_redux[0,:,1],walkers_redux[2,:,1],color='#b5b5b5',alpha=0.5)
            ax2.plot(nsteps_tab,walkers_redux[1,:,1],lw=2)
            ax2.set_ylabel(labels[1])
            #
            ax3.fill_between(nsteps_tab,walkers_redux[0,:,2],walkers_redux[2,:,2],color='#b5b5b5',alpha=0.5)
            ax3.plot(nsteps_tab,walkers_redux[1,:,2],lw=2)
            ax3.set_ylabel(labels[2])
            #
            ax4.fill_between(nsteps_tab,walkers_redux[0,:,3],walkers_redux[2,:,3],color='#b5b5b5',alpha=0.5)
            ax4.plot(nsteps_tab,walkers_redux[1,:,3],lw=2)
            ax4.set_ylabel(labels[3])
            #
            ax5.fill_between(nsteps_tab,walkers_redux[0,:,4],walkers_redux[2,:,4],color='#b5b5b5',alpha=0.5)
            ax5.plot(nsteps_tab,walkers_redux[1,:,4],lw=2)
            ax5.set_ylabel(labels[4])
            #
            ax6.fill_between(nsteps_tab,walkers_redux[0,:,5],walkers_redux[2,:,5],color='#b5b5b5',alpha=0.5)
            ax6.plot(nsteps_tab,walkers_redux[1,:,5],lw=2)
            ax6.set_ylabel(labels[5])
            #
            ax7.fill_between(nsteps_tab,walkers_redux[0,:,6],walkers_redux[2,:,6],color='#b5b5b5',alpha=0.5)
            ax7.plot(nsteps_tab,walkers_redux[1,:,6],lw=2)
            ax7.set_ylabel(labels[6])
            ax7.set_xlabel(r'$n_{\rm steps}$')
            # Results in the observed plane
            freq_array = np.linspace(self.data.freq.min(),self.data.freq.max(),100)
            fig_result = plt.figure()
            gs = gridspec.GridSpec(3,1)
            ax1 = fig_result.add_subplot(gs[:2,0])
            markers = ['o','^']
            for l in np.arange(2):
                i_l = self.data.l == l
                ax1.errorbar(self.data.freq[i_l],self.data.rr010[i_l],
                        yerr=self.data.err[i_l],fmt=markers[l],
                        mfc=colors[l],mec='none',ecolor='#c4c4c4',
                        label=r'$\ell = {}$'.format(l))
            plt.legend()
            for mod_params in self.samples[np.random.randint(len(self.samples), size=100)]:
                mod_params[5] = 1e-6 * mod_params[5]
                ax1.plot(freq_array,rr010_freqinvpoly_amp(freq_array,mod_params),c='#999999',alpha=0.1)
            best_params = self.mod_params_mcmc[:,0]
            best_params[5] = 1e-6 * best_params[5]
            phase_rr010_freqinvpoly_amp = lambda x, phi : rr010_freqinvpoly_amp(x,\
                    [best_params[0],best_params[1],best_params[2],best_params[3],\
                    best_params[4],best_params[5],phi])
            phase_opt, V = op.curve_fit(phase_rr010_freqinvpoly_amp,\
                    self.data.freq,self.data.rr010,bounds=(0,2*np.pi))
            best_params[6] = phase_opt
            ax1.plot(freq_array,rr010_freqinvpoly_amp(freq_array,best_params),color=colors[3])
            ax1.set_xlabel(r'Frequency ($\mu$Hz)')
            ax1.set_ylabel(r'${rr}_{010}$')
            ax2 = fig_result.add_subplot(gs[2,0])
            for l in np.arange(2):
                i_l = self.data.l == l
                ax2.errorbar(self.data.freq[i_l],self.data.rr010[i_l]-rr010_freqinvpoly_amp(self.data.freq[i_l],best_params),
                        yerr=self.data.err[i_l],fmt=markers[l],
                        mfc=colors[l],mec='none',ecolor='#c4c4c4',
                        label=r'$\ell = {}$'.format(l))
            ax2.set_xlabel(r'Frequency ($\mu$Hz)')
            ax2.set_ylabel(r'$\delta{rr}_{010}$')
        elif self.model == rr010_freqinvsq_amp_polyper:
            model_name = 'rr010_freqinvsq_amp_polyper'
            # Log file
            logfile = open(directory+save_params["nameplate"]+'_'+model_name+'.log',"w")
            logfile.write('Glitch fitting with glitch_removal\n'+
                    'Glitch model: '+model_name+'\n'+
                    'c0 + c1*(freq - beta) + c2*(freq - gamma1)*(freq - gamma2) + ' +\
                    'amp0*(freqref/freq)**2*sin(4*pi*freq*T0+phi0) + '+\
                    'amp1*(freqref/freq)**2*sin(4*pi*freq*T1+phi1) + '+\
                    '\n\n')
            logfile.write('Results\n'+'_____________\n'+\
                    'c0\t'+str('%10.4e' % self.mod_params_mcmc[0][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[0][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[0][1])+'\n'+\
                    'c1\t'+str('%10.4e' % self.mod_params_mcmc[1][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[1][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[1][1])+'\n'+\
                    'c2\t'+str('%10.4e' % self.mod_params_mcmc[2][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[2][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[2][1])+'\n'+\
                    'amp0\t'+str('%10.4e' % self.mod_params_mcmc[3][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[3][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[3][1])+'\n'+\
                    'T0'+str('%10.2f' % self.mod_params_mcmc[4][0])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[4][2])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[4][1])+'\n'+\
                    'phi0'+str('%10.4f' % self.mod_params_mcmc[5][0])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[5][2])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[5][1])+'\n'+\
                    'amp1\t'+str('%10.4e' % self.mod_params_mcmc[6][0])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[6][2])+'\t'+\
                    str('%10.4e' % self.mod_params_mcmc[6][1])+'\n'+\
                    'T1'+str('%10.2f' % self.mod_params_mcmc[7][0])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[7][2])+'\t'+\
                    str('%10.2f' % self.mod_params_mcmc[7][1])+'\n'+\
                    'phi1'+str('%10.4f' % self.mod_params_mcmc[8][0])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[8][2])+'\t'+\
                    str('%10.4f' % self.mod_params_mcmc[8][1])+'\n\n')
            logfile.write('emcee parameters\n'+'_____________\n'+\
                    'nwalkers\t'+str('%d' % emcee_params["nwalkers"])+'\n'+\
                    'nburns\t'+str('%d' % emcee_params["nburns"])+'\n'+\
                    'nsteps\t'+str('%d' % emcee_params["nsteps"])+'\n\n')
            logfile.write('fit parameters\n'+'_____________\n'+\
                    'freqref\t'+str('%10.2f' % fit_params["freqref"])+'\n'+\
                    'nsvd\t'+str('%d' % fit_params["nsvd"])+'\n'+\
                    'beta\t'+str("%10.2f" % fit_params["beta"])+'\n'+\
                    'gamma1\t'+str("%10.2f" % fit_params["gamma1"])+'\n'+\
                    'gamma2\t'+str("%10.2f" % fit_params["gamma2"])+'\n\n')
            logfile.write('star parameters\n'+'_____________\n'+\
                    'delta_nu\t'+str('%10.2f' % star_params["delta_nu"])+'\n'+\
                    'numax\t'+str('%10.2f' % star_params["numax"]))
            logfile.close()
            # Corner plot
            labels = ['$c_0$','$c_1$','$c_2$',\
                    '$A_0$',r'$T_0$',r'$\phi_0$','$A_1$',r'$T_1$',r'$\phi_1$']
            fig_corner = corner.corner(self.samples,labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
            # Walkers position
            walkers_redux = np.percentile(self.sampler.chain, [1, 50, 99],axis=0)
            nsteps_tab = np.arange(1,emcee_params["nsteps"]+1)
            fig_walkers = plt.figure(figsize=(10,25))
            gs = gridspec.GridSpec(9,1)
            ax1 = fig_walkers.add_subplot(gs[0,0])
            ax2 = fig_walkers.add_subplot(gs[1,0])
            ax3 = fig_walkers.add_subplot(gs[2,0])
            ax4 = fig_walkers.add_subplot(gs[3,0])
            ax5 = fig_walkers.add_subplot(gs[4,0])
            ax6 = fig_walkers.add_subplot(gs[5,0])
            ax7 = fig_walkers.add_subplot(gs[6,0])
            ax8 = fig_walkers.add_subplot(gs[7,0])
            ax9 = fig_walkers.add_subplot(gs[8,0])
            #
            ax1.fill_between(nsteps_tab,walkers_redux[0,:,0],walkers_redux[2,:,0],color='#b5b5b5',alpha=0.5)
            ax1.plot(nsteps_tab,walkers_redux[1,:,0],lw=2)
            ax1.set_ylabel(labels[0])
            #
            ax2.fill_between(nsteps_tab,walkers_redux[0,:,1],walkers_redux[2,:,1],color='#b5b5b5',alpha=0.5)
            ax2.plot(nsteps_tab,walkers_redux[1,:,1],lw=2)
            ax2.set_ylabel(labels[1])
            #
            ax3.fill_between(nsteps_tab,walkers_redux[0,:,2],walkers_redux[2,:,2],color='#b5b5b5',alpha=0.5)
            ax3.plot(nsteps_tab,walkers_redux[1,:,2],lw=2)
            ax3.set_ylabel(labels[2])
            #
            ax4.fill_between(nsteps_tab,walkers_redux[0,:,3],walkers_redux[2,:,3],color='#b5b5b5',alpha=0.5)
            ax4.plot(nsteps_tab,walkers_redux[1,:,3],lw=2)
            ax4.set_ylabel(labels[3])
            #
            ax5.fill_between(nsteps_tab,walkers_redux[0,:,4],walkers_redux[2,:,4],color='#b5b5b5',alpha=0.5)
            ax5.plot(nsteps_tab,walkers_redux[1,:,4],lw=2)
            ax5.set_ylabel(labels[4])
            #
            ax6.fill_between(nsteps_tab,walkers_redux[0,:,5],walkers_redux[2,:,5],color='#b5b5b5',alpha=0.5)
            ax6.plot(nsteps_tab,walkers_redux[1,:,5],lw=2)
            ax6.set_ylabel(labels[5])
            #
            ax7.fill_between(nsteps_tab,walkers_redux[0,:,6],walkers_redux[2,:,6],color='#b5b5b5',alpha=0.5)
            ax7.plot(nsteps_tab,walkers_redux[1,:,6],lw=2)
            ax7.set_ylabel(labels[6])
            #
            ax8.fill_between(nsteps_tab,walkers_redux[0,:,7],walkers_redux[2,:,7],color='#b5b5b5',alpha=0.5)
            ax8.plot(nsteps_tab,walkers_redux[1,:,7],lw=2)
            ax8.set_ylabel(labels[7])
            #
            ax9.fill_between(nsteps_tab,walkers_redux[0,:,8],walkers_redux[2,:,8],color='#b5b5b5',alpha=0.5)
            ax9.plot(nsteps_tab,walkers_redux[1,:,8],lw=2)
            ax9.set_ylabel(labels[8])
            ax9.set_xlabel(r'$n_{\rm steps}$')
            # Results in the observed plane
            freq_array = np.linspace(self.data.freq.min(),self.data.freq.max(),100)
            fig_result = plt.figure()
            gs = gridspec.GridSpec(3,1)
            ax1 = fig_result.add_subplot(gs[:2,0])
            markers = ['o','^']
            for l in np.arange(2):
                i_l = self.data.l == l
                ax1.errorbar(self.data.freq[i_l],self.data.rr010[i_l],
                        yerr=self.data.err[i_l],fmt=markers[l],
                        mfc=colors[l],mec='none',ecolor='#c4c4c4',
                        label=r'$\ell = {}$'.format(l))
            plt.legend()
            for mod_params in self.samples[np.random.randint(len(self.samples), size=100)]:
                mod_params[[4,7]] = 1e-6 * mod_params[[4,7]]
                ax1.plot(freq_array,rr010_freqinvsq_amp_polyper(freq_array,mod_params),c='#999999',alpha=0.1)
            best_params = self.mod_params_mcmc[:,0]
            best_params[[4,7]] = 1e-6 * best_params[[4,7]]
            phase_rr010_freqinvsq_amp_polyper = lambda x, phi, psi : rr010_freqinvsq_amp_polyper(x,\
                    [best_params[0],best_params[1],best_params[2],best_params[3],\
                    best_params[4],phi,best_params[6],best_params[7],psi])
            phase_opt, V = op.curve_fit(phase_rr010_freqinvsq_amp_polyper,\
                    self.data.freq,self.data.rr010,bounds=((0,0),(2*np.pi,2*np.pi)))
            best_params[[5,8]] = phase_opt
            ax1.plot(freq_array,rr010_freqinvsq_amp_polyper(freq_array,best_params),color=colors[3])
            ax1.set_xlabel(r'Frequency ($\mu$Hz)')
            ax1.set_ylabel(r'${rr}_{010}$')
            ax2 = fig_result.add_subplot(gs[2,0])
            for l in np.arange(2):
                i_l = self.data.l == l
                ax2.errorbar(self.data.freq[i_l],self.data.rr010[i_l]-rr010_freqinvsq_amp_polyper(self.data.freq[i_l],best_params),
                        yerr=self.data.err[i_l],fmt=markers[l],
                        mfc=colors[l],mec='none',ecolor='#c4c4c4',
                        label=r'$\ell = {}$'.format(l))
            ax2.set_xlabel(r'Frequency ($\mu$Hz)')
            ax2.set_ylabel(r'$\delta{rr}_{010}$')
        fig_corner.savefig(directory+save_params["nameplate"]+'_'+model_name+'_'+'corner.png')
        fig_walkers.savefig(directory+save_params["nameplate"]+'_'+model_name+'_'+'walkers.png')
        fig_result.savefig(directory+save_params["nameplate"]+'_'+model_name+'_'+'result.png')
        if 'samples' in what2save:
            path_samples = directory+save_params["nameplate"]+'_'+model_name+'_'+'samples.dat'
            np.savetxt(path_samples,self.samples,fmt="%10.5e")


# List of models to represent glitches in different seismic indicators


def d2nu_verma(x,args):
    """ Functional form used by Verma et al. 2017 to model second
        differences D2nu. """
    return args[0] + args[1]*x + \
            args[2]*(fit_params["freqref"]/x)**2*np.sin(4*np.pi*x*args[3]+args[4]) + \
            args[5]*(x/fit_params["freqref"])*np.exp(-args[6]*x**2)*np.sin(4*np.pi*x*args[7]+args[8])

def d2nu_mazumdar(x,args):
    """ Functional form used by Mazumdar et al. 2014 to model second
        differences D2nu (Method A). """
    return args[0] + \
            args[1]*(fit_params["freqref"]/x)**2*np.sin(4*np.pi*x*args[2]+args[3]) + \
            args[4]*(x/fit_params["freqref"])*np.exp(-args[5]*x**2)*np.sin(4*np.pi*x*args[6]+args[7])

def d2nu_basu(x,args):
    """ Functional form used by Basu et al. 2004 to model second
        differences D2nu. """
    return args[0] + args[1]*x + args[2]/x**2 + \
            (args[3] + args[4]/x**2)*np.sin(4*np.pi*x*args[5]+args[6]) + \
            (args[7] + args[8]/x**2)*np.sin(4*np.pi*x*args[9]+args[10])

def d2nu_houdek(x,args):
    """ Functional form used by Houdek & Gough 2007 to model second
        differences D2nu. """
    return

def rr010_const_amp(x,args):
    """ Functional form to model ratios rr010. Amplitude of the glitch
        signature is assumed to be independent of frequency. """
    return args[0] + args[1]*(x-fit_params["beta"]) + \
            args[2]*(x-fit_params["gamma1"])*(x-fit_params["gamma2"]) +\
            args[3]*np.sin(4*np.pi*x*args[4]+args[5])

def rr010_freqinv_amp(x,args):
    """ Functional form to model ratios rr010. Amplitude of the glitch
        signature is prop. to 1/freq. """
    return args[0] + args[1]*(x-fit_params["beta"]) + \
            args[2]*(x-fit_params["gamma1"])*(x-fit_params["gamma2"]) +\
            (args[3]*fit_params["freqref"]/x)*np.sin(4*np.pi*x*args[4]+args[5])

def rr010_freqinvsq_amp(x,args):
    """ Functional form to model ratios rr010. Amplitude of the glitch
        signature is prop. to 1/freq^2. """
    return args[0] + args[1]*(x-fit_params["beta"]) + \
            args[2]*(x-fit_params["gamma1"])*(x-fit_params["gamma2"]) +\
            (args[3]*fit_params["freqref"]**2/x**2)*np.sin(4*np.pi*x*args[4]+args[5])

def rr010_freqinvpoly_amp(x,args):
    """ Functional form to model ratios rr010. Amplitude of the glitch
        signature has two components : (1/freq) + (1/freq**2). """
    return args[0] + args[1]*(x-fit_params["beta"]) + \
            args[2]*(x-fit_params["gamma1"])*(x-fit_params["gamma2"]) +\
            (args[3]*fit_params["freqref"]/x + \
            args[4]*fit_params["freqref"]**2/x**2) * np.sin(4*np.pi*x*args[5]+args[6])

def rr010_freqinvsq_amp_polyper(x,args):
    """  Functional form to model ratios rr010. Sum of two periodic signatures.
        Amplitude of the glitch signatures is prop to (1/freq**2)."""
    return args[0] + args[1]*(x-fit_params["beta"]) + \
            args[2]*(x-fit_params["gamma1"])*(x-fit_params["gamma2"]) +\
            (args[3]*fit_params["freqref"]**2/x**2)*np.sin(4*np.pi*x*args[4]+args[5]) + \
            (args[6]*fit_params["freqref"]**2/x**2)*np.sin(4*np.pi*x*args[7]+args[8])


# Update parameters in params.py

def update_params(model,data):
    """ Adapts parameters in params.py to suit the chosen GlitchModel. """
    # Update emcee_params to suit the chosen glitch model
    if model == d2nu_verma:
        emcee_params["ndim"] = 9
    elif model == d2nu_mazumdar:
        emcee_params["ndim"] = 8
    elif model == d2nu_basu:
        emcee_params["ndim"] = 11
    # elif model == d2nu_houdek:
    #     emcee_params["ndim"] = 0
    elif model == rr010_const_amp or model == rr010_freqinv_amp or \
            model == rr010_freqinvsq_amp:
        emcee_params["ndim"] = 6
    elif model == rr010_freqinvpoly_amp:
        emcee_params["ndim"] = 7
    elif model == rr010_freqinvsq_amp_polyper:
        emcee_params["ndim"] = 9
    # Check if star_params are consistent with data
    # numax
    numax_backup = np.median(data.freq)
    if not (0.85*numax_backup <= star_params["numax"] <= 1.15*numax_backup):
        star_params["numax"] = numax_backup
        print(f"""Warning: numax in the star_params dictionary is more than 15%
                off the median mode frequency. numax has been set to
                {star_params["numax"]}. Please consider updating it to the
                right value in params.py""")
    # delta_nu
    delta_nu_backup = 0.0
    for l in np.arange(data.l.min(),data.l.max()+1):
        delta_nu_backup = delta_nu_backup + \
                np.median(np.abs(np.sort(data.freq[data.l == l][1:])-\
                np.sort(data.freq[data.l==l][:-1])))
    delta_nu_backup = delta_nu_backup / (data.l.max() - data.l.min() + 1)
    if not (0.9*delta_nu_backup <= star_params["delta_nu"] <= 1.1*delta_nu_backup):
        star_params["delta_nu"] = delta_nu_backup
        print(f"""Warning: delta_nu in the star_params dictionary is more
                than 10% off the mean large separation computed from the
                frequency set. delta_nu has been set to
                {star_params["delta_nu"]}. Please consider updating it to the
                right value in params.py""")
    # Reinitialise the values of beta, gamma1 and gamma2 in fit_params
    fit_params["beta"], fit_params["gamma1"], fit_params["gamma2"] = 0, 0, 0

# Likelihood functions

def lnlikelihood_rr010(args,model,data):
    model_eval = model(data.freq,args)
    return -0.5*np.dot(np.transpose(data.rr010-model_eval),np.dot(data.inv_matcov,data.rr010-model_eval))

def lnlikelihood_d2nu(args,model,data):
    model_eval = model(data.freq,args)
    return -0.5*np.dot(np.transpose(data.d2nu-model_eval),np.dot(data.inv_matcov,data.d2nu-model_eval))

# Priors

def lnprior_d2nu_verma(params,bds):
    a0, a1, b0, tau_CE, phi_CE, c0, c2, tau_He, phi_He = params
    if bds[0][0] < a0 < bds[0][1] and \
            bds[1][0] < a1 < bds[1][1] and \
            bds[2][0] < b0 < bds[2][1] and \
            bds[3][0] < tau_CE < bds[3][1] and \
            bds[4][0] < phi_CE < bds[4][1] and \
            bds[5][0] < c0 < bds[5][1] and \
            bds[6][0] < c2 < bds[6][1] and \
            bds[7][0] < tau_He < bds[7][1] and \
            bds[8][0] < phi_He < bds[8][1]:
        return 0.0
    return -np.inf

def lnprior_d2nu_mazumdar(params,bds):
    a0, b0, tau_CE, phi_CE, c0, c2, tau_He, phi_He = params
    if bds[0][0] < a0 < bds[0][1] and \
            bds[1][0] < b0 < bds[1][1] and \
            bds[2][0] < tau_CE < bds[2][1] and \
            bds[3][0] < phi_CE < bds[3][1] and \
            bds[4][0] < c0 < bds[4][1] and \
            bds[5][0] < c2 < bds[5][1] and \
            bds[6][0] < tau_He < bds[6][1] and \
            bds[7][0] < phi_He < bds[7][1]:
        return 0.0
    return -np.inf

def lnprior_d2nu_basu(params,bds):
    a1, a2, a3, b1, b2, tau_He, phi_He, c1, c2, tau_CE, phi_CE = params
    if bds[0][0] < a1 < bds[0][1] and \
            bds[1][0] < a2 < bds[1][1] and \
            bds[2][0] < a3 < bds[2][1] and \
            bds[3][0] < b1 < bds[3][1] and \
            bds[4][0] < b2 < bds[4][1] and \
            bds[5][0] < tau_He < bds[5][1] and \
            bds[6][0] < phi_He < bds[6][1] and \
            bds[7][0] < c1 < bds[7][1] and \
            bds[8][0] < c2 < bds[8][1] and \
            bds[9][0] < tau_CE < bds[9][1] and \
            bds[10][0] < phi_CE < bds[10][1]:
        return 0.0
    return - np.inf

# def lnprior_d2nu_houdek(params,bds):
#     return

def lnprior_rr010_const_amp(params,bds):
    c0, c1, c2, amp, T, phi = params
    if bds[0][0] < c0 < bds[0][1] and \
            bds[1][0] < c1 < bds[1][1] and \
            bds[2][0] < c2 < bds[2][1] and \
            bds[3][0] < amp < bds[3][1] and \
            bds[4][0] < T < bds[4][1] and \
            bds[5][0] < phi < bds[5][1]:
        return 0.0
    return -np.inf

def lnprior_rr010_freqinv_amp(params,bds):
    c0, c1, c2, amp, T, phi = params
    if bds[0][0] < c0 < bds[0][1] and \
            bds[1][0] < c1 < bds[1][1] and \
            bds[2][0] < c2 < bds[2][1] and \
            bds[3][0] < amp < bds[3][1] and \
            bds[4][0] < T < bds[4][1] and \
            bds[5][0] < phi < bds[5][1]:
        return 0.0
    return -np.inf

def lnprior_rr010_freqinvsq_amp(params,bds):
    c0, c1, c2, amp, T, phi = params
    if bds[0][0] < c0 < bds[0][1] and \
            bds[1][0] < c1 < bds[1][1] and \
            bds[2][0] < c2 < bds[2][1] and \
            bds[3][0] < amp < bds[3][1] and \
            bds[4][0] < T < bds[4][1] and \
            bds[5][0] < phi < bds[5][1]:
        return 0.0
    return -np.inf

def lnprior_rr010_freqinvpoly_amp(params,bds):
        c0, c1, c2, amp0, amp1, T, phi = params
        if bds[0][0] < c0 < bds[0][1] and \
                bds[1][0] < c1 < bds[1][1] and \
                bds[2][0] < c2 < bds[2][1] and \
                bds[3][0] < amp0 < bds[3][1] and \
                bds[4][0] < amp1 < bds[4][1] and \
                bds[5][0] < T < bds[5][1] and \
                bds[6][0] < phi < bds[6][1]:
            return 0.0
        return -np.inf

def lnprior_rr010_freqinvsq_amp_polyper(params,bds):
    c0, c1, c2, amp0, T0, phi0, amp1, T1, phi1 = params
    if bds[0][0] < c0 < bds[0][1] and \
            bds[1][0] < c1 < bds[1][1] and \
            bds[2][0] < c2 < bds[2][1] and \
            bds[3][0] < amp0 < bds[3][1] and \
            bds[4][0] < T0 < bds[4][1] and \
            bds[5][0] < phi0 < bds[5][1] and \
            bds[6][0] < amp1 < bds[6][1] and \
            bds[7][0] < T1 < bds[7][1] and \
            bds[8][0] < phi1 < bds[8][1]:
        return 0.0
    return -np.inf

# Posterior probabilities

def lnprob_d2nu_verma(params,model,data,bds):
    lp = lnprior_d2nu_verma(params,bds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikelihood_d2nu(params,model,data)

def lnprob_d2nu_mazumdar(params,model,data,bds):
    lp = lnprior_d2nu_mazumdar(params,bds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikelihood_d2nu(params,model,data)

def lnprob_d2nu_basu(params,model,data,bds):
    lp = lnprior_d2nu_basu(params,bds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikelihood_d2nu(params,model,data)

# def lnprob_d2nu_houdek(model,data,params,bds):
#     lp = lnprior_d2nu_houdek(params,bds)
#     if not np.isfinite(lp):
#         return -np.inf
#     return lp + lnlikelihood_d2nu(params,model,data)

def lnprob_rr010_const_amp(params,model,data,bds):
    lp = lnprior_rr010_const_amp(params,bds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikelihood_rr010(params,model,data)

def lnprob_rr010_freqinv_amp(params,model,data,bds):
    lp = lnprior_rr010_freqinv_amp(params,bds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikelihood_rr010(params,model,data)

def lnprob_rr010_freqinvsq_amp(params,model,data,bds):
    lp = lnprior_rr010_freqinvsq_amp(params,bds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikelihood_rr010(params,model,data)

def lnprob_rr010_freqinvpoly_amp(params,model,data,bds):
    lp = lnprior_rr010_freqinvpoly_amp(params,bds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikelihood_rr010(params,model,data)

def lnprob_rr010_freqinvsq_amp_polyper(params,model,data,bds):
    lp = lnprior_rr010_freqinvsq_amp_polyper(params,bds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikelihood_rr010(params,model,data)


# Select the function that will be used to estimate the posterior probability
# function based on the model selected

def select_prob(model):
    """ Selects the function that will be used to estimate the posterior
        probability function based on the model selected. """
    if model == d2nu_verma:
        return lnprob_d2nu_verma
    elif model == d2nu_mazumdar:
        return lnprob_d2nu_mazumdar
    elif model == d2nu_basu:
        return lnprob_d2nu_basu
    # elif model == d2nu_houdek:
    #     return lnprob_d2nu_houdek
    elif model == rr010_const_amp:
        return lnprob_rr010_const_amp
    elif model == rr010_freqinv_amp:
        return lnprob_rr010_freqinv_amp
    elif model == rr010_freqinvsq_amp:
        return lnprob_rr010_freqinvsq_amp
    elif model == rr010_freqinvpoly_amp:
        return lnprob_rr010_freqinvpoly_amp
    elif model == rr010_freqinvsq_amp_polyper:
        return lnprob_rr010_freqinvsq_amp_polyper


# Miscellaneous functions

def f2solve_smooth_component(params,freq,inv_matcov):
    """ Represents the equations to solve to determine beta, gamma1 and gamma2.
        See Deheuvels et al. 2016, Eqs. B9-11. """
    beta, gamma1, gamma2 = params
    Jt = np.vstack((np.ones(len(freq)),freq-beta,(freq - gamma1)*(freq - gamma2)))
    JtWJ = np.dot(np.dot(Jt,inv_matcov),np.transpose(Jt))   # J_transposed * inv_matcov * J
    # Non-diagonal coefficients of J_tranposed * inv_matcov * J
    b9 = JtWJ[1,0]
    b10 = JtWJ[2,0]
    b11 = JtWJ[2,1]
    return (b9,b10,b11)
