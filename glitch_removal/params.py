#
#

######### emcee parameters #############
emcee_params = {
"ndim" : 6,
"nwalkers" : 250,   #  number of walkers
"nburns" : 4000,    # number of burn-in steps
"nsteps" : 5000     # number of steps
}

######### Stellar parameters ##########
star_params = {
"delta_nu" : 120.,  # large separation
"numax" : 1800.     # frequency of maximum oscillating power
}

######### Parameters for the fit ########
fit_params = {
"freqref": star_params["numax"], # reference frequency for the glitch(es) amplitude(s)
"nsvd" : 0,    #  number of singular values to keep when inverting the cov. mat
"beta" : 0,
"gamma1" : 0,
"gamma2" : 0
}

########  Saving options #############
save_params = {
"directory": 'results/',  # files are saved in this directory (created if does not exist)
"nameplate": ''   # all saved files will begin by the value of nameplate
}
