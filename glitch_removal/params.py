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
"freqref": 0.8*star_params["numax"], # reference frequency for the glitch(es) amplitude(s)
"nsvd" : 19,    #  number of singular values to keep when inverting the cov. mat
"beta" : 0,
"gamma1" : 0,
"gamma2" : 0
}

########  Saving options #############
save_params = {
"directory": '8938364/',  # files are saved in this directory (created if does not exist)
"nameplate": '8938364'   # all saved files will begin by the value of nameplate
}
