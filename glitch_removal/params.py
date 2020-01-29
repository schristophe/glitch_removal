#
#

######### emcee parameters #############
emcee_params = {
"ndim" : 6,
"nwalkers" : 250,
"nburns" : 4000,
"nsteps" : 5000
}

######### Stellar parameters ##########
star_params = {
"delta_nu" : 120.,
"numax" : 1800.
}

######### Parameters for the fit ########
fit_params = {
"freqref": 0.8*star_params["numax"],
"nsvd" : 19
}
