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
"nsvd" : 19,
"beta" : 0,
"gamma1" : 0,
"gamma2" : 0
}

########  Saving options #############
save_params = {
"directory": '8938364/',  # files are saved in this directory (created if does not exist)
"nameplate": '8938364'   # all saved files will begin by the value of nameplate
}
