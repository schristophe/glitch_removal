# glitch_removal
**Fitting acoustic glitches in solar-like stars with `emcee`**

Rapid variations of the stellar structure introduce an oscillatory component in the eigenmode frequencies of pulsating stars. Such features are called *glitches* in the asteroseismology community. The period and amplitude of the oscillatory component are related to the location and the shape of the glitch, respectively. `glitch_removal` fits the seismic signature of glitches in the pressure mode frequencies of solar-like stars (or combinations thereof).

## Installation

The current version of `glitch_removal` does not require installation. To start using it, download or clone the code and move to the directory:

```
git clone https://github.com/schristophe/glitch_removal/
cd glitch_removal
```

#### Dependencies

`glitch_removal` is developed in Python 3.6+. The following packages are required:
* `numpy`
* `scipy`
* `matplotlib`
* [`emcee`](https://emcee.readthedocs.io/)
* [`corner.py`](https://corner.readthedocs.io/)


## Quickstart

```python
import glitch_removal as gr
```

**1. Load a frequency set from a file.**

For instance, let's load the frequencies of KIC8938364:
 ```python
 star = gr.FreqTable()
 star.load('8938364.fre', coln=0, coll=1)
 ```
By default, the program assumes that the file has the following format: (angular degree l, radial order n, frequencies, uncertainties). In '8938364.fre', angular degrees and radial orders are swapped so we have to give their column numbers to `load()`.

**2. Compute seismic indicators**
```python
star.calc_d2nu()    # second differences
star.calc_ratios()  # frequency ratios
```
Results are stored in the attributes `d2nutable`, `rr010table` and `r02table`. You may want to use `plot()` to have a look at the glitch signatures:
```python
star.d2nutable.plot()
star.rr010table.plot()
star.r02table.plot()
```

**3. Set parameters used by the fitting routine**

The global seismic parameters are used to set priors and find a good initial guess.
```python
gr.star_params['delta_nu'] =  85.7  # large separation
gr.star_params['numax'] = 1675      # frequency of maximum oscillating power
```

Seismic indicators are correlated, which is taken into account in the fitting through the covariance matrix. Its inversion usually requires a form of regularisation. To do that, we use the truncated singular value decomposition method. The number of singular values to keep is set by the following parameter:
```python
gr.fit_params['nsvd'] = 19
```


**4. Initialise a glitch model.**

Here, for example, we will fit the model `rr010_freqinv_amp` (see documentation) to the ratios rr010:
```python
model = gr.GlitchModel(gr.rr010_freqinv_amp,star.rr010table)
```


**5. Determine plausible initial values for model parameters**
```python
model.initial_guess()
```
Initial guess is found by performing a least squares fit. Boundaries for the model parameters (priors) are also set by `initial_guess()`

**6. Run the MCMC sampling with `emcee`**
```python
model.run_mcmc()
```

**7. Save some figures and a log file with the results.**
```python
model.log_and_plot()
```
Figures include corner plot, evolution of the walkers, fitted model overlaid on observed data points.

## Citing

## License

`glitch_removal` is made available under the GNU GPLv3 license. Please see the [LICENSE](https://github.com/schristophe/glitch_removal/blob/master/LICENSE) file for details.
