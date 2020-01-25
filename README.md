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

`glitch_removal` is developed in Python 3.6. The following packages are required:
* `numpy`
* `matplotlib`
* [`emcee`](https://emcee.readthedocs.io/)


## Quickstart

```python
import glitch_removal as gr
```

Load a frequency set from a file located at path:
 ```python
 star = gr.FreqTable(path)
 star.load()
 ```
The program assumes the file has the following format by default: (angular degree l, radial order n, frequencies, uncertainties).  
 
Compute seismic indicators (second differences or frequency ratios):
```python
star.calc_d2nu()
star.calc_ratios()
```

Plot the results:
```python
star.d2nutable.plot()
star.rr010table.plot()
star.r02table.plot()
```

Initiate a glitch model to represent a given seismic indicator, e.g.:
```python
model = gr.GlitchModel(model_name,star.rr010table)

```

## Citing

## License

`glitch_removal` is made available under the GNU GPLv3 license. Please see the [LICENSE](https://github.com/schristophe/glitch_removal/blob/master/LICENSE) file for details.
