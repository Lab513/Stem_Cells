# stem_cells

## Count stem cells and detect gaps

Neural network code for segmenting stem cells to ease cells analysis

## Installation

stem_cells requires Python 3 and [TensorFlow](https://www.tensorflow.org).

run:

```bash
> pip install tensorflow
```

**NB:** To make use of a GPU you should also follow the [set up
instructions](https://www.tensorflow.org/install/gpu#windows_setup) for
installing `tensorflow-gpu`.

### other dependencies

For reading excel files
```bash
> pip install openpyxl
```


NB: If you are upgrading, then you may need to run: `git pull`


## Run using Jupyter Notebook for analysis

```bash
> jupyter-notebook
```

open `count stem.ipynb` and make a copy with a new name in which you will work

Enter the name of the folder containing the folders with the images.     
For example:

```python
addr_imgs = 'E:/Incucyte data/AD-Exp0071/HSC'
```

Set the wells to be analysed and execute the Jupyter cell  
For example:

```python
lspan = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
nspan = range(2,10)

for l in lspan:
    for i in nspan:
        sc.analyse_one_well(curr_well)
sc.plot_analysis_all_wells()
```

## Output

The results are produced in the folder `results`.

### Visualisation

The analysis can be made quickly with the web application *stem_visu*.

Copy the *result* folder in stem_visu/static

Then in a terminal run

```python
python -m stem_visu.run
```
#### Using the interface

<details>
<summary> view on current well, graph and plate </summary>

The interface is composed of three parts.
A window with a view of the current observed well a window with the graph
of the evolution of the number of cell detected in the current
 well during the experiment and a representation of the experimental
  plate with 96 wells


In the current well view there is a slider for moving in the time axis.
To change the position in time axis the two arrows can be used also.
If using the arrows there are two options : hover or click which can be
chosen with the selectors under the window.

The Selector at the bottom is used for choosing between the *brightfield view* or
the *prediction overlaid view*.


For choosing a new well the user can click directly on a well on the
representation of the 96 wells plate or can navigate using the
cursors up, down, right, left.

![interface overview](stem_visu/static/imgs/interf_stem_medium.png)

</details>
