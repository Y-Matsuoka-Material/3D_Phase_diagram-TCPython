# 3D Phase diagram-TCPython

3D Phase diagram-TCPython offers a set of tools for calculating three-dimensional phase diagrams. This suite includes functionalities for computing ternary phase diagrams (`ternary_phase_diagram.py`) and isothermal sections of quaternary phase diagrams (`quaternary_phase_diagram.py`), alongside Gibbs energy surface calculations (`energy_surface.py`). 

## Installation

### Quick Start
No specific installation process is needed for the code itself. Simply download the scripts and execute them.

### Dependencies
The following libraries, including [TC-Python](https://thermocalc.com/products/software-development-kits/tc-python/), are required:

- numpy
- matplotlib
- scipy
- Pillow
- psutil
- opencv-python
- PyOpenGL
- glfw
- pygltflib
- TC-Python

All libraries, except TC-Python, can be installed using pip:

```shell
pip install numpy matplotlib scipy Pillow psutil opencv-python PyOpenGL glfw pygltflib
```

For TC-Python installation, please refer to [the official documentation](https://www2.thermocalc.com/docs/tc-python/2023b/html/). If using [Anaconda](https://www.anaconda.com/), the first five libraries may already be installed, so ensure to install the remaining ones as needed.

## Usage

### Configuring Calculation Conditions

Modify the configuration section, located beneath the import statements at the top of each script, to set the calculation conditions.

#### Example (`quaternary_phase_diagram.py`):

```python
# Elements used in the calculation
elements = ["Ni", "Co", "Cr"]

# List of phases considered in the equilibrium calculation
# If list is empty, the default phases selected by ThermoCalc is used.
considered_phases = []

# Temperature range in Kelvin
T_range = [600, 2400]

# The name of database used in the calculation
# TDB file can be also used by specifying the path to the TDB file.
database = "TCNI11"
```

Additional settings like grid divisions and parallelization numbers are also configurable in the same section. Refer to in-script comments for detailed descriptions of each option.

### Program Execution

Executing `ternary_phase_diagram.py`, `quaternary_phase_diagram.py`, or `energy_surface.py` will generate corresponding figures. By default, numerical data from the calculated phase diagram is saved in `./data`, while the 3D model, animation, and figure legends are stored in `./img`. Upon completion of data saving, a new window will appear, allowing interactive exploration of the calculation results.

### Interactive Visualization Window Controls

Use keyboard inputs to navigate the interactive window as follows:

| Key                 | Operation            |
| ------------------- | -------------------- |
| Space               | Toggle auto rotation |
| Arrow (↑↓←→)        | Rotate the figure    |
| Shift + Arrow(↑↓←→) | Move the figure      |
| PgUp/PgDn           | Zoom in/out          |
| F2                  | Take screenshot      |
| Esc                 | Close the window     |

### Notes

- As the 3D state diagram performs numerous parallel calculations, please adjust the number of parallel computations (`num_core`) according to your PC's capabilities.
- While commercial databases can be used, a path to the TDB file may be specified for the database. Due to a bug with Thermo-Calc, paths containing double-byte characters (e.g., Japanese) are not readable. Ensure to use only alphanumeric characters.
- Although electron gas and vacancies are included in calculations by default, this may prevent the program from reading databases where they are not defined. In such instances, set `add_special_component` to False.
- For phase diagrams involving phase separation, where phase names are distinguished (#1, #2, etc.), coloring may not function optimally. If so, set `distinguish_phase_separation` to False.
