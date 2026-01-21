# UW–Madison Math/CS 513: Code examples
Math/CS 513 is an undergraduate course taught at the University of Wisconsin, Madison on numerical linear algebra. [Visit the course website.](https://web.math.wisc.edu/math513/)

These code examples accompany the lecture materials. They are primarily written in Python and make use of several well-established libraries:

- [NumPy](https://numpy.org) for numerical linear algebra
- [SciPy](https://scipy.org) for scientific routines and algorithms
- [Matplotlib](https://matplotlib.org) for graphing and visualization

## Units
- [Unit 0: Fundamentals](https://github.com/rycroft-group/math513/tree/main/0_fundamentals)
- Unit 1: Singular value decomposition
- Unit 2: QR decomposition and least squares
- Unit 3: Conditioning and stability
- Unit 4: Linear systems
- Unit 5: Eigenvalue problems
- Unit 6: Iterative methods

## Run code

- MATLAB (local machine): navigate to the target folder and run MATLAB scripts
    
- [MATLAB Online](https://matlab.mathworks.com/): first need to create a new project by cloning the repo, then navigate to the target folder and run MATLAB scripts

> Caveat: MATLAB scripts can run interactively online, but have not found a way to update the cloned repo in if new files are added

- Python (local machine): need local Python and Jupyter installation to run notebooks

- Python ([Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb)): click "Open in Colab" badge on the top of each `.ipynb` notebook to run (no need to install anything for Python to run)

> If you are running the notebook on Google Colab, please make a copy of the notebook to your drive:
>
> - click "Copy to Drive"
> - or navigate to "File -> Save a copy in Drive"
> - or navigate to "File -> Download" and save a local copy
>
> Otherwise all your changes will not be saved

## Resources

- [Install Git](https://www.atlassian.com/git/tutorials/install-git)
- [Git cheat sheet](https://education.github.com/git-cheat-sheet-education.pdf) (You will mostly only be using `git pull` to retrieve updated files)
- [Install Jupyter using Anaconda and conda](https://docs.jupyter.org/en/latest/install/notebook-classic.html#installing-jupyter-using-anaconda-and-conda) (The straightforward way, but your libraries are installed in `conda` environment; may encounter path/dependency issues in future)
- [Install Jupyter with pip](https://docs.jupyter.org/en/latest/install/notebook-classic.html#alternative-for-experienced-python-users-installing-jupyter-with-pip) (If you already have Python3 installed and do not want to deal with additional `conda` enviroment, which gives a cleaner setup for future Python3 usage)
- [Python tutorial](https://github.com/rycroft-group/math513/tree/main/python_tutorial)