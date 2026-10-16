# Machine Learning Starter Project

A place to learn the basics of machine learning with Python. The first goal is to build a small model that classifies penguin species from measurements.

## Getting started

This project uses Python, JupyterLab, pandas, scikit-learn, and matplotlib.

From PowerShell in the project folder, create and set up a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install jupyterlab pandas scikit-learn matplotlib
```

Start JupyterLab with:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

In VS Code, select the Python interpreter inside `.venv` so notebooks and scripts use the installed packages.

## First project

The first project will use penguin measurements to predict a penguin's species. The steps are:

1. Load and explore a labeled dataset.
2. Prepare the measurements for a model.
3. Split the data into training and test sets.
4. Train a simple classification model.
5. Evaluate its predictions on the test set.

The model and dataset have not been added yet; this README describes the starting plan.
