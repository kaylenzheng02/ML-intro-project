# Penguin Species Classifier

A beginner machine-learning project that predicts a penguin's species from its bill measurements, flipper length, and body mass.

The project uses the Palmer Penguins dataset, included in `data/penguins.csv`. The data comes from the [Palmer Penguins project](https://github.com/allisonhorst/palmerpenguins).

## Set up

From PowerShell in this project folder, create a virtual environment and install the project dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you already created `.venv`, skip the first command. In VS Code, select the Python interpreter inside `.venv`.

## Explore the data in a website

Start the interactive graph app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

It opens in your browser on your computer. Use the sidebar dropdowns to choose the x-axis measurement, y-axis measurement, and species to show (or all species). Hover over a point to see that penguin's island and sex. In **Try a prediction**, enter four measurements to see the model's predicted species and estimated probability for each species. The example values start with measurements from a Gentoo penguin. Close the terminal or press Ctrl+C to stop the app. This is a local website; it is not published to the internet.

## Train the classifier

```powershell
.\.venv\Scripts\python.exe train.py
```

The script splits the data into training and test sets, fills in missing measurements, trains a logistic-regression classifier, and prints its accuracy and a per-species classification report. The test set is held out during training so it can give an estimate of how the model handles new examples. The website separately fits the same kind of model on the labeled dataset to make interactive predictions.

## Project files

- `data/penguins.csv` — labeled penguin measurements and species.
- `app.py` — interactive graph website for exploring measurements.
- `.streamlit/config.toml` — website colors, theme, and font.
- `train.py` — data loading, model training, and evaluation.
- `requirements.txt` — Python packages needed to run the project.
