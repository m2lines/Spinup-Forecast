# Spin‑Up NEMO Forecasting Framework

**Overview**

This project provides a flexible framework for oceanographic time‑series forecasting. It separates dimensionality reduction (DR) and forecasting into interchangeable components, enabling you to swap in your own algorithms with minimal changes.

---

## 1. Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/m2lines/nemo-spinup-forecast
   cd <repo_dir>
   ```
2. **Set up a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**

   ```bash
   pip install .
   # or for developer dependencies
   pip install .[dev]
   ```

   If you plan to edit the code, use an editable install so your changes
   take effect without reinstalling:

   ```bash
   pip install -e .
   ```

   Without `-e`, any edit to the source requires rerunning `pip install .`
   before the CLI picks it up.

---

## 2. Related projects

This repository is one piece of a toolkit for NEMO spin-up work.

- **[nemo-spinup-restart](https://github.com/m2lines/nemo-spinup-restart)** —
  builds NEMO restart files from forecast output, if you want to
  continue a simulation from a forecast produced here, use that repo.
- **[nemo-spinup-evaluation](https://github.com/m2lines/nemo-spinup-evaluation)** —
  tools for evaluating forecast quality.

---

## 3. Quick Start

1. Once you have cloned the repository and built the environment, there is test data available for quick experimentation. Download it using the script in the `tests` directory:

   ```bash
   ./tests/download_test_data.sh
   ```

2. Run the forecasting script on the test data:

    Set `--data-path` to the directory where you downloaded the test data and `--output-path` to where results should be written:

    ```bash
    python -m nemo_spinup_forecast \
      --ye True \
      --start 20 \
      --end 50 \
      --comp 1 \
      --steps 30 \
      --data-path /path/to/simulation/files \
      --output-path /path/to/output \
      --ocean-terms /path/to/ocean_terms.yaml \
      --techniques-config /path/to/techniques_config.yaml
    ```
    This will fit the model on 30 years of data and forecast a jump of 20 years using PCA and a Gaussian process.

   ### Arguments

   - **`data-path`** — Directory containing the simulation files
   - **`output-path`** — Directory to write forecast results to
   - **`ye`** — The simulation is expressed in years (`True`) or months (`False`)
   - **`start`** — Starting year (training data)
   - **`end`** — Ending year (usually the last simulated year)
   - **`comp`** — Number or ratio of components to accelerate
   - **`steps`** — Jump size (years if `ye=True`, months otherwise)
   - **`ocean-terms`** — Path to a custom `ocean_terms.yaml` mapping logical terms (e.g., SSH, Salinity, Temperature) to dataset variable names. If omitted, a packaged default is used.
   - **`techniques-config`** — Path to a custom `techniques_config.yaml` selecting DR and forecast techniques. If omitted, the default packaged config directory is used.

   ### Outputs

   - Prepared data in `<output-path>/latest/forecast/simu_prepared/{term}/`
   - Forecasted components in `<output-path>/latest/forecast/simu_predicted/{term}.npy`

---

## 4. Configuration

All user‑selectable techniques live in `techniques_config.yaml`:

```yaml
DR_technique:
  name: PCA                # Options: PCA, KernelPCA, or your custom class
Forecast_technique:
  name: GaussianProcessRecursiveForecaster  # Options: GaussianProcessForecaster, GaussianProcessRecursiveForecaster, or your custom class
```

Ocean term definitions live in `src/nemo_spinup_forecast/configs/ocean_terms.DINO.yaml`.
This file is the single source of truth for which variables to forecast
and which NetCDF files to read them from:

```yaml
terms:
  salinity:
    filename: DINO_1y_grid_T.nc
    term: soce
  temperature:
    filename: DINO_1y_grid_T.nc
    term: toce
  ssh:
    filename: DINO_1m_To_1y_grid_T.nc
    term: ssh
```

Each entry has three parts:

- **top-level key** (`ssh`, `salinity`, `temperature`) — identifier used
  throughout the pipeline to reference this term
- **`filename`** — exact NetCDF filename inside `--data-path`. Not a
  glob or regex; the literal filename is expected
- **`term`** — name of the variable inside that NetCDF file

### Using your own simulation data

The packaged config targets DINO output. To forecast different data
(e.g. ORCA1, or DINO files you renamed), copy the packaged YAML, edit
the filenames and variable names to match your files, and pass the
custom path to the CLI:

```bash
python -m nemo_spinup_forecast \
  --ye True \
  --start 20 \
  --end 50 \
  --comp 1 \
  --steps 30 \
  --data-path /path/to/simulation/files \
  --output-path /path/to/output \
  --ocean-terms /path/to/my_ocean_terms.yaml \
  --techniques-config /path/to/techniques_config.yaml
```

Because filenames live in the YAML, changing them does **not** require
reinstalling the package — just edit the YAML and rerun the CLI.

## 5. Project Structure

```text
├── Notebooks
│   ├── Jumper.ipynb
│   └── Resample_ssh.ipynb
├── pyproject.toml
├── README.md
├── ruff.toml
├── src
│   └── nemo_spinup_forecast
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── configs
│       │   ├── ocean_terms.DINO.yaml
│       │   └── techniques_config.yaml
│       ├── density.py
│       ├── dimensionality_reduction.py
│       ├── forecast_method.py
│       ├── forecast.py
│       ├── pipeline.py
│       ├── pipeline_utils.py
│       ├── plotting_utils.py
│       └── utils.py
└── tests
```

---

## 6. Extending the Framework

### 6.1 Adding a Custom Dimensionality Reduction

```python
from nemo_spinup_forecast.dimensionality_reduction import DimensionalityReduction

class MyDR(DimensionalityReduction):
    def __init__(self, comp, **kwargs):
        self.comp = comp
        # initialise other parameters

    def set_from_simulation(self, sim):
        # copy metadata from Simulation
        ...

    def decompose(self, simulation, length):
        # return components, model‑instance, mask
        ...

    @staticmethod
    def reconstruct_predictions(predictions, n, info, begin=0):
        # return mask, reconstructed‑array
        ...
```

Register the class in the `dimensionality_reduction_techniques` dict at the bottom of `src/nemo_spinup_forecast/dimensionality_reduction.py`, and select it in `techniques_config.yaml` just as for the built‑in techniques.

### 6.2 Adding a Custom Forecasting Technique

```python
from nemo_spinup_forecast.forecast_method import BaseForecaster

class MyForecaster(BaseForecaster):
    def __init__(self, **params):
        # initialise your model
        ...

    def apply_forecast(self, y_train, x_train, x_pred):
        # fit your model on y_train (and x_train), predict x_pred
        # return (y_hat, y_hat_std)
        ...
```

Register the class in the `forecast_techniques` dict at the bottom of `src/nemo_spinup_forecast/forecast_method.py` and reference it in `techniques_config.yaml`.

---

## 7. Example Notebook

`Jumper.ipynb` demonstrates forecasting with `PCA`. Copy, modify, or extend it to test your own techniques.

### 7.1 Jumper.ipynb — Prepare and Forecast Simulations

The objective is to implement a Gaussian process forecast to forecast yearly simulations of the NEMO coupled climate model. For this we need simulation files of the sea surface height (zos or ssh), the salinity (so) and temperature (thetao).

We apply PCA on each simulation to transform those features to time series and observe the trend in the first component.

![img1](img/jumper1.png)

We forecast each component with a Gaussian process using the following kernel:

* **Long‑term trend**   : `0.1*DotProduct(sigma_0=0.0)`
* **Periodic patterns** : `10*ExpSineSquared(length_scale=5/45, periodicity=5/45)`
* **White noise**       : `2*WhiteKernel(noise_level=1)`

![img2](img/jumper3.png)

We then evaluate the RMSE:

![img2](img/jumper2.png)

---

## 8. Testing
### *Unit and integration tests for the Spin-Up NEMO project*
The tests are designed to ensure the functionality of the Spin-Up NEMO project, which involves preparing and forecasting simulations.

To run the tests, you first need to download the necessary data files.
You can do this by running the download script within the `tests` directory from the root of the project:

```bash
./tests/download_test_data.sh
```

Then execute the tests using pytest. The tests are located in the `tests` directory, and you can run them with the following command:
```bash
pytest tests/
```

---

## Acknowledgements

This project includes code adapted from the [Spinup-NEMO repository](https://github.com/maudtst/Spinup-NEMO) by Maud Tissot. The original work is licensed under the MIT License.
