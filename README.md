# Spin‑Up NEMO Forecasting Framework

**Overview**

This project provides a flexible framework for oceanographic time‑series forecasting. It separates dimensionality reduction (DR) and forecasting into interchangeable components, enabling you to swap in your own algorithms with minimal changes.

---

## 1. Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/m2lines/Spinup-Forecast
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

---

## 2. Quick Start

1. Once you have cloned the repository and built the environment, there is test data available for quick experimentation. Download it using the script in the `tools` directory:

   ```bash
   ./tools/download_test_data.sh
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

## 3. Configuration

All user‑selectable techniques live in `techniques_config.yaml`:

```yaml
DR_technique:
  name: PCA                # Options: PCA, KernelPCA, or your custom class
Forecast_technique:
  name: GaussianProcessRecursiveForecaster  # Options: GaussianProcessForecaster, GaussianProcessRecursiveForecaster, or your custom class
```

Maps for NEMO grids live in `ocean_terms.yaml`:

```yaml
Terms:
  Salinity: soce
  Temperature: toce
  SSH: ssh
```
## 4. Project Structure

```text
│── Notebooks
│   ├── Jumper.ipynb
│   ├── Resample_ssh.ipynb
│   └── Restart.ipynb
├── ocean_terms.yaml
├── pyproject.toml
├── README.md
├── ruff.toml
├── src
│   └── nemo_spinup_forecast
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── density.py
│       ├── dimensionality_reduction.py
│       ├── forecast_method.py
│       ├── forecast.py
│       ├── optimization.py
│       ├── restart.py
│       └── utils.py
```

---

## 5. Extending the Framework

### 5.1 Adding a Custom Dimensionality Reduction

```python
from lib.dimensionality_reduction import DimensionalityReduction

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

Register the class in `forecast.py` and select it in `techniques_config.yaml` just as for the built‑in techniques.

### 5.2 Adding a Custom Forecasting Technique

```python
from lib.forecast_technique import BaseForecaster

class MyForecaster(BaseForecaster):
    def __init__(self, **params):
        # initialise your model
        ...

    def apply_forecast(self, y_train, x_train, x_pred):
        # fit your model on y_train (and x_train), predict x_pred
        # return (y_hat, y_hat_std)
        ...
```

Again, register your class and reference it in `techniques_config.yaml`.

---

## 6. Example Notebook

### **NOTE: This notebook is expected to fail currently and will be replaced in future updates.**

`Jumper.ipynb` demonstrates forecasting with `PCA`. Copy, modify, or extend it to test your own techniques.

### 6.1 Jumper.ipynb — Prepare and Forecast Simulations

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

## 7. Restart Workflow

### 7.1 Restart.ipynb — Update of Restart Files for NEMO

The objective is to update the last restart file to initialise the jump. For this we need the 340 restart files of the last simulated year, the predictions of sea surface height (zos or ssh), salinity (so) and temperature (thetao), and the Mask dataset of the corresponding simulation.

![img5](img/img3.png)

#### 1 – Predicted features

* **zos**       : Predicted sea surface height (ssh) – grid T – t,y,x
* **so**        : Predicted salinity – grid T – t,z,y,x
* **thetao**    : Predicted temperature – grid T – t,z,y,x

#### 2 – Maskdataset

*dimensions: t:1  y:331  x:360  z:75*

* `umask`   : continent mask for u grid (continent 0, sea 1)
* `vmask`   : continent mask for v grid (continent 0, sea 1)
* `e3t_0`   : initial cell thickness on z‑axis (grid T)
* `e2t`, `e1t`, `ff_f`, … 

#### 3 – Necessary features to update restart

* `e3t`    : `e3t_ini*(1+tmask4D*np.expand_dims(np.tile(ssh*ssmask/(bathy+(1-ssmask)),(75,1,1)),axis=0))`
* `deptht` : depth of the z‑axis – grid T
* `bathy`  : `np.ma.sum(e3t,axis=1)`

#### 4 – Restart file update

There are 340 restart files per year. Each file contains a slice of the x and y dimensions and 58 data variables, 15 of which are updated using the predictions.

> **NB**   `n` (now) and `b` (before) arrays represent the same state; in practice, NEMO saves two successive states in each file. In our code we set the two to the same state to use Euler forward for the restart.

* **ssh(n/b)** : sea‑surface height           → last prediction of zos

* **s(n/b)**   : sea salinity                 → last prediction of so

* **t(n/b)**   : sea temperature              → last prediction of thetao

* **v(n/b)**   : zonal velocity               → *The Planetary Ocean* (M. Fieux) p70

  $V(z)=\frac{g}{p\,f}\int_{z_0}^{Z}\frac{\partial\rho}{\partial x}\,dz + V_0$

* **u(n/b)**   : meridional velocity          → *The Planetary Ocean* p70

  $U(z)=\frac{g}{p\,f}\int_{z_0}^{Z}\frac{\partial\rho}{\partial y}\,dz + U_0$

* **sss\_m**    : sea‑surface salinity        → last prediction of so

* **sst\_m**    : sea‑surface temperature     → last prediction of thetao

* **ssu\_m**    : sea‑surface *u* velocity    → from new u

* **ssv\_m**    : sea‑surface *v* velocity    → from new v

* **rhop**     : potential density (kg m⁻³)  — see Equation of State of Sea‑Water (J. Le Sommer)

*Grid visuals*

![img6](img/grid0.png)
![img7](img/grid1.png)


---

## 8. End‑to‑End Steps for Running Spin‑Up NEMO

1. **Run DINO for 50‑100 years.** A Slurm script is provided in the NEMO [notes](https://github.com/m2lines/Spinup-NEMO-notes/blob/main/nemo/buildandrun_NEMODINO.md). If you need more training data, concatenate monthly outputs `*grid_T.nc` with `ncrcat`.

2. **Create a virtual environment** (e.g. with conda, venv) and install `requirements.txt`.

3. **Resample SSH**. Use the [Resample\_ssh.ipynb](https://github.com/m2lines/Spinup-NEMO/blob/resample_dino_data/Notebooks/Resample_ssh.ipynb) notebook to convert monthly SSH (`DINO_1m_grid_T.nc`) to annual (`DINO_1m_To_1y_grid_T.nc`). Temperature and salinity (3‑D) are already annual (`DINO_1y_grid_T.nc`).

4. **Create the projected state** with the updated `main_forecast.py`. The DINO grid_T files are needed to run the forecast.

    Set `data-path` to the NEMO/DINO data directory:

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

   - **`data-path`** — Directory containing the simulation files
   - **`output-path`** — Directory to write forecast results to
   - **`ye`** — The simulation is expressed in years (`True`) or months (`False`)
   - **`start`** — Starting year (training data)
   - **`end`** — Ending year (usually the last simulated year)
   - **`comp`** — Number or ratio of components to accelerate
   - **`steps`** — Jump size (years if `ye=True`, months otherwise)
   - **`ocean-terms`** — Path to a custom `ocean_terms.yaml` mapping logical terms (e.g., SSH, Salinity, Temperature) to dataset variable names. If omitted, a packaged default is used.
   - **`techniques-config`** — Path to a custom `techniques_config.yaml` selecting DR and forecast techniques. If omitted, the default packaged config directory is used.


5. **Prepare restart files**. Combine `mesh_mask_[0000].nc` and `DINO_[<time>]_restart_[<process>].nc` with **[REBUILD\_NEMO](https://forge.nemo-ocean.eu/nemo/nemo/-/tree/4.2.0/tools/REBUILD_NEMO)**:

    You can use the same module environment which you used to run NEMO/DINO to compile rebuild_nemo.

   ```bash
   ./rebuild_nemo -n ./nam_rebuild /path/to/DINO/restart/file/DINO_00576000_restart 36
   ./rebuild_nemo -n ./nam_rebuild /path/to/DINO/mesh_mask 36
   ```

6. **Create an updated restart file** with `main_restart.py`:

   ```bash
   python main_restart.py \
     --restart_path /path/to/EXP00/ \
     --radical DINO_[<time>]_restart \
     --mask_file /full/path/to/EXP00/mesh_mask.nc \
     --prediction_path /full/path/to/directory/simus_predicted/
   ```

   * `--radical` is the prefix of the restart file;
   * `<time>` is the most recent restart time.

   The modified script (see the `run_with_DINO_data` branch) creates files named as the originals but with `NEW` prepended.

7. **Copy the experiment directory** inside the NEMO repository. Keep it as backup; the original will be overwritten.

8. **Copy the mesh mask and restart files** (`mesh_mask_<proc_id>.nc` and `DINO_[<time>]_restart_<proc_id>.nc`) back to the original experiment directory.

9. **Update `namelist_cfg`**

   Under `namrun` adjust:

   * `nn_it000` : first timestep (last timestep + 1)
   * `nn_itend` : final timestep
   * `cn_ocerst_in` : restart filename syntax (matches latest restart file)
   * `ln_rstart` : `.true.` to start from a restart file

   ![image](https://hackmd.io/_uploads/HJtsvsCPJe.png)

10. **Restart DINO** using the updated restart file.

---

## 9. Testing
### *Unit and integration tests for the Spin-Up NEMO project*
The tests are designed to ensure the functionality of the Spin-Up NEMO project, which involves preparing and forecasting simulations.

To run the tests, you first need to download the necessary data files.
You can do this by running the download script within the tools directory from the root of the project:

```bash
./tools/download_test_data.sh
```

Then execute the tests using pytest. The tests are located in the `tests` directory, and you can run them with the following command:
```bash
pytest tests/
```

---

## Acknowledgements

This project includes code adapted from the [Spinup-NEMO repository](https://github.com/maudtst/Spinup-NEMO) by Maud Tissot. The original work is licensed under the MIT License.
