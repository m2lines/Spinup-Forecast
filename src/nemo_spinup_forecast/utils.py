import os
import uuid
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

import yaml

from nemo_spinup_forecast.forecast import Simulation


def create_run_dir(output_base: str) -> Path:
    """
    Create a new timestamped run directory and update the `latest` symlink.

    A directory is created under ``<output_base>/runs`` with a unique
    timestamp-based name. After creation, the ``latest`` symlink in
    ``<output_base>`` is atomically updated to point to the new directory.

    Parameters
    ----------
    output_base : str
        Base output path under which the run directories are stored.

    Returns
    -------
    Path
        Path to the newly created run directory.
    """
    base = Path(output_base).expanduser().resolve()
    runs_root = base / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S.%fZ")
    random_id = uuid.uuid4().hex[:8]  # 8-char unique ID
    run_id = f"{ts}_{random_id}"

    run_dir = runs_root / run_id
    run_dir.mkdir(parents=False, exist_ok=False)

    # Update 'latest' symlink atomically
    _update_symlink_atomic(base, "latest", run_dir)
    return run_dir


def _update_symlink_atomic(base: Path, name: str, target: Path):
    base.mkdir(parents=True, exist_ok=True)
    tmp = base / f"{name}.tmp"
    final = base / name
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    os.symlink(os.path.relpath(target, base), tmp)
    os.replace(tmp, final)


def prepare(term, filename, data_path, out_path, start, end, ye, comp, dr_technique):
    """
    Prepare the simulation for the forecast.

    Args:
        term (str): term to forecast
        filename (str): name of the NetCDF file containing the term
        data_path (str): path to the directory containing input .nc files
        out_path (str): path to the run directory for writing results
        start (int): start of the simulation
        end (int): end of the simulation
        ye (bool): transform monthly simulation to yearly simulation
        comp (int or float): explained variance ratio for the PCA

    Returns
    -------
        simu (Simulation): simulation object

    """
    # Load yearly or monthly simulations

    simu = Simulation(
        path=str(data_path),
        start=start,
        end=end,
        ye=ye,
        comp=comp,
        term=term,
        filename=filename,
        dimensionality_reduction=dr_technique,
    )
    print(f"{term} loaded")

    simu.get_simulation_data()
    print(f"{term} prepared")

    # Extract time series through PCA
    simu.decompose()
    print(f"PCA applied on {term}")

    os.makedirs(f"{out_path}/simu_prepared/{term}", exist_ok=True)
    print(f"{out_path}/simu_prepared/{term} created")

    # Create dictionary and save:
    simu.save(f"{out_path}/simu_prepared", term)
    print(f"{term} saved at {out_path}/simu_prepared/{term}")

    return simu


def get_ocean_term(ocean_property, yaml_path: Path | str | None = None):
    """
    Retrieve an ocean-related term from an ``ocean_terms.yaml`` file.

    Parameters
    ----------
    ocean_property : str
        The key of the term to retrieve from the YAML under the ``Terms`` section.
        For example, ``"Temperature"`` to fetch the corresponding ocean temperature term.
    yaml_path : Path or str, optional
        Optional path to a custom ``ocean_terms.yaml`` file. If not provided,
        the function falls back to the packaged default file bundled with the
        ``nemo_spinup_forecast`` package.

    Returns
    -------
    str or None
        The term associated with the given property, or ``None`` if the file is missing
        or the property is not defined.

    Notes
    -----
    - When ``yaml_path`` is not provided, the function looks up the packaged
      ``ocean_terms.yaml`` via importlib.resources.
    - This function prints a short diagnostic message and returns ``None`` on failure.
    """
    try:
        if yaml_path is not None:
            yaml_path = Path(yaml_path).expanduser().resolve()
            with yaml_path.open("r") as f:
                terms = yaml.safe_load(f)
        else:
            # Fall back to packaged resource
            config_file = files("nemo_spinup_forecast.configs").joinpath(
                "ocean_terms.DINO.yaml"
            )
            with config_file.open("r") as f:
                terms = yaml.safe_load(f)

        return terms["Terms"][ocean_property]

    except FileNotFoundError:
        print(
            "\nCouldn't find 'ocean_terms.yaml'. Provide --ocean-terms path if needed.\n"
        )
        return None
    except KeyError:
        print(
            f"\nThe term '{ocean_property}' was not found in the "
            "'Terms' section of 'ocean_terms.yaml'.\n"
        )
        return None


def get_forecast_technique(yaml_path, forecast_techniques):
    """Retrieve a forecasting technique from the 'techniques_config.yaml' file.

    Parameters
    ----------
    yaml_path : Path
        The path to the 'techniques_config.yaml' or similarly named file
    forecast_techniques : dict
        A dictionary of available forecasting techniques.

    Returns
    -------
    ForecastTechnique
        An instance of the specified forecasting technique.

    Raises
    ------
    KeyError
        If the specified technique is not found in the `forecast_techniques` dictionary.
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    if config["Forecast_technique"]["name"] not in forecast_techniques:
        msg = (
            f"Forecast_technique {config['Forecast_technique']['name']} not found. "
            "Have you specified a valid forecasting technique in the config file?"
        )
        raise KeyError(msg)
    else:
        return forecast_techniques[config["Forecast_technique"]["name"]]


def get_dr_technique(yaml_path, dimensionality_reduction_techniques):
    """Retrieve a dimensionality reduction technique from a config file.

    Parameters
    ----------
    yaml_path : Path
        The path to the 'techniques_config.yaml' or similarly named file
    dimensionality_reduction_techniques : dict
        A dictionary of available dimensionality reduction techniques.

    Returns
    -------
    DimensionalityReductionTechnique
        An instance of the specified dimensionality reduction technique.

    Raises
    ------
    KeyError
        If the specified technique is not found in the
        `dimensionality_reduction_techniques` dictionary.
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    if config["DR_technique"]["name"] not in dimensionality_reduction_techniques:
        msg = (
            f"DR_technique {config['DR_technique']['name']} not found. "
            "Have you specified a valid dimensionality reduction "
            "technique in the config file?"
        )
        raise KeyError(msg)
    else:
        return dimensionality_reduction_techniques[config["DR_technique"]["name"]]
