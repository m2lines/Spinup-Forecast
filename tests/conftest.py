import pytest

from nemo_spinup_forecast.dimensionality_reduction import (
    DimensionalityReductionKernelPCA,
    DimensionalityReductionPCA,
)
from nemo_spinup_forecast.forecast import Predictions, Simulation, load_ts
from nemo_spinup_forecast.forecast_method import forecast_techniques
from nemo_spinup_forecast.utils import (
    create_run_dir,
    prepare,
)

DR_PARAMS = [
    (DimensionalityReductionPCA, 0.9),
    (DimensionalityReductionKernelPCA, 4),
]


@pytest.fixture(
    params=DR_PARAMS,
    ids=["PCA", "KernelPCA"],
)
def dr_technique(request):
    """Return a (DR class, comp) tuple."""
    return request.param


@pytest.fixture(
    params=list(forecast_techniques.values()),
    ids=list(forecast_techniques.keys()),
)
def forecast_technique(request):
    """Return the forecast technique instance."""
    return request.param


@pytest.fixture()
def setup_simulation_class(request, dr_technique):
    """Fixture to set up the simulation class."""
    # Parameters for the simulation class
    dr_class, comp = dr_technique
    path = "tests/data/nemo_data_e3/"
    start = 20  # Start year for the simulation
    end = 50  # End year for the simulation
    ye = True  # Indicates if the simulation is yearly
    term, filename = request.param  # Tuple (phycial property/term, file)

    simu = Simulation(
        path=path,
        start=start,
        end=end,
        ye=ye,
        comp=comp,
        term=term,
        filename=filename,
        dimensionality_reduction=dr_class,
    )

    return simu


@pytest.fixture()
def setup_prediction_class(request, tmp_path, dr_technique, forecast_technique):
    """Fixture to set up a prediction class."""
    data_path = "tests/data/nemo_data_e3/"

    dr_class, comp = dr_technique
    # create a per-run directory to store results
    out_dir = create_run_dir(str(tmp_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Reminder about handling index of tuple term, filename
    term, filename = request.param  # term to forecast, e.g., "ssh", "toce", "soce"
    start = 20
    end = 50
    ye = True  # Indicates if the simulation is yearly

    # Applies PCA and saves the results to disk
    prepare(term, filename, data_path, out_dir, start, end, ye, comp, dr_class)

    df, infos = load_ts(f"{out_dir}/simu_prepared/{term}", term)

    simu_ts = Predictions(term, df, infos, forecast_technique, dr_class)

    return simu_ts, infos
