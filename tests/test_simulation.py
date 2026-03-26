import os

import numpy as np
import pytest
import xarray as xr

from src.nemo_spinup_forecast.forecast import Simulation


@pytest.mark.parametrize(
    "term, filename, expected_file_pattern,expected_count",
    [
        # Valid terms with their respective file patterns
        ("toce", "DINO_1y_grid_T", "1y_grid_T.nc", 1),  # valid_temperature
        ("soce", "DINO_1y_grid_T", "1y_grid_T.nc", 1),  # valid_salinity
        ("ssh", "DINO_1m_To_1y_grid_T", "1m_To_1y_grid_T.nc", 1),  # valid_ssh
    ],
)
def test_get_data_valid_terms(term, filename, expected_file_pattern, expected_count):
    """Check get_data returns expected files for valid terms."""
    data_path = "tests/data/nemo_data_e3"

    # Run the get_data method
    files = Simulation.get_data(data_path, term, filename)  # TODO: Change argument names

    # Verify correct number of files found
    assert len(files) == expected_count, (
        f"Expected {expected_count} files, got {len(files)}"
    )
    # Files should be sorted
    assert files == sorted(files)

    # Check if all found files match the expected pattern for this variable
    for i, file in enumerate(files):
        assert os.path.dirname(file) == data_path
        assert expected_file_pattern in os.path.basename(file), (
            f"File {i} doesn't contain pattern {expected_file_pattern}"
        )
        assert filename in os.path.basename(file), (
            f"File {i} doesn't contain term {filename}"
        )


@pytest.mark.parametrize(
    "term, filename, expected_count",
    [
        # Valid term with nonexistent grid
        ("ssh", "nonexistent", 0),  # ssh_nonexistent_grid
    ],
)
def test_get_data_invalid_combinations(term, filename, expected_count):
    """Check get_data returns no files for invalid term-file combinations."""
    data_path = "tests/data/nemo_data_e3"

    # Run the get_data method
    files = Simulation.get_data(data_path, term, filename)  # TODO: Change argument names

    # Verify no files are found for invalid combinations
    assert len(files) == expected_count, (
        f"Expected no files for invalid term, got {len(files)}"
    )


@pytest.mark.parametrize(
    "setup_simulation_class, term, shape",
    [
        pytest.param(
            ("toce", "DINO_1y_grid_T.nc"),
            ("toce", "DINO_1y_grid_T.nc"),
            (36, 199, 62),
        ),
        pytest.param(
            ("soce", "DINO_1y_grid_T.nc"),
            ("soce", "DINO_1y_grid_T.nc"),
            (36, 199, 62),
        ),
        (
            ("ssh", "DINO_1m_To_1y_grid_T.nc"),
            ("ssh", "DINO_1m_To_1y_grid_T.nc"),
            (199, 62),
        ),
    ],
    indirect=["setup_simulation_class"],
)
# indirect parameterization of setup_simulation_class fixture
def test_get_attributes(setup_simulation_class, term, shape):
    """Check getAttributes returns correct shape, term, and time_dim.

    Notes
    -----
    See this issue for reason for faliure:
    https://github.com/m2lines/nemo-spinup-forecast/issues/58
    """
    simulation = setup_simulation_class

    simulation.get_attributes()

    # Expected spatial dimensions for this dataset
    assert simulation.shape == shape, f"Shape {simulation.shape} != expected {shape}"
    # Term should match the input parameter exactly
    expected_term_file = (simulation.term, simulation.filename)
    assert expected_term_file == term, f"Term {expected_term_file} != expected {term}"
    # Standard time dimension name for NEMO data
    assert simulation.time_dim == "time_counter", (
        f"Time dimension should be 'time_counter', got {simulation.time_dim}"
    )


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_getSimu(setup_simulation_class):
    """
    Check getSimu sets simulation DataArray and descriptive stats.

    Verifies that the simulation instance has a properly configured xarray DataArray
    and that the computed descriptive statistics (mean, std, min, max) match
    independently calculated values from the underlying data.
    """
    simu = setup_simulation_class

    # Check that 'simulation' attribute exists and is an xarray DataArray
    assert hasattr(simu, "simulation"), (
        "Simulation instance should have 'simulation' attribute"
    )
    assert isinstance(simu.simulation, xr.DataArray), (
        "'simulation' should be a xarray.DataArray"
    )
    # Check DataArray name matches variable name
    assert simu.simulation.name == simu.term, (
        f"DataArray name {simu.simulation.name} does not match term {simu.term}"
    )

    # Extract data values for manual computation
    data = simu.simulation.values

    # Compute expected descriptive statistics
    expected_mean = np.nanmean(data)
    expected_std = np.nanstd(data)
    expected_min = np.nanmin(data)
    expected_max = np.nanmax(data)

    # Check that desc dictionary contains correct keys and values
    for key in ["mean", "std", "min", "max"]:
        assert key in simu.desc, f"'{key}' should be in simu.desc"

    # Compare actual vs expected values
    assert np.isclose(simu.desc["mean"], expected_mean), "Mean calculation incorrect"
    assert np.isclose(simu.desc["std"], expected_std), "Std calculation incorrect"
    assert np.isclose(simu.desc["min"], expected_min), "Min calculation incorrect"
    assert np.isclose(simu.desc["max"], expected_max), "Max calculation incorrect"


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_load_file(setup_simulation_class):
    """Check loadFile returns correct DataArray and updates length."""
    simu = setup_simulation_class

    # Use the first file in the simulation's file list
    file_path = simu.files[0]

    # Reset length to zero to isolate this test
    simu.len = 0

    # Call loadFile
    data_array = simu.load_file(file_path)

    # Ensure the return is a loaded xarray.DataArray
    assert isinstance(data_array, xr.DataArray), (
        "loadFile should return an xarray.DataArray"
    )

    # The DataArray name should match the simulation term
    assert data_array.name == simu.term, (
        f"DataArray name {data_array.name} does not match term {simu.term}"
    )

    # After loading, self.len should equal the size along the time dimension
    assert simu.time_dim == "time_counter", (
        "Expected time dimension to be 'time_counter'"
    )
    expected_len = 50  # Known length for test data files
    assert simu.len == expected_len, (
        f"Length after loadFile ({simu.len}) does not match expected ({expected_len})"
    )


@pytest.fixture()
def dummy_simu():
    """Create bare Simulation instance for prepare tests."""
    simu = Simulation.__new__(Simulation)
    return simu


def test_prepare_slices_based_on_start_end(dummy_simu):
    """Check prepare slices data based on start and end indices."""
    # Create a simple DataArray of length 10 with sequential values
    data = xr.DataArray(np.arange(10, dtype=float), dims=("time",))
    dummy_simu.simulation = data
    dummy_simu.start = 3
    dummy_simu.end = 8
    dummy_simu.desc = {}

    dummy_simu.get_simulation_data(stand=False)

    # After slicing, simulation should be numpy array [3,4,5,6,7]
    expected = np.arange(3, 8, dtype=float)
    expected_len = len(expected)

    assert isinstance(dummy_simu.simulation, np.ndarray), (
        "Simulation should be a numpy array"
    )
    assert dummy_simu.len == expected_len, (
        f"Length {dummy_simu.len} != expected {expected_len}"
    )
    np.testing.assert_array_equal(
        dummy_simu.simulation, expected, err_msg="Simulation data != expected data"
    )


def test_prepare_slices_start_specified_end_none(dummy_simu):
    """Check prepare slices data using only start when end is None."""
    data = xr.DataArray(np.arange(10, dtype=float), dims=("time",))
    dummy_simu.simulation = data
    dummy_simu.start = 4
    dummy_simu.end = None
    dummy_simu.desc = {}

    dummy_simu.get_simulation_data(stand=False)

    # After slicing, simulation should be numpy array [4,5,6,7,8,9]
    expected = np.arange(4, 10, dtype=float)
    expected_len = len(expected)

    assert dummy_simu.len == expected_len, (
        f"Length {dummy_simu.len} != expected {expected_len}"
    )
    np.testing.assert_array_equal(dummy_simu.simulation, expected)


def test_prepare_standardisation_applied(dummy_simu):
    """Check prepare applies standardisation when stand=True.

    This normalises the data
    """
    data = xr.DataArray([0.0, 2.0, 4.0, 6.0], dims=("time",))
    dummy_simu.simulation = data
    dummy_simu.start = 0
    dummy_simu.end = None
    dummy_simu.desc = {}

    dummy_simu.get_simulation_data(stand=True)

    # Manually compute expected standardized values: (x - mean) / (2*std)
    mean = np.nanmean(data)
    std = np.nanstd(data)
    expected = ((data - mean) / (2 * std)).values

    np.testing.assert_allclose(
        dummy_simu.simulation,
        expected,
        err_msg="Standardization formula not applied correctly",
    )


def test_prepare_updates_desc_and_simulation(dummy_simu):
    """Check prepare updates simulation and descriptive statistics."""
    data = xr.DataArray([1.0, 2.0, 3.0, 5.0], dims=("time",))
    dummy_simu.simulation = data
    dummy_simu.start = 1
    dummy_simu.end = 4
    dummy_simu.desc = {}

    dummy_simu.get_simulation_data(stand=False)

    # After slicing, raw numpy array should match values[1:4]
    sliced = data.values[1:4]
    assert isinstance(dummy_simu.simulation, np.ndarray), (
        "Simulation should be numpy array"
    )
    np.testing.assert_array_equal(dummy_simu.simulation, sliced)

    # Check descriptive statistics computed on sliced data
    assert np.isclose(dummy_simu.desc["mean"], np.nanmean(sliced)), (
        "Mean calculation incorrect"
    )
    assert np.isclose(dummy_simu.desc["std"], np.nanstd(sliced)), (
        "Std calculation incorrect"
    )
    assert np.isclose(dummy_simu.desc["min"], np.nanmin(sliced)), (
        "Min calculation incorrect"
    )
    assert np.isclose(dummy_simu.desc["max"], np.nanmax(sliced)), (
        "Max calculation incorrect"
    )


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_standardize(setup_simulation_class):
    """Check standardize transforms simulation and preserves desc."""
    simu = setup_simulation_class
    simu.len = 0
    # Load simulation data and compute descriptive stats
    simu.get_simu()

    # Copy original data and desc
    original_data = simu.simulation.copy().values
    original_mean = simu.desc["mean"]
    original_std = simu.desc["std"]

    # Apply standardization
    simu.standardize()

    # The simulation attribute should remain an xarray.DataArray
    assert isinstance(simu.simulation, xr.DataArray), (
        "simulation should be an xarray.DataArray after standardize"
    )

    # Flatten arrays for comparison
    standardized_data = simu.simulation.values
    expected = (original_data - original_mean) / (2 * original_std)

    # Check that the data was standardized correctly (accounting for NaNs)

    (
        np.testing.assert_allclose(standardized_data, expected, equal_nan=True),
        ("standardize did not correctly transform simulation data"),
    )


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("soce", "DINO_1y_grid_T.nc"),  # 3D case (z,y,x)
        ("toce", "DINO_1y_grid_T.nc"),  # 3D case (z,y,x)
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),  # 2D case (y,x)
    ],
    indirect=True,
)
def test_rmseMap_zero_for_identical(setup_simulation_class):
    """Check rmseMap returns zeros for identical reconstruction."""
    sim = setup_simulation_class
    sim.get_simulation_data(stand=False)
    sim.dimensionality_reduction.set_from_simulation(sim)
    # Reconstruction identical to the truth - should yield zero very close to RMSE
    reconstruction = sim.simulation.copy()
    rmse_map = sim.dimensionality_reduction.rmse_map(reconstruction)

    # Expected RMSE map should be all zeros for perfect reconstruction
    expected_shape = sim.simulation.shape[1:]  # Spatial dimensions only
    expected = np.zeros(expected_shape)

    # Verify return type and shape
    assert isinstance(rmse_map, np.ndarray), "RMSE map should be numpy array"
    assert rmse_map.shape == expected_shape, (
        f"RMSE map shape {rmse_map.shape} != expected {expected_shape}"
    )

    # All values should be zero for identical reconstruction
    np.testing.assert_allclose(
        rmse_map,
        expected,
        err_msg="RMSE should be close to zero for identical reconstruction",
    )


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),  # 3D data (time, z, y, x)
        ("soce", "DINO_1y_grid_T.nc"),  # 3D data (time, z, y, x)
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),  # 2D data (time, y, x)
    ],
    indirect=True,
)
def test_rmseMap_real_data_with_limited_components_positive(setup_simulation_class):
    """
    Check rmseMap returns finite non-negative values.

    This is for single-component reconstruction with NaNs are at masked positions.
    """
    sim = setup_simulation_class
    sim.get_simulation_data(stand=False)

    # Apply PCA retaining the default variance fraction (or set comp explicitly)
    sim.decompose()
    # Reconstruct using only the first principal component - should have some error
    rec_one = sim.dimensionality_reduction.reconstruct_components(1)
    rmse_map = sim.dimensionality_reduction.rmse_map(rec_one)

    # Check return type and shape
    assert isinstance(rmse_map, np.ndarray), "RMSE map should be numpy array"
    assert rmse_map.shape == sim.shape, (
        f"Expected rmse_map shape {sim.shape}, got {rmse_map.shape}"
    )

    mask = sim.bool_mask.reshape(sim.shape)

    # Unmasked positions:
    # RMSE should be >= 0 and at least one should be > 0 (imperfect reconstruction)
    unmasked_vals = rmse_map[mask]
    assert np.all(unmasked_vals >= 0), "Negative RMSE values found at unmasked positions"
    assert np.any(unmasked_vals > 0), (
        "All RMSE values are zero at unmasked positions for "
        "limited-component reconstruction"
    )

    # Masked positions should remain NaN (no data available for RMSE calculation)
    assert np.all(np.isnan(rmse_map[~mask])), (
        "Expected NaN at masked positions in rmse_map"
    )


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),  # 3D data (time, z, y, x)
        ("soce", "DINO_1y_grid_T.nc"),  # 3D data (time, z, y, x)
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),  # 2D data (time, y, x)
    ],
    indirect=True,
)
def test_rmseValues_zero_for_identical(setup_simulation_class):
    """Check rmseValues returns zeros for identical reconstruction."""
    sim = setup_simulation_class
    sim.get_simulation_data(stand=False)
    sim.dimensionality_reduction.set_from_simulation(sim)
    # Reconstruction identical to the truth - should yield zero RMSE
    reconstruction = sim.simulation.copy()
    rmse_values = sim.dimensionality_reduction.rmse_values(reconstruction)

    # Verify return type
    assert isinstance(rmse_values, np.ndarray), "RMSE values should be numpy array"

    # Check the correct output shape based on data dimensionality
    if sim.z_size is not None:
        # For 3D data, shape should be (time, depth)
        # for RMSE per time step and depth level
        expected_shape = (sim.len, sim.z_size)
    else:
        # For 2D data, shape should be (time,) - RMSE per time step
        expected_shape = (sim.len,)

    assert rmse_values.shape == expected_shape, (
        f"RMSE values shape {rmse_values.shape} != expected {expected_shape}"
    )

    # All values should be zero for identical reconstruction
    np.testing.assert_allclose(
        rmse_values,
        0,
        err_msg="RMSE should be close to zero for identical reconstruction",
    )
