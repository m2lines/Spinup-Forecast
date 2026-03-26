import numpy as np
import pytest

from nemo_spinup_forecast.dimensionality_reduction import DimensionalityReductionPCA
from nemo_spinup_forecast.forecast import Predictions, Simulation, load_ts
from nemo_spinup_forecast.forecast_method import forecast_techniques
from nemo_spinup_forecast.utils import create_run_dir, prepare


@pytest.fixture()
def setup_simulation_class(request):
    """Set up a Simulation hardcoded to PCA."""
    path = "tests/data/nemo_data_e3/"
    term, filename = request.param
    simu = Simulation(
        path=path,
        start=20,
        end=50,
        ye=True,
        comp=0.9,
        term=term,
        filename=filename,
        dimensionality_reduction=DimensionalityReductionPCA,
    )
    return simu


@pytest.fixture()
def setup_prediction_class(request, tmp_path):
    """Prediction fixture hardcoded to PCA and GaussianProcessForecaster."""
    data_path = "tests/data/nemo_data_e3/"
    out_dir = create_run_dir(str(tmp_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    term, filename = request.param
    dr_technique = DimensionalityReductionPCA

    prepare(term, filename, data_path, out_dir, 20, 50, True, 0.9, dr_technique)
    df, infos = load_ts(f"{out_dir}/simu_prepared/{term}", term)

    forecast_technique = forecast_techniques["GaussianProcessForecaster"]
    simu_ts = Predictions(term, df, infos, forecast_technique, dr_technique)
    return simu_ts, infos


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_applyPCA_real_data(setup_simulation_class):
    """Check applyPCA works correctly on real data."""
    sim = setup_simulation_class
    # Prepare the data
    sim.get_simulation_data(stand=False)
    # After prepare, simulation attribute should be a NumPy array
    assert isinstance(sim.simulation, np.ndarray), (
        "Simulation should be numpy array after prepare"
    )
    initial_shape = sim.simulation.shape

    sim.decompose()
    components = sim.components

    # Components first dimension should equal time length
    expected_time_dim = sim.len

    msg = (
        f"Components time dimension {components.shape[0]} "
        f"!= expected {expected_time_dim}"
    )
    assert components.shape[0] == expected_time_dim, msg
    # Components second dimension should equal number of PCA components
    expected_n_components = sim.pca.n_components_
    msg = (
        f"Components feature dimension {components.shape[1]} "
        f"!= expected {expected_n_components}"
    )
    assert components.shape[1] == expected_n_components, msg

    # Boolean mask length should equal total number of spatial features
    feature_count = np.prod(initial_shape[1:])
    expected_mask_shape = (feature_count,)
    assert sim.bool_mask.shape == expected_mask_shape, (
        f"Mask shape {sim.bool_mask.shape} != expected {expected_mask_shape}"
    )

    # PCA components shape should match [n_components, n_features]
    expected_pca_shape = (sim.pca.n_components_, feature_count)
    msg = (
        f"PCA components shape {sim.pca.components_.shape} "
        f"!= expected {expected_pca_shape}"
    )
    assert sim.pca.components_.shape == expected_pca_shape, msg


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("soce", "DINO_1y_grid_T.nc"),  # 3D case (z,y,x)
        ("toce", "DINO_1y_grid_T.nc"),  # 3D case (z,y,x)
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),  # 2D case (y,x)
    ],
    indirect=True,
)
def test_getPC_real_data(setup_simulation_class):
    """Check getPC returns correct PC map shape, mask, and values for real data."""
    sim = setup_simulation_class

    # prepare real slice and compute PCA
    sim.get_simulation_data(stand=False)
    sim.decompose()

    std = sim.desc["std"]
    mean = sim.desc["mean"]
    mask = sim.bool_mask  # 1D boolean mask over flattened features
    shape = sim.shape  # e.g. (z,y,x) or (y,x)

    # Flattened mask length must equal product of spatial dimensions
    expected_mask_len = np.prod(shape)
    assert mask.shape == (expected_mask_len,), (
        f"Mask length {mask.shape} != expected ({expected_mask_len},)"
    )

    # Test every principal component
    for comp in range(sim.pca.n_components_):
        pc_map = sim.get_component(
            comp
        )  # Contribution of each coordinate to the components

        # Should return a numpy array with correct spatial shape
        assert isinstance(pc_map, np.ndarray), f"PC map {comp} should be numpy array"
        assert pc_map.shape == shape, (
            f"PC map {comp} shape {pc_map.shape} != expected {shape}"
        )

        flat_map = pc_map.ravel()
        comp_vals = sim.pca.components_[comp]

        # Build expected flattened map: transform component values back to original scale
        expected_flat = np.full(mask.shape, np.nan, dtype=float)
        expected_flat[mask] = 2 * comp_vals * std + mean

        np.testing.assert_allclose(
            flat_map,
            expected_flat,
            equal_nan=True,
            err_msg=f"PC map {comp} values incorrect",
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
def test_reconstruct_shape_and_mask_real_data(setup_simulation_class):
    """
    Check reconstruct returns correct arrays.

    This checks that the array is the correct shape, preserves the mask,
    and has finite values.
    """
    sim = setup_simulation_class

    # set up the PCA on the real data
    sim.get_simulation_data(stand=False)
    sim.decompose()

    # Check for a few n values: 1, all components, and beyond
    ns = [1, sim.pca.n_components_]
    for n in ns:
        rec = sim.dimensionality_reduction.reconstruct_components(n)

        # Should return array with correct shape: (time, *spatial_dims)
        expected_shape = (sim.len, *sim.shape)
        assert isinstance(rec, np.ndarray), (
            f"Reconstruction with {n} components should be numpy array"
        )
        assert rec.shape == expected_shape, (
            f"Reconstruction shape {rec.shape} != expected {expected_shape}"
        )

        # Integer mask should be updated to match spatial shape
        int_mask = sim.dimensionality_reduction.int_mask
        assert int_mask.shape == sim.shape, (
            f"Integer mask shape {int_mask.shape} != expected spatial shape {sim.shape}"
        )

        # For each time slice, masked positions should be NaN, unmasked finite
        flat_mask = int_mask.ravel()
        for t in range(rec.shape[0]):
            flat_rec = rec[t].ravel()
            # Masked positions (0) should contain NaN values
            masked_positions = flat_mask == 0
            assert np.all(np.isnan(flat_rec[masked_positions])), (
                f"Time {t}: masked positions should be NaN"
            )
            # Unmasked positions (1) should contain finite values
            unmasked_positions = flat_mask == 1
            assert np.all(np.isfinite(flat_rec[unmasked_positions])), (
                f"Time {t}: unmasked positions should be finite"
            )


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("soce", "DINO_1y_grid_T.nc"),
        ("toce", "DINO_1y_grid_T.nc"),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_reconstruct_full_components_recovers_original_data(setup_simulation_class):
    """Check reconstruct with all components recovers original data."""
    sim = setup_simulation_class

    sim.get_simulation_data(stand=False)
    # Use all available components for reconstruction
    sim.dimensionality_reduction.comp = None
    sim.decompose()

    # Reconstruct using all components - should recover original data
    rec_all = sim.dimensionality_reduction.reconstruct_components(sim.pca.n_components_)

    # Original simulation was stored as raw values before PCA
    orig = sim.simulation
    assert isinstance(orig, np.ndarray), "Original simulation should be numpy array"

    # Shapes should match exactly
    assert rec_all.shape == orig.shape, (
        f"Reconstruction shape {rec_all.shape} != original shape {orig.shape}"
    )

    # Values should match up to numerical tolerance for full reconstruction
    np.testing.assert_allclose(
        rec_all,
        orig,
        rtol=1e-5,
        atol=1e-1,
        equal_nan=True,
        err_msg="Full reconstruction should recover original data",
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
def test_rmseMap_real_data_full_components_zero(setup_simulation_class):
    """Check rmseMap returns zeros for full-component reconstruction on real data."""
    sim = setup_simulation_class
    # Prepare without standardization to retain raw values
    sim.get_simulation_data(stand=False)
    # Ensure simulation data is a NumPy array
    assert isinstance(sim.simulation, np.ndarray), "Simulation should be numpy array"

    # Use all available components to reconstruct full data
    sim.dimensionality_reduction.comp = None  # None defaults to using all components
    sim.decompose()
    rec_all = sim.dimensionality_reduction.reconstruct_components(sim.pca.n_components_)

    # Compute RMSE map between original and reconstructed data
    rmse_map = sim.dimensionality_reduction.rmse_map(rec_all)
    print("max: ", np.max(rec_all))

    # Check return type and shape
    assert isinstance(rmse_map, np.ndarray), "RMSE map should be numpy array"
    assert rmse_map.shape == sim.shape, (
        f"Expected rmse_map shape {sim.shape}, got {rmse_map.shape}"
    )

    # Boolean mask of valid (unmasked) positions, reshaped to spatial dimensions
    mask = sim.bool_mask.reshape(sim.shape)

    # Unmasked positions (True) should have zero RMSE
    # within tolerance for full reconstruction
    (
        np.testing.assert_allclose(rmse_map[mask], 0.0, atol=1e-1),
        ("Non-zero RMSE found at unmasked positions for full reconstruction"),
    )

    # Masked positions (False) should remain NaN (no data to compute RMSE)
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
def test_rmseValues_real_data_full_components_zero(setup_simulation_class):
    """Check rmseValues returns zeros for full-component reconstruction on real data."""
    sim = setup_simulation_class

    # Prepare raw numpy simulation and compute PCA for full reconstruction
    sim.get_simulation_data(stand=False)
    sim.dimensionality_reduction.comp = None
    sim.decompose()
    rec_all = sim.dimensionality_reduction.reconstruct_components(sim.pca.n_components_)

    # Compute RMSE values over time
    rmse_values = sim.dimensionality_reduction.rmse_values(rec_all)

    # Verify return type
    assert isinstance(rmse_values, np.ndarray), "RMSE values should be numpy array"

    # Check the correct output shape based on data dimensionality
    if sim.z_size is not None:
        # For 3D data, shape should be (time, depth) for
        # RMSE per time step and depth level
        expected_shape = (sim.len, sim.z_size)
    else:
        # For 2D data, shape should be (time,) - RMSE per time step
        expected_shape = (sim.len,)

    assert rmse_values.shape == expected_shape, (
        f"RMSE values shape {rmse_values.shape} != expected {expected_shape}"
    )

    # All RMSE values should be effectively zero for full reconstruction
    np.testing.assert_allclose(
        rmse_values,
        0,
        atol=1e-3,
        err_msg="RMSE should be near zero for full reconstruction",
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
def test_rmseOfPCA_real_full_zero(setup_simulation_class):
    """Check rmseOfPCA returns zeros RMSE values and map for full reconstruction."""
    sim = setup_simulation_class
    sim.dimensionality_reduction.comp = None
    sim.get_simulation_data(stand=False)
    sim.decompose()

    # Use all components for full reconstruction - should be nearly perfect
    n_comp = sim.pca.n_components_
    _rec, rmse_values, rmse_map = sim.error(n_comp)

    # Check RMSE values shape and near-zero values
    if sim.z_size is not None:
        expected_values_shape = (sim.len, sim.z_size)
    else:
        expected_values_shape = (sim.len,)

    assert rmse_values.shape == expected_values_shape, (
        f"RMSE values shape {rmse_values.shape} != expected {expected_values_shape}"
    )
    np.testing.assert_allclose(
        rmse_values,
        0,
        atol=5e-1,
        err_msg="RMSE values should be near zero for full reconstruction",
    )

    # Check RMSE map shape and near-zero values
    if sim.z_size is not None:
        expected_map_shape = (sim.z_size, sim.y_size, sim.x_size)
    else:
        expected_map_shape = (sim.y_size, sim.x_size)

    assert rmse_map.shape == expected_map_shape, (
        f"RMSE map shape {rmse_map.shape} != expected {expected_map_shape}"
    )
    np.testing.assert_allclose(
        rmse_map,
        0,
        atol=5e-1,
        err_msg="RMSE map should be near zero for full reconstruction",
    )


CASES = [
    ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ("soce", "DINO_1y_grid_T.nc"),
    ("toce", "DINO_1y_grid_T.nc"),
]

# Duplicate each case so BOTH fixtures receive the same tuple
PARAM_ROWS = [pytest.param(c, c, id=f"{c[0]}-{c[1]}") for c in CASES]

## NOTE: The test below is not solely pca specific


@pytest.mark.parametrize(
    ("setup_prediction_class", "setup_simulation_class"),
    PARAM_ROWS,
    indirect=["setup_prediction_class", "setup_simulation_class"],
)
def test_predictions_reconstruct(setup_prediction_class, setup_simulation_class):
    """
    Check that the reconstruct function rebuilds the time series correctly.

    This will check the result is the correct shape.
    """
    # setup prediction class
    pred, infos = setup_prediction_class
    sim = setup_simulation_class

    steps = 20

    # Forecast specified number of steps
    y_hat, _y_hat_std, _metrics = pred.parallel_forecast(len(pred), steps)

    # Reconstruct with n predicted components
    n = len(pred.info["pca"].components_)
    reconstructed_preds = sim.reconstruct(
        y_hat, n, infos, begin=0
    )  # TODO: Use simulation prediction class, setup simulation class fixture

    # Expected shape: forecast steps x original spatial dimensions
    expected_shape = (steps, *tuple(pred.info["shape"]))
    assert reconstructed_preds.shape == expected_shape, (
        f"Reconstructed shape {reconstructed_preds.shape} != expected {expected_shape}"
    )
