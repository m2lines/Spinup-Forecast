import numpy as np
import pandas as pd
import pytest


# Use real prepared data via the setup_prediction_class fixture
@pytest.mark.parametrize(
    "setup_prediction_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_forecast_valid_input(setup_prediction_class):
    """
    Check that Forecast returns predictions, standard deviations, and metrics.

    This is tested with valid inputs.
    """
    pred, _infos = setup_prediction_class
    # Choose train_len slightly less than full length to generate test data
    train_len = len(pred) - 5
    steps = 3

    y_hats, y_stds, metrics = pred.parallel_forecast(train_len, steps)

    # Check types
    assert isinstance(y_hats, pd.DataFrame), "y_hats should be a DataFrame"
    assert isinstance(y_stds, pd.DataFrame), "y_stds should be a DataFrame"
    assert isinstance(metrics, list), "metrics should be a list"

    # Check shapes
    n_series = pred.data.shape[1]

    expected_shape = (steps, n_series)
    assert y_hats.shape == expected_shape, (
        f"y_hats shape {y_hats.shape} != expected {expected_shape}"
    )
    assert y_stds.shape == expected_shape, (
        f"y_stds shape {y_stds.shape} != expected {expected_shape}"
    )
    assert len(metrics) == n_series, (
        f"metrics length {len(metrics)} != expected {n_series}"
    )

    # Since train_len < len(data), metrics should be dicts with validation metrics
    for i, m in enumerate(metrics):
        assert isinstance(m, dict), (
            f"Metrics entry {i} should be a dict when test data is present"
        )
        # Check presence of expected metric keys
        for key in ["ma_true", "ma_pred", "mse", "dist_max", "std_true"]:
            assert key in m, f"Metric '{key}' missing in metrics dict for series {i}"


@pytest.mark.parametrize(
    "setup_prediction_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_forecast_no_test_data(setup_prediction_class):
    """Check that metrics are None when forecasting with no test data."""
    pred, _ = setup_prediction_class
    train_len = len(pred)
    steps = 0

    _y_hats, _y_stds, metrics = pred.parallel_forecast(train_len, steps)

    # All metrics entries should be None
    assert all(m is None for m in metrics), (
        "Metrics should be None when train_len equals data length (no test data)"
    )


# Use the setup_prediction_class fixture for real data


@pytest.mark.parametrize(
    "setup_prediction_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_forecast_ts_valid_input(setup_prediction_class):
    """
    Check that forecast_ts returns correct forecast arrays and metrics.

    This is tested with valid time-series inputs.
    """
    sim, _ = setup_prediction_class
    # Use first component, half of the data for training
    n = 1
    train_len = len(sim) // 2
    steps = len(sim) // 2

    y_hat, y_hat_std, metrics = sim.forecast_single_series(n, train_len, steps)

    # Expected shape: original data length + forecasted steps
    expected_shape = (steps,)
    assert isinstance(y_hat, np.ndarray), "y_hat should be a NumPy array"
    assert isinstance(y_hat_std, np.ndarray), "y_hat_std should be a NumPy array"
    assert y_hat.shape == expected_shape, (
        f"y_hat shape {y_hat.shape} != expected {expected_shape}"
    )
    assert y_hat_std.shape == expected_shape, (
        f"y_hat_std shape {y_hat_std.shape} != expected {expected_shape}"
    )

    # Since train_len < len(sim), metrics should be returned
    assert metrics is not None
    assert isinstance(metrics, dict)


@pytest.mark.parametrize(
    "setup_prediction_class",
    [
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("toce", "DINO_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_forecast_ts_no_test_data_no_steps(setup_prediction_class):
    """Check that metrics are None when forecasting zero steps with no test data."""
    sim, _ = setup_prediction_class
    n = 1
    train_len = len(sim)
    steps = 0

    y_hat, y_hat_std, metrics = sim.forecast_single_series(n, train_len, steps)

    # No test data available for validation => metrics must be None
    assert metrics is None, "Metrics should be None when no test data is available"

    # Expected shape: original data length + steps + 1
    # (when steps=0, still returns one extra point)
    expected_shape = (0,)
    assert isinstance(y_hat, np.ndarray), "y_hat should be a NumPy array"
    assert isinstance(y_hat_std, np.ndarray), "y_hat_std should be a NumPy array"
    assert y_hat.shape == expected_shape, (
        f"y_hat shape {y_hat.shape} != expected {expected_shape}"
    )
    assert y_hat_std.shape == expected_shape, (
        f"y_hat_std shape {y_hat_std.shape} != expected {expected_shape}"
    )


@pytest.mark.parametrize(
    "setup_prediction_class",
    [
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("toce", "DINO_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_forecast_ts_no_test_data_with_steps(setup_prediction_class):
    """Check that metrics are None when forecasting multiple steps with no test data."""
    sim, _ = setup_prediction_class
    n = 1
    train_len = len(sim)
    steps = 20

    y_hat, y_hat_std, metrics = sim.forecast_single_series(n, train_len, steps)

    # No test data available for validation => metrics must be None
    assert metrics is None, "Metrics should be None when no test data is available"

    # Expected shape: original data length + forecasted steps
    expected_shape = (steps,)
    assert isinstance(y_hat, np.ndarray), "y_hat should be a NumPy array"
    assert isinstance(y_hat_std, np.ndarray), "y_hat_std should be a NumPy array"
    assert y_hat.shape == expected_shape, (
        f"y_hat shape {y_hat.shape} != expected {expected_shape}"
    )
    assert y_hat_std.shape == expected_shape, (
        f"y_hat_std shape {y_hat_std.shape} != expected {expected_shape}"
    )


@pytest.mark.parametrize(
    "setup_prediction_class",
    [
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
        ("toce", "DINO_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_train_test_series_standard_case(setup_prediction_class):
    """
    Check the prepare function.

    This checks it splits the series, normalizes training data,
    and returns expected arrays.
    """
    sim, _ = setup_prediction_class
    # pick component 1, train_len < len(data)
    n = 1
    train_len = len(sim.data) // 2
    steps = 20

    mean, std, y_train, y_test, x_train, x_pred = sim.train_test_series(
        n, train_len, steps
    )

    # Compute expected mean and std from raw series
    series = sim.data[f"{sim.var}-{n}"]
    expected_mean = np.nanmean(series.iloc[:train_len])
    expected_std = np.nanstd(series.iloc[:train_len])
    assert np.isclose(mean, expected_mean), f"Mean {mean} != expected {expected_mean}"
    assert np.isclose(std, expected_std), f"Std {std} != expected {expected_std}"

    # y_train should be normalized: (raw - mean) / (2*std)
    raw_train = series.iloc[:train_len].values
    expected_y_train = (raw_train - expected_mean) / (2 * expected_std)
    assert np.allclose(y_train.flatten(), expected_y_train)

    # y_test should be raw values for remaining points
    raw_test = series.iloc[train_len:].values
    assert np.allclose(y_test.flatten(), raw_test)

    # x_train should be linspace
    expected_x_train, pas = np.linspace(0, 1, train_len, retstep=True)
    expected_x_train = expected_x_train.reshape(-1, 1)

    assert np.allclose(x_train, expected_x_train)

    # x_pred should span entire series length + steps with uniform spacing
    pas = x_train[1, 0] - x_train[0, 0]  # time step size
    total_len = steps
    expected_x_pred = np.linspace(1 + pas, 1 + steps * pas, steps).reshape(-1, 1)
    assert np.allclose(x_pred, expected_x_pred)
