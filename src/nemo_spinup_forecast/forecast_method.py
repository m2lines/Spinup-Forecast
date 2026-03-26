from abc import ABC, abstractmethod

import pandas as pd
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    DotProduct,
    ExpSineSquared,
    WhiteKernel,
)  # , Matérn


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasters.

    Notes
    -----
    Subclasses must implement :meth:`apply_forecast`.
    """

    @abstractmethod
    def apply_forecast(self, y_train, x_train, x_pred):
        """
        Fit the model on training data and predict on new data.

        Parameters
        ----------
        y_train : array-like
            Target values used to train the forecaster/regressor.
        x_train : array-like
            Feature matrix aligned with `y_train`. Content/shape depends on the
            specific forecaster implementation.
        x_pred : array-like
            Feature matrix for which predictions should be generated.

        Returns
        -------
        tuple
            A tuple ``(y_hat, y_hat_std)`` where:

            * **y_hat** : ndarray
                Point forecasts.
            * **y_hat_std** : ndarray or None
                Estimated standard deviation of each forecast if available;
                otherwise ``None``.
        """


# Example usage of inheriting the BaseForecaster class and implementing apply_forecast


class DirectForecaster(BaseForecaster):
    """
    Forecaster that uses a regressor directly for predictions.

    Parameters
    ----------
    regressor : object
        A scikit-learn compatible regressor implementing ``fit`` and ``predict``.
        The ``predict`` method is expected to optionally accept
        ``return_std=True`` and return a tuple ``(y_hat, y_hat_std)`` if
        supported.

    Attributes
    ----------
    regressor : object
        Stored reference to the provided regressor.
    """

    def __init__(self, regressor):
        """
        Initialize the direct forecaster.

        Parameters
        ----------
        regressor : object
            See class docstring.
        """
        self.regressor = regressor

    def apply_forecast(self, y_train, x_train, x_pred):
        """
        Fit the regressor and make direct predictions.

        Parameters
        ----------
        y_train : array-like
            Training target series.
        x_train : array-like
            Training features corresponding to `y_train`.
        x_pred : array-like
            Feature matrix for which to obtain predictions.

        Returns
        -------
        tuple
            ``(y_hat, y_hat_std)`` where:

            * **y_hat** : ndarray
                Predicted values for `x_pred`.
            * **y_hat_std** : ndarray or None
                Predicted standard deviations if the underlying regressor
                supports ``return_std=True``; otherwise ``None``.

        Notes
        -----
        This method assumes the wrapped regressor's ``predict`` method supports
        the ``return_std`` argument. If it does not, an exception will be raised.
        """
        self.regressor.fit(x_train, y_train)

        y_hat, y_hat_std = self.regressor.predict(x_pred, return_std=True)

        return y_hat, y_hat_std


class RecursiveForecaster(BaseForecaster):
    """
    Forecaster that uses a recursive strategy for multi-step predictions.

    Parameters
    ----------
    regressor : object
        A scikit-learn compatible regressor.
    lags : int, default=10
        Number of lagged values of the target to include as features.
    window_size : int, default=10
        Size of the rolling window used to compute additional statistics.

    Attributes
    ----------
    regressor : object
        Stored reference to the provided regressor.
    lags : int
        Number of lags used in forecasting.
    window_size : int
        Rolling window size for feature engineering.
    """

    def __init__(self, regressor, lags=10, window_size=10):
        """
        Initialize the recursive forecaster.

        Parameters
        ----------
        regressor : object
            See class docstring.
        lags : int, default=10
            See class docstring.
        window_size : int, default=10
            See class docstring.
        """
        self.regressor = regressor
        self.lags = lags
        self.window_size = window_size

    def apply_forecast(self, y_train, _x_train, x_pred):
        """
        Fit a recursive forecaster using the provided regressor.

        Parameters
        ----------
        y_train : array-like
            Training target series.
        _x_train : array-like
            Not used in this recursive implementation.
        x_pred : array-like
            Feature matrix whose first dimension determines the number of
            forecast steps.

        Returns
        -------
        tuple
            ``(y_hat, y_hat_std)`` where:

            * **y_hat** : ndarray
                Predicted values for ``x_pred.shape[0]`` periods.
            * **y_hat_std** : ndarray
                Placeholder equal to ``y_hat`` since the recursive strategy
                does not compute prediction intervals.

        Notes
        -----
        - Rolling features are created with simple statistics (currently only
          mean) over the specified `window_size`.
        - Standard deviation estimates are not provided by
          :class:`skforecast.ForecasterRecursive`; therefore `y_hat` is reused
          as a placeholder.
        """
        # Create rolling features for the forecaster
        window_features = RollingFeatures(stats=["mean"], window_sizes=self.window_size)

        # Initialize the forecaster with the regressor
        forecaster = ForecasterRecursive(
            regressor=self.regressor,
            lags=self.lags,
            window_features=window_features,
        )

        # Fit on the training series
        forecaster.fit(pd.Series(y_train))

        # Predict the specified steps
        steps = x_pred.shape[0]
        y_hat = forecaster.predict(steps=steps)

        # Recursive forecaster does not return standard deviation for each point
        # Use y_hat as a placeholder for std
        y_hat_std = y_hat

        return y_hat, y_hat_std


def create_gp_regressor(
    data_length=45,
    trend_scale=0.1,
    irregularities_scale=10.0,
    noise_scale=2.0,
    optimize_restarts=8,
    random_seed=42,
):
    """
    Create a Gaussian Process regressor with basic parameter configuration.

    Parameters
    ----------
    data_length : int, default=45
        Length of the data, used to scale kernel parameters
    trend_scale : float, default=0.1
        Scaling factor for the trend kernel
    irregularities_scale : float, default=10.0
        Scaling factor for the irregularities kernel
    noise_scale : float, default=2.0
        Scaling factor for the noise kernel
    optimize_restarts : int, default=8
        Number of restarts for the optimizer
    random_seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    GaussianProcessRegressor
        Configured Gaussian Process regressor
    """
    # Scale factor based on data length
    scale_factor = 5.0 / data_length

    # Long-term trend kernel (linear behavior)
    trend_kernel = trend_scale * DotProduct(sigma_0=0.0)

    # Periodic kernel for irregularities
    irregularities_kernel = irregularities_scale * ExpSineSquared(
        length_scale=scale_factor, periodicity=scale_factor
    )

    # Noise kernel
    noise_kernel = noise_scale * WhiteKernel(noise_level=1.0)

    # Combine kernels
    kernel = irregularities_kernel + noise_kernel + trend_kernel

    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=optimize_restarts,
        random_state=random_seed,
    )


gp_recursive_forecaster = RecursiveForecaster(
    create_gp_regressor(), lags=10, window_size=10
)
gp_forecaster = DirectForecaster(create_gp_regressor())

# Creates a dictionary of Dict[classname -> class instance] key, value pairs
forecast_techniques = {
    "GaussianProcessForecaster": gp_forecaster,
    "GaussianProcessRecursiveForecaster": gp_recursive_forecaster,
}
