from abc import ABC, abstractmethod

import numpy as np
from sklearn.decomposition import (
    PCA,
    KernelPCA,
)


class DimensionalityReduction(ABC):
    """Abstract interface for dimensionality reduction techniques.

    Subclasses must implement the core API used throughout the code base.
    """

    @abstractmethod
    def decompose(self):
        """Fit the model and project the data into a lower-dimensional space.

        This method should set any attributes necessary for later reconstruction.
        """

    @staticmethod
    @abstractmethod
    def reconstruct_predictions():
        """Reconstruct data from previously predicted component scores.

        Implementations should not rely on instance state since the method is static.
        """

    @abstractmethod
    def reconstruct_components(self):
        """Reconstruct the original space using the stored component scores."""

    @abstractmethod
    def get_component(self):
        """Return a spatial map corresponding to a single component."""

    @abstractmethod
    def error(self):
        """Compute an error metric (e.g., RMSE) between reconstructions and truth."""

    @abstractmethod
    def set_from_simulation(self):
        """Attach metadata (shape, scaling, etc.) from a Simulation object."""


class DimensionalityReductionPCA(DimensionalityReduction):
    """Dimensionality reduction using classical Principal Component Analysis (PCA).

    Parameters
    ----------
    comp : int
        Number of principal components to retain.

    Attributes
    ----------
    components : ndarray or None
        Projected data of shape ``(time, comp)`` after :meth:`decompose`.
    comp : int
        Requested number of components.
    pca : sklearn.decomposition.PCA or None
        Fitted PCA object.
    shape : tuple[int, ...] or None
        Original spatial shape of a single time slice (e.g. ``(height, width)``).
    desc : dict or None
        Dictionary with keys such as ``'mean'`` and ``'std'`` used for rescaling.
    bool_mask : ndarray of bool or None
        Mask of valid (finite) features in flattened space.
    time_dim, len, simulation : various
        Metadata copied from the provided Simulation object
        via :meth:`set_from_simulation`.
    """

    def __init__(self, comp):
        self.components = None
        self.comp = comp
        self.pca = None
        self.shape = None
        self.desc = None
        print("Using normal PCA")

    def set_from_simulation(self, sim):
        """Copy metadata from a ``Simulation`` instance.

        Parameters
        ----------
        sim : object
            Object possessing attributes ``time_dim``, ``shape``, ``len``, ``desc`` and
            ``simulation`` (NumPy array). No type enforcement is done here.
        """
        self.time_dim = sim.time_dim
        self.shape = sim.shape
        self.len = sim.len
        self.desc = sim.desc
        self.simulation = sim.simulation

    def get_num_components(self):
        """Return the number of components retained by the fitted PCA.

        Returns
        -------
        int
            Number of components learned (``pca.n_components_``).
        """
        return self.pca.n_components_

    def decompose(self, simulation, length):
        """Run PCA on ``simulation``.

        Parameters
        ----------
        simulation : ndarray
            Shape ``(time, *spatial_dims)``.
        length : int
            Number of time steps to use.

        Returns
        -------
        components : ndarray
            Shape ``(length, comp)``.
        pca : PCA
            Fitted PCA object.
        bool_mask : ndarray[bool]
            Mask of finite features in the flattened space.
        """
        array = simulation.reshape(length, -1)

        self.bool_mask = np.asarray(np.isfinite(array[0, :]), dtype=bool)
        array_masked = array[:, self.bool_mask]
        pca = PCA(self.comp, whiten=False)
        self.components = pca.fit_transform(array_masked)
        self.pca = pca

        return self.components, self.pca, self.bool_mask

    @staticmethod
    def reconstruct_predictions(predictions, n, info, begin=0):
        """Rebuild fields from predicted PCA scores.

        Parameters
        ----------
        predictions : pandas.DataFrame
            Rows = time, columns = components.
        n : int
            Number of components to use.
        info : dict
            Keys: ``'mask'``, ``'shape'``, ``'pca'``, ``'desc'``.
        begin : int, default 0
            First row to reconstruct.

        Returns
        -------
        int_mask : ndarray[int]
            ``info['mask']`` reshaped.
        rec : ndarray
            Reconstructed array, rescaled.
        """
        rec = []
        int_mask = info["mask"].astype(np.int32).reshape(info["shape"])
        # Reconstruct each year/time interval from components
        for t in range(begin, len(predictions)):
            map_ = np.zeros((info["shape"]), dtype=np.float32)
            arr = np.array(
                list(predictions.iloc[t, :n]) + [0] * (len(info["pca"].components_) - n)
            )

            # Reshape arr to 2D (sample, n_components) for inverse_transform,
            # flatten inverse transform output, assign output to orginal ocean grid
            map_[int_mask == 1] = (
                info["pca"].inverse_transform(arr.reshape(1, -1)).flatten()
            )

            map_[int_mask == 0] = np.nan
            rec.append(map_)
        return int_mask, np.array(rec) * 2 * info["desc"]["std"] + info["desc"]["mean"]

    def reconstruct_components(self, n):
        """
        Reconstruct data using a specified number of principal components.

        Parameters
        ----------
        n : int
            The number of components used for reconstruction.

        Returns
        -------
        numpy.ndarray
            The reconstructed data.
        """
        rec = []
        # int_mask =   # Convert the boolean mask to int mask once
        self.int_mask = self.bool_mask.astype(np.int32).reshape(
            self.shape
        )  # Reshape to match the shape of map_

        # Reconstruct each year/time interval from components
        for t in range(len(self.components)):
            map_ = np.zeros(self.shape, dtype=np.float32)
            arr = np.array(
                list(self.components[t, :n]) + [0] * (len(self.pca.components_) - n)
            )
            # Reshape arr to 2D (sample, n_components) for inverse_transform,
            # flatten inverse transform output, assign output to orginal ocean grid
            map_[self.int_mask == 1] = self.pca.inverse_transform(
                arr.reshape(1, -1)
            ).flatten()
            map_[self.int_mask == 0] = np.nan
            rec.append(map_)
        return np.array(rec)

    def get_component(self, n):
        """
        Get an approximate kernel principal component map for component n.

        For non-linear kernels the mapping is implicit, so this is only a proxy.

        Parameters
        ----------
        n : int
            Component index.

        Returns
        -------
        ndarray
            ``self.shape``: A 2D map corresponding to the n-th component.
        """
        # map_ = np.zeros((np.product(self.shape)), dtype=float)
        map_ = np.zeros((np.prod(self.shape)), dtype=float)
        map_[~self.bool_mask] = np.nan
        map_[self.bool_mask] = self.pca.components_[n]
        map_ = map_.reshape(self.shape)

        map_ = 2 * map_ * self.desc["std"] + self.desc["mean"]
        return map_

    def error(self, n):
        """Alias of :meth:`rmse`.

        Parameters
        ----------
        n : int
            Components used in reconstruction.
        """
        return self.rmse(n)

    def rmse(self, n):
        """RMSE using ``n`` components.

        Parameters
        ----------
        n : int
            Components used in reconstruction.

        Returns
        -------
        reconstruction : ndarray
        rmse_values : ndarray
            Per-time RMSE.
        rmse_map : ndarray
            Per-point RMSE.
        """
        reconstruction = self.reconstruct_components(n)
        rmse_values = self.rmse_values(reconstruction) * 2 * self.desc["std"]
        rmse_map = self.rmse_map(reconstruction) * 2 * self.desc["std"]
        return reconstruction, rmse_values, rmse_map

    def rmse_values(self, reconstruction):
        """RMSE per time sample.

        Parameters
        ----------
        reconstruction : ndarray
            Output of :meth:`reconstruct_components`.

        Returns
        -------
        ndarray
            RMSE for each time index.
        """
        truth = self.simulation  # * 2 * self.desc["std"] + self.desc["mean"]
        rec = reconstruction  # * 2 * self.desc["std"] + self.desc["mean"]
        if len(np.shape(truth)) == 3:
            n = np.count_nonzero(~np.isnan(truth[0]))

            rmse_values = np.sqrt(np.nansum((truth - rec) ** 2, axis=(1, 2)) / n)
        else:
            n = np.count_nonzero(~np.isnan(self.simulation[0]), axis=(1, 2))

            rmse_values = np.nansum((truth - rec) ** 2, axis=(2, 3))
            for i in range(len(n)):
                rmse_values[:, i] = rmse_values[:, i] / n[i]
            rmse_values = np.sqrt(rmse_values)
        return rmse_values

    def rmse_map(self, reconstruction):
        """RMSE per spatial location.

        Parameters
        ----------
        reconstruction : ndarray
            Output of :meth:`reconstruct_components`.

        Returns
        -------
        ndarray
            ``self.shape`` RMSE map.
        """
        t = self.len
        return np.sqrt(np.sum((self.simulation[:] - reconstruction) ** 2, axis=0) / t)


class DimensionalityReductionKernelPCA(DimensionalityReduction):
    """Kernel PCA-based reduction.

    Parameters
    ----------
    comp : int
        Components to keep.
    kernel : str, default 'rbf'
        Kernel passed to :class:`KernelPCA`.
    **kwargs
        Extra ``KernelPCA`` args.

    Attributes
    ----------
    components : ndarray | None
        Projected data, ``(time, comp)``.
    pca : KernelPCA | None
        Fitted KernelPCA estimator.
    shape : tuple[int, ...] | None
        Spatial shape of one frame.
    desc : dict | None
        Scaling stats: ``{'mean', 'std'}``.
    bool_mask : ndarray[bool] | None
        Valid-feature mask after flattening.
    kernel : str
        Kernel name used.
    kwargs : dict
        Extra parameters forwarded to ``KernelPCA``.
    """

    def __init__(self, comp, kernel="rbf", **kwargs):
        # comp is the number of components
        self.comp = comp  # TODO: default value passing
        self.components = None  # Transformed (projected) data
        self.pca = None  # Will hold the KernelPCA instance
        self.shape = None  # Shape of the original spatial grid
        self.desc = None  # Dictionary containing metadata (e.g., mean, std)
        self.bool_mask = None  # Mask for valid features
        self.kernel = kernel  # Kernel type for KernelPCA
        self.kwargs = kwargs  # Additional parameters for KernelPCA

        print("Using Kernel PCA")

    # Create setter method for class variables
    def set_from_simulation(self, sim):
        """Copy metadata from ``sim`` (see PCA variant)."""
        self.time_dim = sim.time_dim
        self.shape = sim.shape
        self.len = sim.len
        self.desc = sim.desc
        self.simulation = sim.simulation

    def get_num_components(self):
        """Return ``self.comp``."""
        return self.comp

    def decompose(self, simulation, length):
        """Run KernelPCA.

        Parameters
        ----------
        simulation : ndarray
            ``(time, *spatial_dims)``.
        length : int
            Number of time steps.

        Returns
        -------
        components : ndarray
        pca : KernelPCA
        bool_mask : ndarray[bool]
        """
        # Reshape the simulation data: assume simulation is
        # originally (time, height, width)
        array = simulation.reshape(length, -1)
        # Create a boolean mask of valid (finite) features
        self.bool_mask = np.asarray(np.isfinite(array[0, :]), dtype=bool)
        array_masked = array[:, self.bool_mask]
        # Save the original spatial shape (for later reconstruction)
        self.shape = simulation.shape[1:]
        # Instantiate KernelPCA with inverse transform enabled
        kpca = KernelPCA(
            n_components=self.comp,
            kernel=self.kernel,
            fit_inverse_transform=True,
            **self.kwargs,
        )
        # Fit and transform the masked data
        self.components = kpca.fit_transform(array_masked)
        self.pca = kpca
        return self.components, self.pca, self.bool_mask

    @staticmethod
    def reconstruct_predictions(predictions, n, info, begin=0):
        """Rebuild fields from predicted PCA scores.

        Parameters
        ----------
        predictions : pandas.DataFrame
            Rows = time, columns = components.
        n : int
            Number of components to use.
        info : dict
            Keys: ``'mask'``, ``'shape'``, ``'pca'``, ``'desc'``.
        begin : int, default 0
            First row to reconstruct.

        Returns
        -------
        int_mask : ndarray[int]
            ``info['mask']`` reshaped.
        rec : ndarray
            Reconstructed array, rescaled.
        """
        rec = []
        int_mask = info["mask"].astype(np.int32).reshape(info["shape"])

        # Reconstruct each year/time interval from components
        for t in range(begin, len(predictions)):
            # Create an array for the t-th prediction;
            # pad with zeros for any missing components
            arr = np.array(
                list(predictions.iloc[t, :n]) + [0] * (info["pca"].n_components - n)
            )
            map_ = np.zeros(info["shape"], dtype=np.float32)

            # Reshape arr to 2D (sample, n_components) for inverse_transform,
            # flatten inverse transform output, assign output to orginal ocean grid
            map_[int_mask == 1] = (
                info["pca"].inverse_transform(arr.reshape(1, -1)).flatten()
            )

            map_[int_mask == 0] = np.nan
            rec.append(map_)
        # Scale the reconstruction back using provided descriptors
        return int_mask, np.array(rec) * 2 * info["desc"]["std"] + info["desc"]["mean"]

    def reconstruct_components(self, n):
        """Reconstruct array with first ``n`` kernel PCs.

        Parameters
        ----------
        n : int
            Number of components retained.

        Returns
        -------
        ndarray
            ``(time, *self.shape)`` reconstruction.
        """
        rec = []
        # Convert the boolean mask to an integer mask and reshape to match original grid
        self.int_mask = self.bool_mask.astype(np.int32).reshape(self.shape)

        # Reconstruct each year/time interval from components
        for t in range(len(self.components)):
            arr = np.array(
                list(self.components[t, :n]) + [0] * (self.pca.n_components - n)
            )
            map_ = np.zeros(self.shape, dtype=np.float32)

            # Reshape arr to 2D (sample, n_components) for inverse_transform,
            # flatten inverse transform output, assign output to orginal ocean grid
            map_[self.int_mask == 1] = self.pca.inverse_transform(
                arr.reshape(1, -1)
            ).flatten()
            map_[self.int_mask == 0] = np.nan
            rec.append(map_)
        return np.array(rec)

    def get_component(self, n):
        """
        Get an approximate kernel principal component map for component n.

        For non-linear kernels the mapping is implicit, so this is only a proxy.

        Parameters
        ----------
        n : int
            Component index.

        Returns
        -------
        ndarray
            ``self.shape``: A 2D map corresponding to the n-th component.
        """
        # Create a flat map, fill with NaNs for invalid entries
        map_ = np.zeros(np.prod(self.shape), dtype=float)
        map_[~self.bool_mask] = np.nan
        # For linear kernels one might extract an interpretable component;
        # here we use the dual coefficients as a proxy (if available)
        if hasattr(self.pca, "alphas_") and n < self.pca.alphas_.shape[1]:
            map_[self.bool_mask] = self.pca.alphas_[:, n]
        else:
            # Otherwise, default to zeros if the component is not available
            map_[self.bool_mask] = 0
        map_ = map_.reshape(self.shape)
        # Scale the component using the stored descriptor parameters
        map_ = 2 * map_ * self.desc["std"] + self.desc["mean"]
        return map_

    def error(self, n):
        """Alias of :meth:`rmse`."""
        return self.rmse(n)

    def rmse(self, n):
        """RMSE using ``n`` kernel PCs.

        Parameters
        ----------
        n : int
            Components used in reconstruction.

        Returns
        -------
        reconstruction : ndarray
        rmse_values : ndarray
            Per-time RMSE.
        rmse_map : ndarray
            Per-point RMSE.
        """
        reconstruction = self.reconstruct_components(n)
        rmse_values = self.rmse_values(reconstruction) * 2 * self.desc["std"]
        rmse_map = self.rmse_map(reconstruction) * 2 * self.desc["std"]

        return reconstruction, rmse_values, rmse_map

    def rmse_values(self, reconstruction):
        """RMSE per time sample.

        Parameters
        ----------
        reconstruction : ndarray
            Output of :meth:`reconstruct_components`.

        Returns
        -------
        ndarray
            RMSE for each time index.
        """
        truth = self.simulation  # Assumes self.simulation is set externally
        rec = reconstruction
        if len(np.shape(truth)) == 3:
            valid_count = np.count_nonzero(~np.isnan(truth[0]))
            rmse_values = np.sqrt(
                np.nansum((truth - rec) ** 2, axis=(1, 2)) / valid_count
            )
        else:
            valid_count = np.count_nonzero(~np.isnan(self.simulation[0]), axis=(1, 2))
            rmse_values = np.nansum((truth - rec) ** 2, axis=(2, 3))
            for i in range(len(valid_count)):
                rmse_values[:, i] = rmse_values[:, i] / valid_count[i]
            rmse_values = np.sqrt(rmse_values)
        return rmse_values

    def rmse_map(self, reconstruction):
        """RMSE per spatial location.

        Parameters
        ----------
        reconstruction : ndarray
            Output of :meth:`reconstruct_components`.

        Returns
        -------
        ndarray
            ``self.shape`` RMSE map.
        """
        t = (
            self.len
        )  # Assumes self.len is defined elsewhere (e.g., number of time steps)
        return np.sqrt(np.sum((self.simulation[:] - reconstruction) ** 2, axis=0) / t)


# Creates a dictionary of Dict[classname -> class] key, value pairs
dimensionality_reduction_techniques = {
    "PCA": DimensionalityReductionPCA,
    "KernelPCA": DimensionalityReductionKernelPCA,
}
