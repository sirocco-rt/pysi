#!/usr/bin/env python3
"""Base class for Wind objects.

The base class which contains variables containing the parameters of the
wind, as well as the most basic variables which describe the wind.
"""

import pathlib
import re

import numpy
from astropy.constants import h, k_B

import pysi
import pysi.util.files
from pysi.wind import elements, enum

# Do it once, because apparently this is expensive
BOLTZMANN_CONSTANT = k_B.cgs.value
PLANCK_CONSTANT = h.cgs.value

DEFAULT_CELL_SPEC_BINS = 1000


class WindBase:
    """Base wind class for describing a wind object."""

    # Special methods ----------------------------------------------------------

    def __init__(self, root: str, directory: str | None = None, **kwargs) -> None:
        """Initialize the class.

        Parameters
        ----------
        root : str
            The root name of the simulation.
        directory : str
            The directory file path containing the simulation.
        **kwargs : dict
            Various other keywords arguments.

        """
        self.root, self.directory, _ = pysi.util.files.split_root_and_directory(root, directory)
        self.pf = f"{self.directory}/{root}.pf"
        self.version = kwargs.get("version")
        self.check_version()

        self.n_x = 0
        self.n_z = 0
        self.dims = (0, 0)
        self.n_cells = 0
        self.coord_type = enum.CoordSystem.UNKNOWN
        self.x_coords = []
        self.z_coords = []
        self.n_model_freq_bands = 0

        self.parameters = {}
        self.things_read_in = []
        self.ions_read_in = []

        # These units are the default in python. In a higher level class, you
        # should be able to modify the units

        self.distance_units = enum.DistanceUnits.CENTIMETRES
        self.velocity_units = enum.VelocityUnits.CENTIMETRES_PER_SECOND

        # Read in all the variables, spectra, etc.

        self.read_in_wind_parameters()
        self.read_in_wind_ions()
        self.read_in_wind_cell_spectra()
        self.read_in_wind_jnu_models()
        self.things_read_in = self.parameters.keys()
        self._set_axes_coords()

        # get a list of all the heating and cooling processes
        self.heating = [f for f in self.things_read_in if "heat_" in f]
        self.cooling = [f for f in self.things_read_in if "cool_" in f]

        self.descriptions = {
            "x": "left-hand lower cell corner x-coordinate, cm",
            "z": "left-hand lower cell corner z-coordinate, cm",
            "xcen": "cell centre x-coordinate, cm",
            "zcen": "cell centre z-coordinate, cm",
            "i": "cell index (column)",
            "j": "cell index (row)",
            "inwind": "is the cell in wind (0), partially in wind (1) or out of wind (<0)",
            "converge": "how many convergence criteria is the cell failing?",
            "v_x": "x-velocity, cm/s",
            "v_y": "y-velocity, cm/s",
            "v_z": "z-velocity, cm/s",
            "vol": "volume in cm^3",
            "rho": "density in g/cm^3",
            "ne": "electron density in cm^-3",
            "t_e": "electron temperature in K",
            "t_r": "radiation temperature in K",
            "h1": "H1 ion fraction",
            "he2": "He2 ion fraction",
            "c4": "C4 ion fraction",
            "n5": "N5 ion fraction",
            "o6": "O6 ion fraction",
            "dmo_dt_x": "momentum rate, x-direction",
            "dmo_dt_y": "momentum rate, y-direction",
            "dmo_dt_z": "momentum rate, z-direction",
            "ip": "U ionization parameter",
            "xi": "xi ionization parameter",
            "ntot": "total photons passing through cell",
            "nrad": "total wind photons produced in cell",
            "nioniz": "total ionizing photons passing through cell",
        }

    def __getitem__(self, key: str) -> numpy.ndarray:
        """Get the value of a key.

        Parameters
        ----------
        key : str
            The key to get the value of.

        Returns
        -------
        numpy.ndarray
            The value of the key.

        """
        # if no frac or den is no specified for an ion, default to fractional
        # populations
        if re.match("[A-Z]_i[0-9]+", key):  # matches ion specification, e.g. C_i04  # noqa: SIM102
            if re.match("[A-Z]_i[0-9]+$", key):  # but no type specification at the end, e.g. C_i04_frac
                key += "_frac"  # default to frac if not specified

        return self.parameters[key] if self.n_z > 1 else self.parameters[key][:, 0]

    def __str__(self) -> str:
        """Return a string representation of the Wind object.

        Returns
        -------
        str: A string representation of the Wind object in the format
            "Wind(root=<root> directory=<directory>)".

        """
        return f"Wind(root={self.root!r} directory={str(self.directory)!r})"

    def _set_axes_coords(self) -> None:
        """Set attributes for the x and z axes."""
        self.x_coords = (
            numpy.unique(self.parameters["x"])
            if self.coord_type == enum.CoordSystem.CYLINDRICAL
            else numpy.unique(self.parameters["r"])
        )
        if self.n_z > 1:
            self.z_coords = (
                numpy.unique(self.parameters["z"])
                if self.coord_type == enum.CoordSystem.CYLINDRICAL
                else numpy.unique(self.parameters["theta"])
            )
        else:
            self.z_coords = numpy.zeros_like(self.x_coords)

    def check_version(self) -> None:
        """Get the SIROCCO version from file if not already set.

        If the .sirocco-version file cannot be fine, the version is set to
        UNKNOWN.
        """
        if not self.version:
            try:
                with pathlib.Path(f"{self.directory}/.sirocco-version").open() as file_in:
                    self.version = file_in.read()
            except OSError:
                self.version = "unknown"

    def get_windsave_descriptions(self, key: str | None = None) -> None:
        """Print a description of the windsave parameters.

        Parameters
        ----------
        key : str | None, optional
            The parameter to get the description of, by default None which will
            print all.

        """
        if key is None:
            for name in self.descriptions:
                print(f"{name:10s} --  {self.descriptions[name]}")
        else:
            try:
                print(f"{key:10s} --  {self.descriptions[key]}")
            except KeyError:
                print(f"no description for parameter {key}")

    @staticmethod
    def _apply_jnu_model(
        model_type: int,
        model_p1: float,
        model_p2: float,
        band_frequency_bins: numpy.ndarray,
    ) -> numpy.ndarray:
        """Update the J_nu model for a frequency band.

        Parameters
        ----------
        model_type: int
            The type of model to use.
        model_p1: float
            The first parameter of the model.
        model_p2: float
            The second parameter of the model.
        band_frequency_bins: numpy.ndarray
            The frequency bins for the band

        Returns
        -------
        band_flux: List[numpy.ndarry]
            The computed flux for the band

        """
        if model_type == 1:  # Power-law model
            log_freq_bins = numpy.log10(band_frequency_bins)
            band_flux = 10 ** (model_p1 + log_freq_bins * model_p2)
        else:  # Exponential model
            inverse_temp = 1 / (model_p1 * BOLTZMANN_CONSTANT)
            band_flux = model_p2 * numpy.exp(-PLANCK_CONSTANT * band_frequency_bins * inverse_temp)

        return band_flux

    @staticmethod
    def _adjust_overlapping_bins(
        freq_min: numpy.ndarray, freq_max: numpy.ndarray, n_bins_per_band: int, *, tolerance: float = 1e-10
    ) -> numpy.ndarray:
        """Adjust the edges of the frequency bands to remove overlapping bins.

        Parameters
        ----------
        freq_min : numpy.ndarray
            The minimum frequency of the frequency bands.
        freq_max : numpy.ndarray
            The maximum frequency of the frequency bands.
        n_bins_per_band : int
            The number of frequency bins per band.
        tolerance : float, optional
            The floating point tolerance to use, by default 1e-10

        Returns
        -------
        freq_min_adj : numpy.ndarray
            The updated minimum frequencies
        freq_max_adj : numpy.ndarray
            The updated maximum frequencies

        """
        band_mask = numpy.array(
            [numpy.any(numpy.isclose(freq_min[i], freq_max, atol=tolerance)) for i in range(len(freq_min))]
        )
        log_dfreq = (numpy.log10(freq_max[band_mask]) - numpy.log10(freq_min[band_mask])) / n_bins_per_band
        freq_min_adj = freq_min.copy()
        freq_min_adj[band_mask] = freq_min[band_mask] * 10**log_dfreq

        return freq_min_adj, freq_max

    def _get_model_band_freq_bins(
        self,
        band_index_col: list[float] | numpy.ndarray,
        freq_min_col: list[float] | numpy.ndarray,
        freq_max_col: list[float] | numpy.ndarray,
        n_bins_per_band: int,
    ) -> numpy.ndarray:
        """Generate the frequency bins for each Jnu model band.

        Parameters
        ----------
        band_index_col : list[float] | numpy.ndarray
            The band indices column in the data.
        freq_min_col : list[float] | numpy.ndarray
            The frequency minimum column in the data.
        freq_max_col : list[float] | numpy.ndarray
            The frequency maximum column in the data.
        n_bins_per_band : int
            The number of frequency bins per band.

        Returns
        -------
        list[numpy.ndarray]
            The frequency bins for all bands in a 1D array.

        """
        band_index = numpy.unique(band_index_col)
        freq_min = numpy.zeros(len(band_index))
        freq_max = numpy.zeros(len(band_index))

        for i, band in enumerate(band_index):
            mask = band_index_col == band
            freq_min[i] = freq_min_col[mask].min()
            freq_max[i] = freq_max_col[mask].max()
            # There is a bug in SIROCCO which sometimes means that the fmin
            # and fmax columns in the .spec.txt file get swapped around. So we
            # need to swap them back around
            if freq_min[i] > freq_max[i]:
                freq_min[i], freq_max[i] = freq_max[i], freq_min[i]

        freq_min, freq_max = self._adjust_overlapping_bins(freq_min, freq_max, n_bins_per_band)
        return numpy.concatenate(
            [
                numpy.logspace(numpy.log10(freq_min[i]), numpy.log10(freq_max[i]), int(n_bins_per_band))
                for i in range(len(freq_min))
            ]
        )

    def _create_empty_parameter_array(self, parameter_names: list[str]) -> None:
        """Create an empty parameter array for each parameter in parameter_names.

        Parameters
        ----------
        parameter_names : list[str]
            The names of the parameters to create empty arrays for.

        """
        parameter_names = list(parameter_names)
        for parameter in parameter_names:
            if parameter not in self.parameters:
                self.parameters[parameter] = numpy.zeros((self.n_x, self.n_z), dtype=object)

    def get_elem_number_from_ij(self, i: int, j: int) -> int:
        """Get the wind element number for a given i and j index.

        Used when indexing into a 1D array, such as in Python itself.

        Parameters
        ----------
        i: int
            The i-th index of the cell.
        j: int
            The j-th index of the cell.

        """
        return int(self.n_z * i + j)

    def get_ij_from_elem_number(self, elem: int) -> tuple[int, int]:
        """Get the i and j index for a given wind element number.

        Used when converting a wind element number into two indices for use
        in this package.

        Parameters
        ----------
        elem: int
            The element number.

        """
        i = int(elem / self.n_z)
        j = int(elem - i * self.n_z)

        return i, j

    def read_in_wind_table(self, table: str) -> tuple[list[str], numpy.ndarray]:
        """Get variables for a specific table type.

        Parameters
        ----------
        table: str
            The type of table to read in, e.g. master, heat, etc.

        Returns
        -------
        table_header: List[str]
            The table headers for each column.
        table_parameters: numpy.ndarray
            An array of the numerical values of the table.

        """
        file_path = pathlib.Path(f"{self.directory}/{self.root}.{table}.txt")

        if file_path.is_file() is False:
            file_path = pathlib.Path(f"{file_path.parent!s}/tables/{file_path.stem}.txt")
            if file_path.is_file() is False:
                return [], {}

        table_header, table_parameters = pysi.util.files.read_file_with_header(file_path)

        return table_header, table_parameters

    def read_in_wind_jnu_models(self, n_bins_per_band: int = 250) -> None:
        """Read in the J_nu models for each cell.

        TODO: this should be simplified in the future.

        Parameters
        ----------
        n_bins_per_band: int
            The number of frequency bins to use for the model.

        """
        self._create_empty_parameter_array(["model_freq", "model_flux"])
        table_header, model_array = self.read_in_wind_table("spec")
        if model_array.size == 0:
            return
        self.n_model_freq_bands = n_bands = int(numpy.max(model_array[:, table_header.index("nband")])) + 1

        # Pre-compute the frequency bins for each band which is a massive time
        # save. This means we ignore cell min/max frequencies and use what
        # the photon banding is in the parameter file.
        inwind_array = model_array[model_array[:, table_header.index("inwind")] >= 0]
        freq_bins = self._get_model_band_freq_bins(
            inwind_array[:, table_header.index("nband")],
            inwind_array[:, table_header.index("fmin")],
            inwind_array[:, table_header.index("fmax")],
            n_bins_per_band,
        )

        # Indices of columns used in array - this is a lot faster than pandas
        # is for some reason
        try:
            model_type_index = table_header.index("spec_mod_type")
        except ValueError:
            model_type_index = table_header.index("spec_mod_")
        m1p1_index = table_header.index("pl_log_w")
        m1p2_index = table_header.index("pl_alpha")
        m2p1_index = table_header.index("exp_w")
        m2p2_index = table_header.index("exp_temp")

        # Ignore all numpy warnings because there are lots of overflows and
        # divisions by 0 which we don't care about in this case
        with numpy.errstate(all="ignore"):
            # The next block will loop over each cell and constuct a model for each
            # frequency band, and put that (and the frequency bins) into an array
            # for each cell.
            for cell_index in range(self.n_cells):
                i, j = self.get_ij_from_elem_number(cell_index)
                if self.parameters["inwind"][i, j] < 0:
                    continue

                model_flux = numpy.zeros(n_bands * n_bins_per_band)

                for band_index in range(n_bands):
                    offset = cell_index + band_index * self.n_cells
                    model_type = int(model_array[offset, model_type_index])
                    band_start, band_stop = band_index * n_bins_per_band, (band_index + 1) * n_bins_per_band
                    if model_type == 1:
                        p1 = model_array[offset, m1p1_index]
                        p2 = model_array[offset, m1p2_index]
                    else:
                        p2 = model_array[offset, m2p1_index]
                        p1 = model_array[offset, m2p2_index]
                    band_flux = self._apply_jnu_model(
                        model_type,
                        p1,
                        p2,
                        freq_bins[band_start:band_stop],
                    )
                    model_flux[band_start:band_stop] = band_flux

                self.parameters["model_freq"][i, j] = freq_bins
                self.parameters["model_flux"][i, j] = model_flux

    def read_in_wind_cell_spectra(self) -> None:
        """Read in the cell spectra."""
        self._create_empty_parameter_array(["spec_freq", "spec_flux"])
        spec_table_files = pysi.util.shell.find_file_with_pattern("{}*xspec.*.txt".format(self.root), self.directory)
        if len(spec_table_files) == 0:
            return

        for file in spec_table_files:
            file_header, file_array = pysi.util.files.read_file_with_header(file)
            file_header = file_header[1:]  # remove the Freq. entry

            # Go through each coord string and figure out the coords, and place
            # the spectrum into 1d/2d array

            for i, coord_string in enumerate(file_header):
                coords = numpy.array(coord_string[1:].split("_"), dtype=numpy.int32)
                # todo(ep): no idea why they have to be separate cases, but should investigate
                # print (coords, self.n_z, file_array.shape, i)
                if self.n_z > 1:
                    self.parameters["spec_flux"][coords[0], coords[1]] = file_array[:, i + 1]
                    self.parameters["spec_freq"][coords[0], coords[1]] = file_array[:, 0]
                else:
                    self.parameters["spec_flux"][coords[0], 0] = file_array[:, i + 1]
                    self.parameters["spec_freq"][coords[0], 0] = file_array[:, 0]

    def read_in_wind_ions(self, elements_to_read: list[str] = elements.ELEMENTS) -> None:
        """Read in the different ions in the wind.

        Parameters
        ----------
        elements_to_read: List[str], optional
            A list of atomic element names, e.g. H, He, whose ions in the wind
            will attempted to be read in. The default value is to try to read in
            all elements up to Cobalt.

        """
        n_read = 0

        # We need to loop over "frac" and "den" because ions are printed in
        # fractional populations or absolute density. The second loop is over
        # the elements passed to the function

        for ion_type in ["frac", "den"]:
            for element in elements_to_read:
                table_header, table_parameters = self.read_in_wind_table(f"{element}.{ion_type}")

                if not table_header:
                    continue

                for i, column in enumerate(table_header):
                    # the re.match here is to ignore any spatial parameters,
                    # e.g. x, z or i and j
                    if re.match("i[0-9]+", column) and column not in self.parameters:
                        ion_name = f"{element}_{column}_{ion_type}"
                        self.parameters[ion_name] = table_parameters[:, i].reshape(self.n_x, self.n_z)
                        self.ions_read_in.append(ion_name)

                n_read += 1

        if n_read == 0:
            raise OSError(f"Have been unable to read in any wind ion tables in {self.directory}")

    def read_in_wind_parameters(self) -> None:
        """Read in the different parameters which describe state of the wind."""
        n_read = 0

        for table in ["master", "heat", "gradient", "converge"]:
            table_header, table_parameters = self.read_in_wind_table(table)

            if not table_header:
                continue

            for i, column in enumerate(table_header):
                if column not in self.parameters:
                    self.parameters[column] = table_parameters[:, i]

            n_read += 1

        if n_read == 0:
            raise OSError(f"Have been unable to read in any wind parameter tables in {self.directory}")

        self.things_read_in = self.parameters.keys()

        # Determine the number of cells in the x and z direction, and the
        # coordinate type of the grid

        self.n_x = int(numpy.max(self.parameters["i"]) + 1)
        if "z" in self.things_read_in or "theta" in self.things_read_in:
            self.n_z = int(numpy.max(self.parameters["j"]) + 1)
        else:
            self.n_z = 1
        self.dims = (self.n_x, self.n_z)
        self.n_cells = int(self.n_x * self.n_z)

        if "r" in self.parameters and "theta" in self.parameters:
            self.coord_type = enum.CoordSystem.POLAR
        elif "r" in self.parameters:
            self.coord_type = enum.CoordSystem.SPHERICAL
        else:
            self.coord_type = enum.CoordSystem.CYLINDRICAL

        # Reshape the parameters into (nx, nz) which are currently just flat
        # arrays

        self.parameters = {col: val.reshape(self.n_x, self.n_z) for col, val in self.parameters.items()}
