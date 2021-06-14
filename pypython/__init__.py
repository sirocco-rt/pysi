#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pypython - making using Python a wee bit easier.

pypython is a companion python package to handle and analyse the data which
comes out of a python simulation.
"""

import copy
import pkgutil
import re
import textwrap
import time
from os import listdir, path, remove
from pathlib import Path
from platform import system
from shutil import which
from subprocess import PIPE, Popen, run

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import boxcar, convolve

from pypython.constants import CMS_TO_KMS, PARSEC, PI, VLIGHT
from pypython.math import vector
from pypython.physics.blackhole import gravitational_radius
from pypython.simulation.grid import get_parameter_value

# Functions --------------------------------------------------------------------

MINIMUM_PY_VERSION = "85"
MINIMUM_PY_SUB_VERSION = ""
PY_VERSION = ""

# Constants --------------------------------------------------------------------


def check_python_version():
    """Check the version of Python available in $PATH.

    There are a number of features in this package which are not
    available in older versions of Python.
    """

    command = run("py --version", stdout=PIPE, stderr=PIPE, shell=True)
    stdout = command.stdout.decode("utf-8").split("\n")
    stderr = command.stderr.decode("utf-8")

    if stderr:
        raise SystemError(f"{stderr}")

    version = None
    for line in stdout:
        if line.startswith("Python Version"):
            version = line[len("Python Version") + 1:]

    if version is None:
        raise SystemError("Unable to determine Python version")

    sub_version = version.lstrip("0123456789")
    main_version = version[:version.index(sub_version)]

    if main_version < MINIMUM_PY_VERSION or sub_version < MINIMUM_PY_VERSION:
        raise SystemError(f"Python version {main_version}{sub_version} below minimum version of {MINIMUM_PY_VERSION}"
                          f"{MINIMUM_PY_SUB_VERSION}")

    return version


def cleanup_data(fp=".", verbose=False):
    """Remove data symbolic links created by Python.

    Search recursively from the specified directory for symbolic links named
    data. This script will only work on Unix systems where the find command is
    available.
    todo: update to a system agnostic method to find symbolic links like pathlib

    Parameters
    ----------
    fp: str
        The starting directory to search recursively from for symbolic links
    verbose: bool [optional]
        Enable verbose output

    Returns
    -------
    n_del: int
        The number of symbolic links deleted
    """
    n_del = 0

    os = system().lower()
    if os != "darwin" and os != "linux":
        raise OSError("your OS does not work with this function, sorry!")

    # - type l will only search for symbolic links
    cmd = f"cd {fp}; find . -type l -name 'data'"
    stdout, stderr = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True).communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    if stderr:
        print("sent from stderr")
        print(stderr)

    if stdout and verbose:
        print(f"deleting data symbolic links in the following directories:\n\n{stdout[:-1]}")
    else:
        print("no data symlinks to delete")
        return n_del

    directories = stdout.split()

    for directory in directories:
        current = fp + directory[1:]
        cmd = f"rm {current}"
        stdout, stderr = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True).communicate()
        if stderr:
            print(stderr.decode("utf-8"))
        else:
            n_del += 1

    return n_del


def get_files(pattern, fp="."):
    """Find files of the given pattern recursively.

    Used to blah blah

    Parameters
    ----------
    pattern: str
        Patterns to search recursively for, i.e. *.pf, *.spec, tde_std.pf
    fp: str [optional]
        The directory to search from, if not specified in the pattern.
    """

    files = [str(file_) for file_ in Path(f"{fp}").rglob(pattern)]
    if ".pf" in pattern:
        files = [file_ for file_ in files if "out.pf" not in file_ and "py_wind" not in file_]

    try:
        files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    except TypeError as e:
        print(f"{get_files.__name__}: have been unable to sort output, be careful as results may not be reproducible")

    return files


def get_array_index(x, target):
    """Return the index for a given value in an array.

    This function will not be happy if you pass an array with duplicate values.
    It will always return the first instance of the duplicate array.

    Parameters
    ----------
    x: np.ndarray
        The array of values.
    target: float
        The value, or closest value, to find the index of.

    Returns
    -------
    The index for the target value in the array x.
    """
    if target < np.min(x):
        return 0
    if target > np.max(x):
        return -1

    return np.abs(x - target).argmin()


def get_root_name(fp):
    """Get the root name of a Python simulation.

    Extracts both the file path and the root name of the simulation.

    Parameters
    ----------
    fp: str
        The directory path to a Python .pf file

    Returns
    -------
    root: str
        The root name of the Python simulation
    where: str
        The directory path containing the provided Python .pf file
    """
    if type(fp) is not str:
        raise TypeError("expected a string as input for the file path, not whatever you put")

    dot = fp.rfind(".")
    slash = fp.rfind("/")

    root = fp[slash + 1:dot]
    fp = fp[:slash + 1]

    if fp == "":
        fp = "./"

    return root, fp


def smooth_array(array, width):
    """Smooth a 1D array of data using a boxcar filter.

    Parameters
    ----------
    array: np.array[float]
        The array to be smoothed.
    width: int
        The size of the boxcar filter.

    Returns
    -------
    smoothed: np.ndarray
        The smoothed array
    """
    if width is None or width == 0:
        return array

    array = np.reshape(array, (len(array), ))  # todo: why do I have to do this? safety probably

    return convolve(array, boxcar(width) / float(width), mode="same")


def create_wind_save_tables(root, fp=".", ion_density=False, cell_spec=False, verbose=False):
    """Run windsave2table in a directory to create the standard data tables.

    The function can also create a root.all.complete.txt file which merges all
    the data tables together into one (a little big) file.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    fp: str
        The directory where windsave2table will run
    ion_density: bool [optional]
        Use windsave2table in the ion density version instead of ion fractions
    cell_spec: bool [optional]
        Use windsave2table to get the cell spectra.
    verbose: bool [optional]
        Enable verbose output
    """
    in_path = which("windsave2table")
    if not in_path:
        raise OSError("windsave2table not in $PATH and executable")

    files_before = listdir(fp)

    command = f"cd {fp};"
    if not Path(f"{fp}/data").exists():
        command += "Setup_Py_Dir;"
    command += "windsave2table"
    if ion_density:
        command += " -d"
    if cell_spec:
        command += " -xall"
    command += f" {root}"

    cmd = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = cmd.communicate()

    files_after = listdir(fp)

    if verbose:
        print(stdout.decode("utf-8"))
    if stderr:
        print("There may have been a problem running windsave2table")
        print(stderr.decode("utf-8"))

    # Move the new files in fp/tables

    s = set(files_before)
    new_files = [x for x in files_after if x not in s]
    Path(f"{fp}/tables").mkdir(exist_ok=True)
    for new in new_files:
        try:
            Path(f"{fp}/{new}").rename(f"{fp}/tables/{new}")
        except PermissionError:
            time.sleep(1.5)
            Path(f"{fp}/{new}").rename(f"{fp}/tables/{new}")

    return cmd.returncode


def run_py_wind(root, commands, fp="."):
    """Run py_wind with the provided commands.

    Parameters
    ----------
    root: str
        The root name of the model.
    commands: list[str]
        The commands to pass to py_wind.
    fp: [optional] str
        The directory containing the model.

    Returns
    -------
    output: list[str]
        The stdout output from py_wind.
    """
    cmd_file = f"{fp}/.tmpcmds.txt"

    with open(cmd_file, "w") as f:
        for i in range(len(commands)):
            f.write(f"{commands[i]}\n")

    sh = Popen(f"cd {fp}; py_wind {root} < .tmpcmds.txt", stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = sh.communicate()
    if stderr:
        print(stderr.decode("utf-8"))

    remove(cmd_file)

    return stdout.decode("utf-8").split("\n")


# These are put here to solve a circular dependency ----------------------------

SPECTRUM_UNITS_LNU = "erg/s/Hz"
SPECTRUM_UNITS_FNU = "erg/s/cm^-2/Hz"
SPECTRUM_UNITS_FLM = "erg/s/cm^-2/A"
SPECTRUM_UNITS_UNKNOWN = "unknown"

WIND_COORD_TYPE_CYLINDRICAL = WIND_COORD_TYPE_CARTESIAN = "rectilinear"
WIND_COORD_TYPE_POLAR = "polar"
WIND_COORD_TYPE_SPHERICAL = "spherical"
WIND_COORD_TYPE_UNKNOWN = "unknown"

WIND_VELOCITY_UNITS_KMS = "kms"
WIND_VELOCITY_UNITS_CMS = "cms"
WIND_VELOCITY_UNITS_LIGHT = "c"

WIND_DISTANCE_UNITS_CM = "cm"
WIND_DISTANCE_UNITS_RG = "rg"

# Spectrum class ---------------------------------------------------------------


class Spectrum:
    """A class to store PYTHON .spec and .log_spec files.

    The Python spectra are read in and stored within a dict of dicts,
    where each column name is the spectrum name and the columns in that
    dict are the names of the columns in the spectrum file. The data is
    stored as numpy arrays.
    """
    def __init__(self, root, fp=".", default=None, log_spec=True, smooth=None, distance=None, delim=None):
        """Create the Spectrum object.

        Construct the file path of the spectrum files given the
        root, directory and whether the logarithmic spectrum is used or not.
        The different spectra are then read in, with either the .spec or the first
        spectrum file read in being the default index choice.

        Parameters
        ----------
        root: str
            The root name of the model.
        fp: str [optional]
            The directory containing the model.
        default: str [optional]
            The default spectrum to make the available spectrum for indexing.
        log_spec: bool [optional]
            Read in the logarithmic version of the spectra.
        smooth: int [optional]
            The amount of smoothing to use.
        distance: float [optional]
            The distance of the spectrum flux, in units of parsec.
        delim: str [optional]
            The deliminator in the spectrum file.
        """
        self.root = root
        self.fp = fp
        if self.fp[-1] != "/":
            self.fp += "/"
        self.pf = self.fp + self.root + ".pf"

        self.log_spec = log_spec
        if default and self.log_spec:
            if not default.startswith("log_"):
                default = "log_" + default

        # These are for the "current/target" spectrum. Anything with all_ are
        # for recording all of the spectra

        self.spectrum = {}
        self.columns = ()
        self.inclinations = ()
        self.n_inclinations = 0
        self.units = SPECTRUM_UNITS_UNKNOWN
        self.distance = 100

        self.all_spectrum = {}
        self.all_columns = {}
        self.all_inclinations = {}
        self.all_n_inclinations = {}
        self.all_units = {}
        self.all_distance = {}

        # The next method call reads in the spectrum and initializes the above
        # member variables. We also keep track of what spectra have been loaded
        # in and set the "target" spectrum for indexing

        self.get_spectra(delim)
        self.available = tuple(self.all_spectrum.keys())

        # Now set the units, etc., to the target spectrum. If default is
        # provided, then this is used as the default spectrum.

        if default:
            if default in self.available:
                self.current = default
            else:
                raise ValueError(f"{self.root}.{default} is not available as it has not been read in")
        else:
            self.current = self.available[0]
        self._set_current(self.current)

        if distance:
            self.rescale_flux(distance)

        # self.original is a variable which keeps a copy of the spectrum for
        # safe keeping if it is smoothed

        self.original = None

        if smooth:
            self.smooth(smooth)

    def get_spectra(self, delim=None):
        """Read in a spectrum file given in self.filepath. The spectrum is
        stored as a dictionary in self.spectrum where each key is the name of
        the columns.

        Parameters
        ----------
        delim: str [optional]
            A custom delimiter, useful for reading in files which have sometimes
            between delimited with commas instead of spaces.
        """
        n_read = 0
        files_to_read = ["spec", "spec_tot", "spec_tot_wind", "spec_wind", "spec_tau"]

        for spec_type in files_to_read:
            fp = self.fp + self.root + "."
            if self.log_spec and spec_type != "spec_tau":
                spec_type = "log_" + spec_type
            fp += spec_type
            if not path.exists(fp):
                continue

            n_read += 1
            self.all_spectrum[spec_type] = {}
            self.all_units[spec_type] = SPECTRUM_UNITS_UNKNOWN
            self.all_distance[spec_type] = 0.0

            with open(fp, "r") as f:
                spectrum_file = f.readlines()

            # Read in the spectrum file, ignoring empty lines and lines which have
            # been commented out by # at the beginning
            # todo: need some method to detect incorrect syntax

            spectrum = []

            for line in spectrum_file:
                line = line.strip()
                if delim:
                    line = line.split(delim)
                else:
                    line = line.split()
                if "Units:" in line:
                    self.all_units[spec_type] = line[4][1:-1]
                    if self.all_units[spec_type] in [SPECTRUM_UNITS_FLM, SPECTRUM_UNITS_FNU]:
                        self.all_distance[spec_type] = float(line[6])
                    else:
                        self.all_distance[spec_type] = 0
                if len(line) == 0 or line[0] == "#":
                    continue
                spectrum.append(line)

            # Extract the header columns of the spectrum. This assumes the first
            # read line in the spectrum is the header. If no header is found, then
            # the columns are numbered instead

            header = []

            if spectrum[0][0] == "Freq." or spectrum[0][0] == "Lambda":
                for i, column_name in enumerate(spectrum[0]):
                    if column_name[0] == "A":
                        j = column_name.find("P")
                        column_name = column_name[1:j].lstrip("0")  # remove leading 0's
                    header.append(column_name)
                spectrum = np.array(spectrum[1:], dtype=np.float64)
            else:
                header = np.arange(len(spectrum[0]))

            # Add the actual spectrum to the spectrum dictionary, the keys of the
            # dictionary are the column names as given above. Set the header and
            # also the inclination angles here as well

            for i, column_name in enumerate(header):
                self.all_spectrum[spec_type][column_name] = spectrum[:, i]

            inclinations = []

            for col in header:
                if col.isdigit() and col not in inclinations:
                    inclinations.append(col)

            self.all_columns[spec_type] = tuple(header)
            self.all_inclinations[spec_type] = tuple(inclinations)
            self.all_n_inclinations[spec_type] = len(inclinations)

        if n_read == 0:
            raise IOError(f"Unable to open any spectrum files for {self.root} in {self.fp}")

    def rescale_flux(self, distance):
        """Rescale the flux to the given distance.

        Parameters
        ----------
        distance: float or int
            The distance to scale the flux to.
        """
        if type(distance) is not float and type(distance) is not int:
            raise ValueError("distance is not a float or integer")

        for spectrum in self.available:
            if self.all_units[spectrum] == SPECTRUM_UNITS_LNU:
                continue
            for key in self.all_spectrum[spectrum].keys():
                if key in ["Lambda", "Freq."]:
                    continue
                self.all_spectrum[spectrum][key] *= (self.all_distance[spectrum] * PARSEC)**2 / (distance * PARSEC)**2
            self.all_distance[spectrum] = distance

        self.distance = distance

    def smooth(self, width=5, to_smooth=None):
        """Smooth the spectrum flux/luminosity bins.

        If this is used after the spectrum has already been smoothed, then the
        "original" is copied back into the spectrum before smoothing again. This
        way the function does not smooth an already smoothed spectrum.

        Parameters
        ----------
        width: int [optional]
            The width of the boxcar filter (in bins).
        to_smooth: list or tuple of strings [optional]
            A list or tuple
        """
        if self.original is None:
            self.original = copy.deepcopy(self.spectrum)
        else:
            self.spectrum = copy.deepcopy(self.original)

        if to_smooth is None:
            to_smooth = ("Created", "WCreated", "Emitted", "CenSrc", "Disk", "Wind", "HitSurf", "Scattered") + tuple(
                self.inclinations)
        elif type(to_smooth) is str:
            to_smooth = to_smooth,
        elif type(to_smooth) is tuple or type(to_smooth) is list:
            pass
        else:
            raise ValueError("unknown format for argument to_smooth, must be a tuple/list of str or str")

        # Loop over each available spectrum and smooth it

        for key in self.available:
            if key == "spec_tau":  # todo: cleaner way to skip spec_tau
                continue
            for thing_to_smooth in to_smooth:
                try:
                    self.all_spectrum[key][thing_to_smooth] = smooth_array(self.all_spectrum[key][thing_to_smooth],
                                                                           width)
                except KeyError:
                    pass  # some spectra do not have the inclination angles...

    def restore_original_spectra(self):
        """Restore the spectrum to its original unsmoothed form."""

        self.spectrum = copy.deepcopy(self.original)

    def _set_current(self, target):
        """Set the current target spectrum.

        Parameters
        ----------
        target: str
            The name of the spectrum to become the current.
        """
        if target not in self.available:
            raise IndexError(f"spectrum {target} is not available: available are {self.available}")

        self.current = target
        self.spectrum = self.all_spectrum[target]
        self.columns = self.all_columns[target]
        self.inclinations = self.all_inclinations[target]
        self.n_inclinations = self.all_n_inclinations[target]
        self.units = self.all_units[target]
        self.distance = self.all_distance[target]

    def set(self, name):
        """Set a spectrum as the default.

        Sets a different spectrum to be the currently available spectrum for
        indexing.

        Parameters
        ----------
        name: str
            The name of the spectrum, i.e. log_spec or spec_tot, etc. The
            available spectrum types are stored in self.available.
        """
        if self.log_spec and not name.startswith("log_"):
            name = "log_" + name

        self._set_current(name)

    def _plot_thing(self, thing, label_lines=False, ax_update=None):
        """Plot a specific column in a spectrum file.

        Parameters
        ----------
        thing: str
            The name of the thing to be plotted.
        label_lines: bool
            Plot line IDs.
        ax_update: plt.Axes
            An plt.Axes object to update, i.e. to plot on.
        """
        normalize_figure_style()

        fig, ax = plt.subplots(figsize=(9, 5))  # doing it like this stops static analysis from complaining :)
        if ax_update:
            ax = ax_update

        ax.set_yscale("log")
        ax.set_xscale("log")

        if self.units == SPECTRUM_UNITS_FLM:
            ax.plot(self.spectrum["Lambda"], self.spectrum[thing], label=thing)
            ax.set_xlabel(r"Wavelength [\AA]")
            ax.set_ylabel(r"Flux Density " + f"{self.distance}" + r"pc [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]")
            if label_lines:
                ax = ax_add_line_ids(ax, common_lines(freq=False), logx=True)
        else:
            ax.plot(self.spectrum["Freq."], self.spectrum[thing], label=thing)
            ax.set_xlabel("Frequency [Hz]")
            if self.units == SPECTRUM_UNITS_LNU:
                ax.set_ylabel(r"Luminosity [erg s$^{-1}$ Hz$^{-1}$]")
            else:
                ax.set_ylabel(r"Flux Density " + f"{self.distance}" + r"pc [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
            if label_lines:
                ax = ax_add_line_ids(ax, common_lines(freq=True), logx=True)

        if not ax_update:
            fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
            return fig, ax
        else:
            return ax

    def _plot_observer_spec_file(self, label_lines=False):
        """Plot the spectrum components and observer spectra on a 1x2 panel
        plot. The left panel has the components, whilst the right panel has the
        observer spectrum.

        Parameters
        ----------
        label_lines: bool
            Plot line IDs.
        """
        if "spec" not in self.available and "log_spec" not in self.available:
            raise IOError("A .spec/.log_spec file was not read in, cannot use this function")

        normalize_figure_style()

        fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey="row")

        for component in self.columns[:-self.n_inclinations]:
            if component in ["Lambda", "Freq."]:
                continue
            ax[0] = self._plot_thing(component, label_lines, ax[0])

        for line in ax[0].get_lines():
            line.set_alpha(0.7)
        ax[0].legend(ncol=2, loc="upper right").set_zorder(0)

        for inclination in self.inclinations:
            ax[1] = self._plot_thing(inclination, label_lines, ax[1])

        for label, line in zip(self.inclinations, ax[1].get_lines()):
            line.set_alpha(0.7)
            line.set_label(str(label) + r"$^{\circ}$")
        ax[1].set_ylabel("")
        ax[1].legend(ncol=2, loc="upper right").set_zorder(0)

        ax[0].set_title("Components")
        ax[1].set_title("Observer spectra")

        fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
        fig.subplots_adjust(wspace=0)

        return fig, ax

    def plot(self, name=None, label_lines=False):
        """Plot the spectra or a single component in a single figure. By
        default this creates a 1 x 2 of the components on the left and the
        observer spectra on the right. Useful for when in an interactive
        session.

        Parameters
        ----------
        name: str
            The name of the thing to plot.
        label_lines: bool
            Plot line IDs.
        """
        ot = self.current

        if name:
            if name not in self.columns:
                raise ValueError(f"{name} is not in the current spectrum columns")
            fig, ax = self._plot_thing(name, label_lines)
            if name.isdigit():
                name += r"$^{\circ}$"
            ax.set_title(name.replace("_", r"\_"))
        else:
            self.current = "spec"
            fig, ax = self._plot_observer_spec_file(label_lines)

        self.current = ot

        return fig, ax

    @staticmethod
    def show(block=True):
        """Show any plots which have been generated.

        Parameters
        ----------
        block: bool [optional]
            Enable non-blocking show from matplotlib with False.
        """
        plt.show(block=block)

    def __getitem__(self, key):
        """Return an array in the spectrum dictionary when indexing."""
        if key not in self.available:
            return self.spectrum[key]
        else:
            return self.all_spectrum[key]

    def __setitem__(self, key, value):
        """Allows to modify the arrays in the spectrum dictionary."""
        if key not in self.available:
            self.spectrum[key] = value
        else:
            self.all_spectrum[key] = value

    def __str__(self):
        """Print the basic details about the spectrum."""
        msg = f"Spectrum for the model {self.root} in {self.fp}\n"
        msg += f"Available spectra: {self.available}\n"
        msg += f"Current spectrum {self.current}\n"
        if "spec" in self.available or "log_spec" in self.available:
            msg += f"Spectrum inclinations: {self.all_inclinations['spec']}\n"
        if "tau_spec" in self.available:
            msg += f"Optical depth inclinations {self.all_inclinations['tau_spec']}\n"

        return textwrap.dedent(msg)


# Cell spectrum ----------------------------------------------------------------


class CellSpectra:
    """A class to store the cell spectra accumulated during the ionization
    cycles.

    This class is mostly a storage space, but there is also a rough method for
    plotting. In the future, this will probably be amalgamated into the Wind
    or Spectrum class.
    todo: determine nx, nz to be consistent with actual model
    """
    def __init__(self, root, fp=".", nx=0, nz=0, force_make_spectra=False, delim=None):
        """Initialize the object.
        Reads in the cell spectra into a 1D list. This function will attempt
        to run windsave2table to create the cell spectra files if they do not
        exist.
        Parameters
        ----------
        root: str
            The root name of the Python simulation.
        fp: str
            The directory containing the model.
        force_make_spectra: bool [optional]
            Force windsave2table to be run to re-make the files in the
            tables directory.
        delim: str [optional]
            The delimiter used in the wind table files.
        """
        self.root = root

        self.fp = fp
        if self.fp[-1] != "/":
            self.fp += "/"
        self.pf = self.fp + self.root + ".pf"

        self.nx = int(nx)
        self.nz = int(nz)
        self.nspec = 0
        self.header = []
        self.available = []
        self.spectra = []
        self.original = None

        if force_make_spectra:
            create_wind_save_tables(root, fp, False, True)

        # Try to read in the spectra, if we can't then we'll try and create the
        # spectra and read them in then. If this is unsuccessful, then an
        # IOError will be raised somewhere

        try:
            self.get_cell_spectra(delim)
        except IOError:
            create_wind_save_tables(root, fp, False, True)
            self.get_cell_spectra(delim)

    def get_cell_spectra(self, delim=None):
        """Read in the cell spectra."""

        # In some cases windsave2table will output the cell spectra into
        # multiple files so we read them one by one

        self.header = []
        cell_spectra = []
        frequency_bins = []

        for fp in get_files("*xspec.*.txt"):

            with open(fp, "r") as f:
                spectrum_file = f.readlines()

            spectra = []

            for line in spectrum_file:
                if len(line) == 0 or line.startswith("#"):
                    continue
                if delim:
                    line = line.split(delim)
                else:
                    line = line.split()
                spectra.append(line)

            self.header += spectra[0][1:]
            array = np.array(spectra[1:], dtype=np.float64)
            frequency_bins.append(array[:, 0])
            cell_spectra.append(array[:, 1:])

        # Now we need to join everything together if we have read in multiple
        # files

        if len(cell_spectra) > 1:
            cell_spectra = [np.hstack(cell_spectra)]

        frequency_bins = np.array(frequency_bins[0], dtype=np.float64)  # assuming they're all the same, but we should check
        cell_spectra = cell_spectra[0]

        # We have to make the header differently depending on if the model is
        # 1d or 2d
        # todo: probably make 4 a constant so it is not a magic number

        if len(self.header[0]) > 4:
            self.available = [(int(cell[1:4]), int(cell[5:])) for cell in self.header[1:]]
        else:
            self.available = [(int(cell[1:4]), 0) for cell in self.header[1:]]

        if self.nx == 0 or self.nz == 0:
            try:
                self.nx = int(get_parameter_value(self.pf, "Wind.dim.in.x_or_r.direction"))
                if len(self.header[1]) > 4:
                    self.nz = int(get_parameter_value(self.pf, "Wind.dim.in.z_or_theta.direction"))
            except (ValueError, IOError):
                self.nx = self.available[-1][0] + 1
                self.nz = self.available[-1][1] + 1

        self.nspec = len(self.available)
        self.spectra = np.array([None for _ in range(self.nx * self.nz)], dtype=dict).reshape(self.nx, self.nz)

        for n, (i, j) in enumerate(self.available):
            self.spectra[i, j] = {
                "Freq.": np.copy(frequency_bins),
                "Flux": cell_spectra[:, n]
            }

    def get_elem_number_from_ij(self, i, j):
        """Get the wind element number for a given i and j index."""
        return int(self.nz * i + j)

    def plot(self, i, j, scale="loglog"):
        """Plot the cell spectrum for cell i, j."""
        normalize_figure_style()
        fig, ax = plt.subplots(figsize=(12, 6))

        spectrum = self.spectra[i, j]
        if spectrum is None:
            raise ValueError(f"no cell spectra for cell ({i}, {j}) as cell is probably not in the wind")

        ax.plot(spectrum["Freq."], spectrum["Flux"])
        ax.set_ylabel(r"$J_{\nu}$ [ergs s$^{-1}$ cm$^{-3}$ Sr$^{-1}$ Hz$^{-1}$]")
        ax.set_xlabel("Rest-frame Frequency")

        ax = set_axes_scales(ax, scale)
        fig.suptitle(f"Spectrum in cell ({i}, {j})")

        return fig, ax

    def smooth(self):
        raise NotImplementedError()

    @staticmethod
    def show():
        """Wrapper around matplotlib show."""
        plt.show()

    def __getitem__(self, pos):
        """Index."""
        i, j = pos
        return self.spectra[i, j]

    def __setitem__(self, pos, value):
        """Write to an index."""
        i, j = pos
        self.spectra[i, j] = value

    def __str__(self):
        return print(self.spectra)


# Wind class -------------------------------------------------------------------


class Wind:
    """A class to store 1D and 2D Python wind tables.

    Contains methods to extract variables, as well as convert various
    indices into other indices.
    """
    def __init__(self,
                 root,
                 fp=".",
                 include_cell_spec=True,
                 distance_units="cm",
                 co_mass=None,
                 velocity_units="kms",
                 mask=True,
                 force_make_tables=False,
                 delim=None):
        """Initialize the Wind object.

        Each of the available wind save tables or ion tables are read in, and
        stored in the same dictionary. To access each paramter, using the get()
        method is perferred. However, it is also possible to index using the
        regular [ ] operator. To get an ion, the index is as follows
        w["H"]["density"]["i01"]. If using get(), it is insted as
        get("H_i01d") or get("H_i01f") for the ion fractions.

        Parameters
        ----------
        root: str
            The root name of the Python simulation.
        fp: str
            The directory containing the model.
        include_cell_spec: bool
            Load in the the cell spectra as well.
        distance_units: str
            The distance units of the wind.
        co_mass: float
            The mass of the central object, optional to use in conjunction with
            distance_units == "rg"
        velocity_units: str [optional]
            The velocity units of the wind.
        mask: bool [optional]
            Store the wind parameters as masked arrays.
        force_make_tables: bool [optional]
            Force windsave2table to be run to re-make the files in the
            tables directory.
        delim: str [optional]
            The delimiter used in the wind table files.
        """
        self.root = root
        self.fp = fp
        if self.fp[-1] != "/":
            self.fp += "/"
        self.pf = self.fp + self.root + ".pf"

        self.nx = 1
        self.nz = 1
        self.n_elem = 1
        self.m_coords = ()
        self.n_coords = ()
        self.m_cen_coords = ()
        self.n_cen_coords = ()
        self.axes = ()
        self.parameters = ()
        self.elements = ()
        self.variables = {}
        self.coord_system = WIND_COORD_TYPE_UNKNOWN

        # Set up the distance units and convert to Rg if required

        distance_units = distance_units.lower()
        self.units = ""

        if distance_units not in [WIND_DISTANCE_UNITS_CM, WIND_DISTANCE_UNITS_RG]:
            raise ValueError(f"Unknown distance units {distance_units}: allowed are {WIND_DISTANCE_UNITS_CM} or "
                             f"{WIND_DISTANCE_UNITS_RG}")

        # Set up the velocity units and conversion factors

        velocity_units = velocity_units.lower()

        if velocity_units not in [WIND_VELOCITY_UNITS_CMS, WIND_VELOCITY_UNITS_KMS, WIND_VELOCITY_UNITS_LIGHT]:
            raise ValueError(f"unknown velocity units {velocity_units}. Allowed units [kms, cms, c]")

        self.velocity_units = velocity_units

        if velocity_units == WIND_VELOCITY_UNITS_KMS:
            self.velocity_conversion_factor = CMS_TO_KMS
        elif velocity_units == WIND_VELOCITY_UNITS_CMS:
            self.velocity_conversion_factor = 1
        else:
            self.velocity_conversion_factor = 1 / VLIGHT

        # The next method reads in the wind and initializes the above members.
        # If no wind tables can be found in read_in_wind_parameters, an IOError
        # is raised. If raised, try to create the wind table and read the
        # wind parameters again

        if force_make_tables:
            self.create_wind_tables()

        try:
            self.get_wind_parameters(delim)
            self.get_wind_elements(delim)
            if include_cell_spec:
                self.variables["cell_spec"] = CellSpectra(self.root, self.fp, self.nx, self.nz, force_make_tables)
        except IOError:
            self.create_wind_tables()
            self.get_wind_parameters(delim)
            self.get_wind_elements(delim)
            if include_cell_spec:
                self.variables["cell_spec"] = CellSpectra(self.root, self.fp, self.nx, self.nz)

        self.columns = self.variables.keys()

        # In the case of using rg, the units are set in convert_cm_to_rg, so
        # we set the units here only in the case of using cm

        if distance_units == WIND_DISTANCE_UNITS_RG:
            self.convert_cm_to_rg(co_mass)
        else:
            self.units = distance_units

        # Create masked cells, if that's the users deepest desire for their
        # data

        if mask:
            self.create_masked_arrays()

    def get_wind_parameters(self, delim=None):
        """Read in the wind parameters.

        Parameters
        ----------
        delim: str [optional]
            The deliminator in the wind table files.
        """
        wind_all = []
        wind_columns = []

        # Read in each file, one by one, if they exist. Note that this makes
        # the assumption that all the tables are the same size.

        n_read = 0
        files_to_read = ["master", "heat", "gradient", "converge"]

        for table in files_to_read:
            fp = self.fp + self.root + "." + table + ".txt"
            if not path.exists(fp):
                fp = self.fp + "tables/" + self.root + "." + table + ".txt"
                if not path.exists(fp):
                    continue
            n_read += 1

            with open(fp, "r") as f:
                wind_file = f.readlines()

            # Read in the wind_save table, ignoring empty lines and comments.
            # Each file is stored as a list of lines within a list, so a list
            # of lists.
            # todo: need some method to detect incorrect syntax

            wind_list = []

            for line in wind_file:
                line = line.strip()
                if delim:
                    line = line.split(delim)
                else:
                    line = line.split()
                if len(line) == 0 or line[0] == "#":
                    continue
                wind_list.append(line)

            # Keep track of each file header and add the wind lines for the
            # current file into wind_all, the list of lists, the master list

            if wind_list[0][0].isdigit() is False:
                header = wind_list[0]

                # We need to deal with the case where the mean intensity is
                # name j, the same as the j cell index...

                if table == "heat":
                    header = self._rename_j_to_j_bar(table, header)

                wind_columns += header
            else:
                wind_columns += list(np.arange(len(wind_list[0]), dtype=np.dtype.str))

            wind_all.append(np.array(wind_list[1:], dtype=np.float64))

        if n_read == 0:
            raise IOError(f"Unable to open any wind tables for root {self.root} directory {self.fp}")

        # Determine the number of nx and nz elements. There is a basic check to
        # only check for nz if a j column exists, i.e. if it is a 2d model.

        i_col = wind_columns.index("i")
        self.nx = int(np.max(wind_all[0][:, i_col]) + 1)

        if "z" in wind_columns or "theta" in wind_columns:
            j_col = wind_columns.index("j")
            self.nz = int(np.max(wind_all[0][:, j_col]) + 1)
        self.n_elem = int(self.nx * self.nz)

        wind_all = np.hstack(wind_all)

        # Assign each column header to a key in the dictionary, ignoring any
        # column which is already in the dict and extract the x and z
        # coordinates

        for index, col in enumerate(wind_columns):
            if col in self.variables.keys():
                continue
            self.variables[col] = wind_all[:, index].reshape(self.nx, self.nz)
            self.parameters += col,

        # Get the x/r coordinates

        if "x" in self.parameters:
            self.m_coords = tuple(np.unique(self.variables["x"]))
            self.m_cen_coords = tuple(np.unique(self.variables["xcen"]))
        else:
            self.m_coords = tuple(np.unique(self.variables["r"]))
            self.m_cen_coords = tuple(np.unique(self.variables["rcen"]))

        # Get the z/theta coordinates

        if "z" in self.parameters:
            self.n_coords = tuple(np.unique(self.variables["z"]))
            self.n_cen_coords = tuple(np.unique(self.variables["zcen"]))
        elif "theta" in self.parameters:
            self.n_coords = tuple(np.unique(self.variables["theta"]))
            self.n_cen_coords = tuple(np.unique(self.variables["theta_cen"]))

        # Record the coordinate system and the axes labels

        if self.nz == 1:
            self.coord_system = WIND_COORD_TYPE_SPHERICAL
            self.axes = ["r", "r_cen"]
        elif "r" in self.parameters and "theta" in self.parameters:
            self.coord_system = WIND_COORD_TYPE_POLAR
            self.axes = ["r", "theta", "r_cen", "theta_cen"]
        else:
            self.coord_system = WIND_COORD_TYPE_CYLINDRICAL
            self.axes = ["x", "z", "x_cen", "z_cen"]

        # Convert velocity into desired units and also calculate the cylindrical
        # velocities. This doesn't work for polar or spherical coordinates as
        # they will not have these velocities

        if self.coord_system == WIND_COORD_TYPE_CYLINDRICAL:
            self.project_cartesian_velocity_to_cylindrical()

        # Convert the velocity into the correct units

        self.variables["v_x"] *= self.velocity_conversion_factor
        self.variables["v_y"] *= self.velocity_conversion_factor
        self.variables["v_z"] *= self.velocity_conversion_factor

    def get_wind_elements(self, delim=None, elements_to_get=None):
        """Read in the ion parameters.

        Parameters
        ----------
        delim: str [optional]
            The file delimiter.
        elements_to_get: List[str] or Tuple[str]
            The elements to read ions in for.
        """

        if elements_to_get is None:
            elements_to_get = ("H", "He", "C", "N", "O", "Si", "Fe")
        else:
            if type(elements_to_get) not in [str, list, tuple]:
                raise TypeError("ions_to_get should be a tuple/list of strings or a string")

        # Read in each ion file, one by one. The ions will be stored in the
        # self.variables dict as,
        # key = ion name
        # values = dict of ion keys, i.e. i_01, i_02, etc, and the values
        # in this dict will be the values of that ion

        ion_types_to_get = ["frac", "den"]
        ion_types_index_names = ["fraction", "density"]

        n_elements_read = 0

        for element in elements_to_get:
            element = element.capitalize()

            # Each element will have a dict of two keys, either frac or den.
            # Inside each dict with be more dicts of keys where the values are
            # arrays of the

            self.variables[element] = {}

            for ion_type, ion_type_index_name in zip(ion_types_to_get, ion_types_index_names):
                fp = self.fp + self.root + "." + element + "." + ion_type + ".txt"
                if not path.exists(fp):
                    fp = self.fp + "tables/" + self.root + "." + element + "." + ion_type + ".txt"
                    if not path.exists(fp):
                        continue

                n_elements_read += 1
                if element not in self.elements:
                    self.elements += element,

                with open(fp, "r") as f:
                    ion_file = f.readlines()

                # Read in ion the ion file. this can be done in a list
                # comprehension, I think, but I want to skip commented out lines
                # and I think it's better(?) to do it this way

                wind = []

                for line in ion_file:
                    if delim:
                        line = line.split(delim)
                    else:
                        line = line.split()
                    if len(line) == 0 or line[0] == "#":
                        continue
                    wind.append(line)

                # Now construct the tables, how this is done is described in
                # some of the comments above

                if wind[0][0].isdigit() is False:
                    columns = tuple(wind[0])
                    index = columns.index("i01")
                else:
                    columns = tuple(np.arange(len(wind[0]), dtype=np.dtype.str))
                    index = 0
                columns = columns[index:]
                wind = np.array(wind[1:], dtype=np.float64)[:, index:]

                self.variables[element][ion_type_index_name] = {}
                for index, col in enumerate(columns):
                    self.variables[element][ion_type_index_name][col] = wind[:, index].reshape(self.nx, self.nz)

        if n_elements_read == 0 and len(self.columns) == 0:
            raise IOError("Unable to open any parameter or ion tables: Have you run windsave2table?")

    # def get_cell_spectra(self, delim=None):
    #     """Read in the cell spectra.
    #
    #     Parameters
    #     ----------
    #     delim: str [optional]
    #        The file delimiter for the xspec file.
    #     """
    #     fp = self.fp + self.root + ".xspec.all.txt"
    #     if not path.exists(fp):
    #         fp = self.fp + "tables/" + self.root + ".xspec.all.txt"
    #         if not path.exists(fp):
    #             raise IOError(f"cell spectra file {fp} wasn't found")
    #
    #     self.variables["cell_spec"] = [None for _ in range(self.n_elem)]
    #
    #     with open(fp, "r") as f:
    #         spectrum_file = f.readlines()
    #
    #     spectra = []
    #
    #     for line in spectrum_file:
    #         if len(line) == 0 or line.startswith("#"):
    #             continue
    #         if delim:
    #             line = line.split(delim)
    #         else:
    #             line = line.split()
    #         spectra.append(line)
    #
    #     header = spectra[0]
    #     cell_spectra = np.array(spectra[1:], dtype=np.float64)
    #     frequency_bins = cell_spectra[:, 0]
    #     cell_spectra = cell_spectra[:, 1:]
    #
    #     # We have to make the header differently depending on if the model is
    #     # 1d or 2d
    #     # todo: probably make 4 a constant so it is not a magic number
    #
    #     if len(header[1]) > 4:
    #         cells = [(int(cell[1:4]), int(cell[5:])) for cell in header[1:]]
    #     else:
    #         cells = [(int(cell[1:4]), 0) for cell in header[1:]]
    #
    #     self.variables["cell_spec"] = np.array(self.variables["cell_spec"], dtype=dict)
    #     self.variables["cell_spec"] = self.variables["cell_spec"].reshape(self.nx, self.nz)
    #
    #     for n, (i, j) in enumerate(cells):
    #         self.variables["cell_spec"][i, j] = {
    #             "Freq.": np.copy(frequency_bins),
    #             "Flux": cell_spectra[:, n]
    #         }
    #
    #     self.parameters = tuple(list(self.parameters) + ["cell_spec"])

    def project_cartesian_velocity_to_cylindrical(self):
        """Project the cartesian velocities of the wind into cylindrical
        coordinates."""
        v_l = np.zeros_like(self.variables["v_x"])
        v_rot = np.zeros_like(v_l)
        v_r = np.zeros_like(v_l)
        n1, n2 = v_l.shape

        for i in range(n1):
            for j in range(n2):
                cart_point = np.array([self.variables["x"][i, j], 0, self.variables["z"][i, j]])
                # todo: don't think I need to do this check anymore
                if self.variables["inwind"][i, j] < 0:
                    v_l[i, j] = 0
                    v_rot[i, j] = 0
                    v_r[i, j] = 0
                else:
                    cart_velocity_vector = np.array(
                        [self.variables["v_x"][i, j], self.variables["v_y"][i, j], self.variables["v_z"][i, j]])
                    cyl_velocity_vector = vector.project_cartesian_to_cylindrical_coordinates(
                        cart_point, cart_velocity_vector)
                    if type(cyl_velocity_vector) is int:
                        continue
                    v_l[i, j] = np.sqrt(cyl_velocity_vector[0]**2 + cyl_velocity_vector[2]**2)
                    v_rot[i, j] = cyl_velocity_vector[1]
                    v_r[i, j] = cyl_velocity_vector[0]

        self.variables["v_l"] = v_l * self.velocity_conversion_factor
        self.variables["v_rot"] = v_rot * self.velocity_conversion_factor
        self.variables["v_r"] = v_r * self.velocity_conversion_factor

    def create_masked_arrays(self):
        """Convert each array into a masked array using inwind."""
        to_mask_wind = list(self.parameters)

        # Remove some of the columns, as these shouldn't be masked because
        # weird things will happen when creating a plot. This doesn't need to
        # be done for the wind ions as they don't have the below items in their
        # data structures

        for item_to_remove in [
                "x", "z", "r", "theta", "xcen", "zcen", "rcen", "theta_cen", "i", "j", "inwind", "cell_spec"
        ]:
            try:
                to_mask_wind.remove(item_to_remove)
            except ValueError:
                continue

        # First, create masked arrays for the wind parameters which is simple
        # enough.

        for col in to_mask_wind:
            self.variables[col] = np.ma.masked_where(self.variables["inwind"] < 0, self.variables[col])

        # self.variables["cell_spec"].spectra = np.ma.masked_where(self.variables["inwind"] < 0, self.variables["cell_spec"].spectra)

        # Now, create masked arrays for the wind ions. Have to do it for each
        # element and each ion type and each ion. This is probably slow :)

        for element in self.elements:
            for ion_type in self.variables[element].keys():
                for ion in self.variables[element][ion_type].keys():
                    self.variables[element][ion_type][ion] = np.ma.masked_where(self.variables["inwind"] < 0,
                                                                                self.variables[element][ion_type][ion])

    def create_wind_tables(self):
        """Force the creation of wind save tables for the model.

        This is best used when a simulation has been re-run, as the
        library is unable to detect when the currently available wind
        tables do not reflect a new simulation.
        """

        create_wind_save_tables(self.root, self.fp, ion_density=True)
        create_wind_save_tables(self.root, self.fp, ion_density=False)
        create_wind_save_tables(self.root, self.fp, cell_spec=True)

    def convert_cm_to_rg(self, co_mass_in_msol=None):
        """Convert the spatial units from cm to gravitational radius.

        If the mass of the central source isn't supplied, then the parameter
        file will be searched for the mass.

        Parameters
        ----------
        co_mass_in_msol: float
            The mass of the central object in solar masses.

        Returns
        -------
        rg: float
            The gravitational radius for the model.
        """

        if self.units == WIND_DISTANCE_UNITS_RG:
            return

        if not co_mass_in_msol:
            try:
                co_mass_in_msol = float(get_parameter_value(self.pf, "Central_object.mass(msol)"))
            except Exception as e:
                print(e)
                raise ValueError("unable to find CO mass from parameter file, please supply the mass instead")

        rg = gravitational_radius(co_mass_in_msol)

        if self.coord_system in [WIND_COORD_TYPE_SPHERICAL, WIND_COORD_TYPE_POLAR]:
            self.variables["r"] /= rg
        else:
            self.variables["x"] /= rg
            self.variables["z"] /= rg

        self.units = WIND_DISTANCE_UNITS_RG

        return rg

    def convert_rg_to_cm(self, co_mass_in_msol=None):
        """Convert the spatial units from gravitational radius to cm.

        If the mass of the central source isn't supplied, then the parameter
        file will be searched for the mass.

        Parameters
        ----------
        co_mass_in_msol: float
            The mass of the central object in solar masses.

        Returns
        -------
        rg: float
            The gravitational radius for the model.
        """

        if self.units == WIND_DISTANCE_UNITS_CM:
            return

        if not co_mass_in_msol:
            try:
                co_mass_in_msol = float(get_parameter_value(self.pf, "Central_object.mass(msol)"))
            except Exception as e:
                print(e)
                raise ValueError("unable to find CO mass from parameter file, please supply the mass instead")

        rg = gravitational_radius(co_mass_in_msol)

        if self.coord_system == WIND_COORD_TYPE_SPHERICAL or self.coord_system == WIND_COORD_TYPE_POLAR:
            self.variables["r"] *= rg
        else:
            self.variables["x"] *= rg
            self.variables["z"] *= rg

        self.units = WIND_DISTANCE_UNITS_CM

        return rg

    def _get_element_variable(self, element_name, ion_name):
        """Helper function to get the fraction or density of an ion depending
        on the final character of the requested variable.

        Parameters
        ----------
        element_name: str
            The element symbol, i.e. H, He.
        ion_name: str
            The name of the element, in the format i_01, i_01f, i_01d, etc.
        """

        ion_frac_or_den = ion_name[-1]
        if not ion_frac_or_den.isdigit():
            ion_name = ion_name[:-1]
            if ion_frac_or_den == "d":
                variable = self.variables[element_name]["density"][ion_name]
            elif ion_frac_or_den == "f":
                variable = self.variables[element_name]["fraction"][ion_name]
            else:
                raise ValueError(f"{ion_frac_or_den} is an unknown ion type, try f or d")
        else:
            variable = self.variables[element_name]["density"][ion_name]

        return variable

    def get(self, parameter):
        """Get a parameter array. This is just another way to access the
        dictionary self.variables.

        Parameters
        ----------
        parameter: str
            The name of the parameter to get.
        """
        element_name = parameter[:2].replace("_", "")
        ion_name = parameter[2:].replace("_", "")
        if element_name in self.elements:
            variable = self._get_element_variable(element_name, ion_name)
        else:
            variable = self.variables[parameter]

        return variable

    def get_sight_line_coordinates(self, theta):
        """Get the vertical z coordinates for a given set of x coordinates and
        inclination angle.

        Parameters
        ----------
        theta: float
            The angle of the sight line to extract from. Given in degrees.
        """
        return np.array(self.m_coords, dtype=np.float64) * np.tan(PI / 2 - np.deg2rad(theta))

    def get_variable_along_sight_line(self, theta, parameter):
        """Extract a variable along a given sight line.

        Parameters
        ----------
        theta: float
            The angle to extract along.
        parameter: str
            The parameter to extract.
        """
        if self.coord_system == WIND_COORD_TYPE_POLAR:
            raise NotImplementedError("This hasn't been implemented for polar winds")

        if type(theta) is not float:
            theta = float(theta)

        z_array = np.array(self.n_coords, dtype=np.float64)
        z_coords = self.get_sight_line_coordinates(theta)
        values = np.zeros_like(z_coords, dtype=np.float64)
        w_array = self.get(parameter)

        # This is the actual work which extracts along a sight line

        for x_index, z in enumerate(z_coords):
            z_index = get_array_index(z_array, z)
            values[x_index] = w_array[x_index, z_index]

        return np.array(self.m_coords), z_array, values

    def _get_wind_coordinates(self):
        """Get coordinates of the wind.

        This returns the 2d array of coordinate points for the grid or
        1d depending on the coordinate type. This is different from
        using self.n_coords which returns only the axes points.
        """
        if self.coord_system == WIND_COORD_TYPE_SPHERICAL:
            n_points = self.variables["r"]
            m_points = np.zeros_like(n_points)
        elif self.coord_system == WIND_COORD_TYPE_CYLINDRICAL:
            n_points = self.variables["x"]
            m_points = self.variables["z"]
        else:
            m_points = np.log10(self.variables["r"])
            n_points = np.deg2rad(self.variables["theta"])

        return n_points, m_points

    def _get_wind_indices(self):
        """Get cell indices of the grid cells of the wind.

        This returns the 2d array of grid cell indices for the grid or
        1d depending on the coordinate type.
        """
        if self.coord_system == WIND_COORD_TYPE_SPHERICAL:
            n_points = self.variables["i"]
            m_points = np.zeros_like(n_points)
        elif self.coord_system == WIND_COORD_TYPE_CYLINDRICAL:
            n_points = self.variables["i"]
            m_points = self.variables["i"]
        else:
            raise ValueError("Cannot plot with the cell indices for polar winds")

        return n_points, m_points

    @staticmethod
    def _rename_j_to_j_bar(table, header):
        """Rename j, the mean intensity, to j_bar.

        In old versions of windsave2table, the mean intensity of the field
        was named j which created a conflict for 2D models which have a j
        cell index.

        Parameters
        ----------
        table: str
            The name of the table
        header: list[str]
            Rename the header for j to j_bar.
        """

        count = header.count("j")
        if count < 1:
            return header
        if count > 2:
            raise ValueError(f"too many j's {count} in header for {table} table")

        if "z" in header:
            idx = header.index("j")
            idx = header[idx:].index("j")
            header[idx] = "j_bar"
        else:
            idx = header.index("j")
            header[idx] = "j_bar"

        return header

    def plot(self, variable_name, use_cell_coordinates=True, scale="loglog", log_variable=True):
        """Create a plot of the wind for the given variable.

        Parameters
        ----------
        variable_name: str
            The name of the variable to plot. Ions are accessed as, i.e.,
            H_i01, He_i02, etc.
        use_cell_coordinates: bool [optional]
            Plot using the cell coordinates instead of cell index numbers
        scale: str [optional]
            The type of scaling for the axes
        log_variable: bool [optional]
            Plot the variable in logarithmic units
        """
        variable = self.get(variable_name)

        if use_cell_coordinates:
            n_points, m_points = self._get_wind_coordinates()
        else:
            n_points, m_points = self._get_wind_indices()

        if self.coord_system == "spherical":
            fig, ax = plot_1d_wind(n_points, variable, self.units, scale=scale)
        else:
            if log_variable:
                variable = np.log10(variable)
            fig, ax = plot_2d_wind(n_points, m_points, variable, self.units, self.coord_system, scale=scale)

        if len(ax) == 1:
            ax = ax[0, 0]
            title = f"{variable_name}".replace("_", " ")
            if self.coord_system == "spherical":
                ax.set_ylabel(title)
            else:
                ax.set_title(title)

        return fig, ax

    @staticmethod
    def show(block=True):
        """Show a plot which has been created."""
        plt.show(block=block)

    def get_elem_number_from_ij(self, i, j):
        """Get the wind element number for a given i and j index."""
        return self.nz * i + j

    def get_ij_from_elem_number(self, elem):
        """Get the i and j index for a given wind element number."""
        return np.unravel_index(elem, (self.nx, self.nz))

    def __getitem__(self, key):
        """Return an array in the variables dictionary when indexing."""
        return self.variables[key]

    def __setitem__(self, key, value):
        """Set an array in the variables dictionary."""
        self.variables[key] = value

    def __str__(self):
        """Print basic details about the wind."""
        txt = f"root: {self.root}\nfilepath: {self.fp}\ncoordinate system:{self.coord_system}\n" \
              f"parameters: {self.parameters}\nelements: {self.elements}\n"

        return txt


# Load in all the submodules ---------------------------------------------------

__all__ = []
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    _module = loader.find_module(module_name).load_module(module_name)
    globals()[module_name] = _module

__all__.append("Wind")
__all__.append("Spectrum")
__all__.append("check_python_version")
__all__.append("cleanup_data")
__all__.append("get_files")
__all__.append("get_array_index")
__all__.append("get_root")
__all__.append("smooth_array")
__all__.append("create_wind_save_tables")
__all__.append("run_py_wind")

# These are put here to solve a circular dependency ----------------------------

from pypython.plot import (ax_add_line_ids, common_lines, normalize_figure_style, set_axes_scales)
from pypython.plot.wind import plot_1d_wind, plot_2d_wind

# Raise a system error if the version of Python is too old, or if we can't find
# Python, etc. in the $PATH ----------------------------------------------------

# PY_VERSION = check_python_version()
