"""Utility functions for Input/Output."""

import os


class NoDataRootError(Exception):
    """Exception to be thrown when data root doesn't exist."""
    pass


def get_data_root():
    """Returns the data root, which we assume is contained in an environment variable.

    Returns:
        string, the data root.

    Raises:
        NoDataRootError: If environment variable doesn't exist.
    """
    data_root_var = 'DATAROOT'
    try:
        return os.environ[data_root_var]
    except KeyError:
        raise NoDataRootError('Data root must be in environment variable {}, which'
                              ' doesn\'t exist.'.format(data_root_var))
