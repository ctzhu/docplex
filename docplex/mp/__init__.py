# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore
"""
This is the MP package.
"""

import platform

from sys import version_info

ERROR_STRING = "docplex is not compatible with this version of Python: only 64 bits on Windows, Linux and Darwin, with python 2.7.9+ and 3.4.x are supported."

platform_system = platform.system()
if platform_system in ('Darwin', 'Linux', 'Windows', 'Microsoft'):
    if version_info[0] == 3:
        if version_info < (3, 4, 0):
            raise Exception(ERROR_STRING)
        elif version_info >= (3, 5, 0):
            raise Exception(ERROR_STRING)
    elif version_info[0] == 2:
        if version_info[1] != 7:
            raise Exception(ERROR_STRING)
    else:
        raise Exception(ERROR_STRING)
else:
    raise Exception(ERROR_STRING)

from docplex.version import docplex_version_major, docplex_version_minor, docplex_version_micro
__version_info__ = (docplex_version_major, docplex_version_minor, docplex_version_micro)
