# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

"""
IBM Decision Optimization CPLEX Modeling for Python - Constraint Programming

This package contains a Python API allowing to build Constraint Programming
models and their solving using the Decision Optimization cloud services.
"""

import platform
import sys
import docplex.version as dcpv

ERROR_STRING = "docplex is not compatible with this version of Python: only 64 bits on Windows, Linux and Darwin, with python 2.7.9+ and 3.4.x are supported."

# Check platform system
if platform.system() not in ('Darwin', 'Linux', 'Windows', 'Microsoft'):
    raise Exception("DOcplex.CP is not supported on this version of your system. Supported versions are Windows, Linux and Darwin.")

# Check version of Python
pv = sys.version_info
if (pv < (2, 7)) or ((pv[0] == 3) and ((pv < (3, 4) or pv >= (3, 6)))):
    raise Exception("DOcplex.CP is supported by Python versions 2.7.9+, 3.4.x and 3.5.x")

# Set version information
__version_info__ = (dcpv.docplex_version_major, dcpv.docplex_version_minor, dcpv.docplex_version_micro)
