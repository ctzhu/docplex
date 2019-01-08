# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

"""
Configuration of the CPO Python API

This module is the top-level handler of the configuration parameters for
the Python API to CPO. It contains the default values of the different
configuration parameters.

It should NOT be changed directly.
The preferable way is to add at least one of the
following files that contain the changes to be performed:

* cpo_config.py, a local set of changes on these parameters,
* cpo_config_<hostname>.py, a hostname dependent set of changes.

Final set of parameters is obtained by loading first this module,
then cpo_config.py and finally cpo_config_<hostname>.py.
These modules should be visible from the PYTHONPATH and are loaded in
this order to overwrite default values.

For compatibility with CPLEX DOcloud configuration, if the DOcloud
key and url are not set in cpo_config_*.py files, the file.docplexrc is loaded
from the user home directory.

If called as main, this module prints the configuration on standard output.
"""

from docplex.cp.utils import Context
from docplex.cp.parameters import CpoParameters

import sys, socket, os


##############################################################################
## Default configuration parameters
##############################################################################

#-----------------------------------------------------------------------------
# Global context

# Create context infrastructure
context = Context(model=Context(),
                  params=CpoParameters(),
                  solver=Context())

# Default log output
context.log_output = sys.stdout

# Visu enable indicator (can be disabled for testing purpose)
context.visu_enabled = True


#-----------------------------------------------------------------------------
# Modeling context

# Indicate to add source location in model
context.model.add_source_location = True

# Minimal variable name length that trigger use of shorter alias. None for no alias.
context.model.length_for_alias = None

# Name of the directory where store copy of the generated CPO files. None for no dump.
context.model.dump_directory = None


#-----------------------------------------------------------------------------
# Solving parameters

# Default time mode
context.params.TimeMode = "ElapsedTime"

# Default time mode
context.params.TimeLimit = 100

# Workers count
context.params.Workers = 4


#-----------------------------------------------------------------------------
# Solving context

# Indicate to trace CPO model before solving
context.solver.trace_cpo = False

# Indicate to trace solver log after solving
context.solver.trace_log = False

# Indicate to add solver log to the solution
context.solver.add_log_to_solution = True

# Log prefix
context.solver.log_prefix = "[Solver] "

# Name of the agent to be used for solving. Value is name of one of this context child context (i.e. 'docloud').
context.solver.agent = 'docloud'


#-----------------------------------------------------------------------------
# DoCloud solving agent context

context.solver.docloud = Context()

# Agent class name
context.solver.docloud.class_name = "docplex.cp.solver_docloud.CpoSolverDocloud"

# Url of the DOCloud service
context.solver.docloud.url = "https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/"

# Authentication key.
context.solver.docloud.key = "'Set your key in docloud_config.py''"

# Secret key.
context.solver.docloud.secret = None

#  Indicate to verify SSL certificates
context.solver.docloud.verify_ssl = True

# Default unitary request timeout in seconds
context.solver.docloud.request_timeout = 30

# Time added to expected solve time to compute the total result waiting timeout
context.solver.docloud.result_wait_extra_time = 60

# Clean job after solve indicator
context.solver.docloud.clean_job_after_solve = True

# Add 'Connection close' in all headers
context.solver.docloud.always_close_connection = False

# Log prefix
context.solver.docloud.log_prefix = "[DOcloud] "


##############################################################################
## Overloading with other configuraton python files
##############################################################################

def _eval_file(file):
    """ If exists, evaluate the content of a python module in this module.
    Args:
        file: Python file to evaluate
    """
    for f in filter(os.path.isfile, [dir + "/" + file for dir in sys.path]):
         exec(open(f).read())


def _load_properties(ppf):
    """ Load property file (if exists)
    Args:
        ppf: Property file to load, if exists
    """
    if os.path.isfile(ppf):
        with open(ppf, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                snx = line.find('=')
                if snx < 0:
                    snx = line.find(':')
                if snx > 0:
                    key = line[:snx].strip()
                    val = line[snx + 1:].strip()
                    if key in ("api_key", "docloud.api_key", "docloud.key") :
                        context.solver.docloud.key = val
                    elif key in ("url", "docloud.url"):
                        context.solver.docloud.url = val


# Initialize default list of files to load
FILE_LIST = (os.path.expanduser("~") + os.path.sep + ".docplexrc",
             "cpo_config_local.py",  # For upward compatibility
             "cpo_config.py",
             "cpo_config_" + socket.gethostname() + ".py",
             "docloud_config.py")

# Load all config changes
for f in FILE_LIST:
    if f.endswith(".py"):
        _eval_file(f)
    else:
        _load_properties(f)


##############################################################################
## Print configuration when called as main
##############################################################################

if __name__ == "__main__":
    context.print_context()
