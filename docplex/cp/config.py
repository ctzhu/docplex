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

If called as main, this module prints the configuration on standard output.
"""

from docplex.cp.utils import Context, CpoException
from docplex.cp.parameters import CpoParameters, ALL_PARAMETER_NAMES

import sys, socket, os


##############################################################################
## Default configuration parameters
##############################################################################

#-----------------------------------------------------------------------------
# Global context

# Create default context infrastructure
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
context.model.length_for_alias = 15

# Automatically add a name to every top-level constraint
context.model.name_all_constraints = True

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

# Indicate to trace solver log on log_output.
context.solver.trace_log = False

# Max number of threads allowed for model solving
context.solver.max_threads = None
try:
    import docplex.util.environment as runenv
    context.solver.max_threads = runenv.get_environment().get_available_core_count()
except:
    pass

# Indicate to add solver log to the solution
context.solver.add_log_to_solution = True

# Indicate to auto-publish solve details and results in environment
context.solver.auto_publish = True

# Log prefix
context.solver.log_prefix = "[Solver] "

# Name of the agent to be used for solving. Value is name of one of this context child context (i.e. 'docloud').
context.solver.agent = 'docloud'


#-----------------------------------------------------------------------------
# DoCloud solving agent context

context.solver.docloud = Context()

# Agent class name
context.solver.docloud.class_name = "docplex.cp.solver.solver_docloud.CpoSolverDocloud"

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

# Polling delay (min, max and increment)
context.solver.docloud.polling = Context(min=1, max=3, incr=0.2)


##############################################################################
## Public functions
##############################################################################

def get_default():
    """ Get the default context

    Returns:
        Current default context
    """
    return context

def set_default(ctx):
    """ Set the default context

    Args:
        ctx: New default context
    Returns:
        Current default context
    """
    if ctx is None:
        ctx = Context()
    else:
        assert isinstance(ctx, Context), "Context object must be of class Context"
    sys.modules[__name__].context = ctx


# Attribute values denoting a default value
DEFAULT_VALUES = ("ENTER YOUR KEY HERE", "ENTER YOUR URL HERE", "default")


def _get_effective_context(**kwargs):
    """ Build a effective context from a variable list of arguments that may specify changes to default.

    Args:
        context (optional):   Source context, if not default.
        params (optional):    Solving parameters (CpoParameters) that overwrite those in the solving context
        (others) (optional):  All other context parameters that can be changed.
    Returns:
        Updated (cloned) context
    """
    # Determine source context
    ctx = kwargs.get('context', None)
    if (ctx is None) or (ctx in DEFAULT_VALUES):
        ctx = context
    # print("*** Source context");
    # ctx.print_context()

    # Process changes
    ctx = ctx.clone()
    rplist = []  # List of replacements to be done in solving parameters
    for k, v in kwargs.items():
        if (k != 'context') and (v not in DEFAULT_VALUES):
            rp = ctx.search_and_replace_attribute(k, v)
            # If not found, set in solving parameters
            if (rp is None):
                rplist.append((k, v))

    # Replace or set remaining fields in parameters
    if rplist:
        params = ctx.params
        #if params is None:
        #    params = CpoParameters()
        #    ctx.params = params;
        if isinstance(params, CpoParameters):
            for k, v in rplist:
                if not k in ALL_PARAMETER_NAMES:
                    raise CpoException("CPO solver does not accept a parameter named '{}'".format(k))
                params.set_attribute(k, v)

    # Return
    # print("*** Result context");
    # ctx.print_context()
    return ctx


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


# Initialize default list of files to load
FILE_LIST = ("cpo_config.py",
             "cpo_config_" + socket.gethostname() + ".py",
             "docloud_config.py")

# Load all config changes
for f in FILE_LIST:
    _eval_file(f)


##############################################################################
## Print configuration when called as main
##############################################################################

if __name__ == "__main__":
    context.print_context()
