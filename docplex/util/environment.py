# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2016
# --------------------------------------------------------------------------

"""
Representation of the DOcplex solving environment.

This module handles the different elements that allow to execute the same
optimization program independently from the solving environment.
This environment may be:

 * on premise, using a local version of CPLEX Optimization Studio,
 * on DOCloud, the python program running inside the Python Worker,

As far as possible, the adaptation to the execution environment is done
automatically. Functions that are presented here are useful to handle
very specific use cases.
"""

import multiprocessing
import tempfile
import types

class Environment(object):
    """ Methods allowing to interact with execution environment.

    The docplex package internally provides the appropriate implementation
    according to the actual execution environment.
    The correct instance of this class is returned by the method get_environment()
    provided in this module
    """

    def get_input(self, name):
        """ Get an input of the program.

        Args:
            name: Name of the input object
        Returns:
            A file object to read the input from.
        """
        return None

    def get_output(self, name):
        """ Create a file that will contain a result of the program.

        The file is recorded as being part of the program output.
        This method can be called multiple times if the program contains multiple output objects.

        Args:
            name: Name of the output object
        Returns:
            A file object to write the output to.
        """
        return None

    def get_available_core_count(self):
        """ Returns the number of cores available for the processing.

        Returns:
            The available number of cores.
        """
        return 0
    
    def get_parameter(self, name):
        """ Returns a parameter of the program.
        
        On DOcloud, this method returns return the job parameter which name is specified
        
        Args:
            name: The name of the parameter
        Returns:
            The parameter which name is specified
        """
        return None
    
    def publish_solve_details(self, details):
        """ Publish the solve details.
        
        Args:
            details: A ``dict`` with solve details as key/value pairs.
        """


class LocalEnvironment(Environment):
    """ The environment solving environment using all local input and outputs.
    """
    def __init__(self):
        super(LocalEnvironment, self).__init__()
        
    def get_available_core_count(self):
        return multiprocessing.cpu_count()

    def get_input(self, name):
        return open(name, "rb")
    
    def get_output(self, name):
        return open(name, "wb")
    
    def get_parameter(self, name):
        return None

    def publish_solve_details(self, details):
        pass


class OutputFileWrapper(object):
    # Wraps a file object so that on __exit__() and on close(), the wrapped file is closed and
    # the output attachments are actually set in the worker
    def __init__(self, file, solve_hook, attachment_name):
        self.file = file
        self.solve_hook = solve_hook
        self.attachment_name = attachment_name
        self.closed = False
    
    def __getattr__(self, name):
        if name == 'close':
            return self.my_close
        else:
            return getattr(self.file, name)
    
    def __enter__(self, *args, **kwargs):
        return self.file.__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        self.file.__exit__(*args, **kwargs) 
        self.close()
        
    def close(self):
        # actually close the output then set attachment
        if not self.closed:
            self.file.close()
            self.solve_hook.set_output_attachments({ self.attachment_name : self.file.name })
            self.closed = True

class WorkerEnvironment(Environment):
    """ The solving environment when we run in the DOcplexCloud worker.
    """
    def __init__(self, solve_hook):
        super(WorkerEnvironment, self).__init__()
        self.solve_hook = solve_hook
        
    def get_available_core_count(self):
        return self.solve_hook.get_available_core_count()
    
    def get_input(self, name):
        # inputs are in the current working directory
        return open(name, "rb")
    
    def get_output(self, name):
        # open the output in a place we know we can write
        f = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
        return OutputFileWrapper(f, self.solve_hook, name)
    
    def get_parameter(self, name):
        return self.solve_hook.get_parameter_value(name)
    
    def publish_solve_details(self, details):
        self.solve_hook.update_solve_details(details)

def get_environment():
    """ Returns the Environment object that represents the actual execution environment.

    Returns:
        An instance of the Environment class that implements methods corresponding
        to actuel execution environment
    """
    try:
        import docplex.worker.solvehook as worker_env
        hook = worker_env.get_solve_hook()
        if hook:
            return WorkerEnvironment(hook)
    except ImportError:
        pass
    return LocalEnvironment()
