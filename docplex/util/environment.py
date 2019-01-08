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
 * on DOcplexcloud, the python program running inside the Python Worker,

As far as possible, the adaptation to the execution environment is done
automatically. Functions that are presented here are useful to handle
very specific use cases.

The following code is a program that sums its input (``sum.py``)::

    import json
    from docplex.util.environment import get_environment
    if __name__ == "__main__":
        sum = 0
        # open program input named "data.txt" and sum the contents
        with get_environment().get_input_stream("data.txt") as input:
            for i in input.read().split():
                sum += int(i)
        # write the result as a simple json in program output "solution.json"
        with get_environment().get_output_stream("solution.json") as output:
            output.write(json.dumps({'result': sum}))

Let's put some data in a ``data.txt`` file::

    4 7 8
    19

When you run ``sum.py`` with your python interpreter, it opens the ``data.txt`` file and sums all integers
in that file. The result is saved as a json fragment in file ``solution.json``::

    $ python sum.py
    $ more solution.json
    {"result": 38}

To submit that program to DOcplexcloud service, we write a ``submit.py`` program that uses
the `DOcplexcloud Python API <https://developer.ibm.com/docloud/documentation/docloud/python-api/>`_
to create and submit a job. That jobs has two attachments:

- ``sum.py``, the program to execute
- ``data.txt``, the data expected by ``sum.py``

After the solve is completed, the result of the program is downloaded and saved as ``solution.json``::

    from docloud.job import JobClient
    if __name__ == "__main__":
        url = "ENTER_YOUR_URL_HERE"
        key = "ENTER_YOUR_KEY_HERE"
        client = JobClient(url, key)
        client.execute(input=["sum.py", "data.txt"], output="solution.json")

Then you run ``submit.py``::

    $ python submit.py
    $ more solution.json
    {"result": 38}
"""
import tempfile


class Environment(object):
    """ Methods allowing to interact with execution environment.

    The ``docplex`` package internally provides the appropriate implementation
    according to the actual execution environment.
    The correct instance of this class is returned by the method :meth:`docplex.util.environment.get_environment`
    provided in this module
    """

    def get_input_stream(self, name):
        """ Get an input of the program as a stream (file like object).

        An input of the program is a file available in the working directory.

        When run on DOcplexcloud, all input attachments are copied in the working directory before
        the program is run. ``get_input_stream`` allows you to open input attachments of the job.

        Args:
            name: Name of the input object
        Returns:
            A file object to read the input from.
        """
        return None

    def get_output_stream(self, name):
        """ Get a file-like object to write an output of the program.

        The file is recorded as being part of the program output.
        This method can be called multiple times if the program contains multiple output objects.

        When run on premise, output of the program are written as files in the working directory.
        When run on DOcplexcloud, the files are attached as output attachment.

        Args:
            name: Name of the output object
        Returns:
            A file object to write the output to.
        """
        return None

    def get_available_core_count(self):
        """ Returns the number of cores available for the processing.

        This number is used in the solving engine as the number of threads.

        Returns:
            The available number of cores.
        """
        return 0

    def get_parameter(self, name):
        """ Returns a parameter of the program.

        On DOcplexcloud, this method returns return the job parameter which name is specified.

        Args:
            name: The name of the parameter
        Returns:
            The parameter which name is specified
        """
        return None

    def update_solve_details(self, details):
        #===============================================================================
        #         """Update the solve details.
        # 
        #         You use this method to send solve details to the DOcplexcloud service.
        # 
        #         If ``context.solver.auto_publish.solve_details`` is set, the underlying solver will automatically
        #         send details. If you want to craft and send your own solve details, you can use the following
        #         keys (non exaustive list):
        # 
        #             - MODEL_DETAIL_TYPE : Model type
        #             - MODEL_DETAIL_CONTINUOUS_VARS : Number of continuous variables
        #             - MODEL_DETAIL_INTEGER_VARS : Number of integer variables
        #             - MODEL_DETAIL_BOOLEAN_VARS : Number of boolean variables
        #             - MODEL_DETAIL_INTERVAL_VARS : Number of interval variables
        #             - MODEL_DETAIL_SEQUENCE_VARS : Number of sequence variables
        #             - MODEL_DETAIL_NON_ZEROS : Number of non zero variables
        #             - MODEL_DETAIL_CONSTRAINTS : Number of constraints
        #             - MODEL_DETAIL_LINEAR_CONSTRAINTS : Number of linear constraints
        #             - MODEL_DETAIL_QUADRATIC_CONSTRAINTS : Number of quadratic constraints
        # 
        #         Args:
        #             details: A ``dict`` with solve details as key/value pairs.
        # 
        #         See:
        #             :attr:`.Context.solver.auto_publish.solve_details`
        #         """
        #===============================================================================
        pass


class LocalEnvironment(Environment):
    # The environment solving environment using all local input and outputs.
    def __init__(self):
        super(LocalEnvironment, self).__init__()

    def get_available_core_count(self):
        return None  # none !

    def get_input_stream(self, name):
        return open(name, "rb")

    def get_output_stream(self, name):
        return open(name, "wb")

    def get_parameter(self, name):
        return None

    def update_solve_details(self, details):
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
            self.solve_hook.set_output_attachments({self.attachment_name: self.file.name})
            self.closed = True


class WorkerEnvironment(Environment):
    # The solving environment when we run in the DOcplexCloud worker.
    def __init__(self, solve_hook):
        super(WorkerEnvironment, self).__init__()
        self.solve_hook = solve_hook

    def get_available_core_count(self):
        return self.solve_hook.get_available_core_count()

    def get_input_stream(self, name):
        # inputs are in the current working directory
        return open(name, "rb")

    def get_output_stream(self, name):
        # open the output in a place we know we can write
        f = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
        return OutputFileWrapper(f, self.solve_hook, name)

    def get_parameter(self, name):
        return self.solve_hook.get_parameter_value(name)

    def update_solve_details(self, details):
        self.solve_hook.update_solve_details(details)


def get_environment():
    """ Returns the Environment object that represents the actual execution environment.

    Returns:
        An instance of the :class:`.Environment` class that implements methods corresponding
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
