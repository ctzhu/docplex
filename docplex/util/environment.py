# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2016
# --------------------------------------------------------------------------

"""
Representation of the DOcplex solving environment.

This module handles the various elements that allow an
optimization program to run independently from the solving environment.
This environment may be:

 * on premise, using a local version of CPLEX Optimization Studio to solve MP problems, or
 * on DOcplexcloud, with the Python program running inside the Python Worker.

As much as possible, the adaptation to the solving environment is
automatic. The functions that are presented here are useful for handling
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

When you run ``sum.py`` with a Python interpreter, it opens the ``data.txt`` file and sums all of the integers
in that file. The result is saved as a JSON fragment in file ``solution.json``::

    $ python sum.py
    $ more solution.json
    {"result": 38}

To submit the program to the DOcplexcloud service, we write a ``submit.py`` program that uses
the `DOcplexcloud Python API <https://developer.ibm.com/docloud/documentation/docloud/python-api/>`_
to create and submit a job. That job has two attachments:

- ``sum.py``, the program to execute and
- ``data.txt``, the data expected by ``sum.py``.

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
    """ Methods for interacting with the execution environment.

    Internally, the ``docplex`` package provides the appropriate implementation
    according to the actual execution environment.
    The correct instance of this class is returned by the method :meth:`docplex.util.environment.get_environment`
    that is provided in this module.
    """

    def get_input_stream(self, name):
        """ Get an input of the program as a stream (file-like object).

        An input of the program is a file that is available in the working directory.

        When run on DOcplexcloud, all input attachments are copied to the working directory before
        the program is run. ``get_input_stream`` lets you open the input attachments of the job.

        Args:
            name: Name of the input object.
        Returns:
            A file object to read the input from.
        """
        return None

    def get_output_stream(self, name):
        """ Get a file-like object to write the output of the program.

        The file is recorded as being part of the program output.
        This method can be called multiple times if the program contains multiple output objects.

        When run on premise, the output of the program is written as files in the working directory.
        When run on DOcplexcloud, the files are attached as output attachments.

        Args:
            name: Name of the output object.
        Returns:
            A file object to write the output to.
        """
        return None

    def get_available_core_count(self):
        """ Returns the number of cores available for processing if the environment
        sets a limit.

        This number is used in the solving engine as the number of threads.

        Returns:
            The available number of cores or ``None`` if the environment does not
            limit the number of cores.
        """
        return None

    def get_parameter(self, name):
        """ Returns a parameter of the program.

        On DOcplexcloud, this method returns the job parameter whose name is specified.

        Args:
            name: The name of the parameter.
        Returns:
            The parameter whose name is specified.
        """
        return None


    def notify_start_solve(self, solve_details):
        #===============================================================================
        #         """Notify the solving environment that a solve is starting.
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
        #             solve_details: A ``dict`` with solve details as key/value pairs
        #         See:
        #             :attr:`.Context.solver.auto_publish.solve_details`
        #         """
        #===============================================================================
        pass

    def update_solve_details(self, details):
        #===============================================================================
        #         """Update the solve details.
        # 
        #         You use this method to send solve details to the DOcplexcloud service.
        #         If ``context.solver.auto_publish.solve_details`` is set, the underlying solver will automatically
        #         update solve details.
        # 
        #         Args:
        #             details: A ``dict`` with solve details as key/value pairs.
        # 
        #         See:
        #             :attr:`.Context.solver.auto_publish.solve_details`
        #         """
        #===============================================================================
        pass

    def notify_end_solve(self, status):
        #===============================================================================
        #         """Notify the solving environment that the solve as ended.
        # 
        #         The ``status`` can be a docloud.status.JobSolveStatus enum or an integer.
        # 
        #         When ``status`` is an integer, it is converted with the following conversion table:
        # 
        #             0 - UNKNOWN: The algorithm has no information about the solution.
        #             1 - FEASIBLE_SOLUTION: The algorithm found a feasible solution.
        #             2 - OPTIMAL_SOLUTION: The algorithm found an optimal solution.
        #             3 - INFEASIBLE_SOLUTION: The algorithm proved that the model is infeasible.
        #             4 - UNBOUNDED_SOLUTION: The algorithm proved the model unbounded.
        #             5 - INFEASIBLE_OR_UNBOUNDED_SOLUTION: The model is infeasible or unbounded.
        # 
        #         Args:
        #             status: The solve status
        #         """
        #===============================================================================
        pass


class LocalEnvironment(Environment):
    # The environment solving environment using all local input and outputs.
    def __init__(self):
        super(LocalEnvironment, self).__init__()

    def get_input_stream(self, name):
        return open(name, "rb")

    def get_output_stream(self, name):
        return open(name, "wb")


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

    def notify_start_solve(self, solve_details):
        self.solve_hook.notify_start_solve(None, # model
                                           solve_details)

    def notify_end_solve(self, status):
        try:
            from docloud.status import JobSolveStatus
            engine_status = JobSolveStatus(status)
            self.solve_hook.notify_end_solve(None, #model, unused
                                             None, # has_solution, unused
                                             engine_status,
                                             None, # reported_obj, unused
                                             None, # var_value_dict, unused
                                             )
        except ImportError:
            raise RuntimeError("This should have been called only when in a worker environment")


def get_environment():
    """ Returns the Environment object that represents the actual execution environment.

    Returns:
        An instance of the :class:`.Environment` class that implements methods corresponding
        to actual execution environment.
    """
    try:
        import docplex.worker.solvehook as worker_env
        hook = worker_env.get_solve_hook()
        if hook:
            return WorkerEnvironment(hook)
    except ImportError:
        pass
    return LocalEnvironment()
