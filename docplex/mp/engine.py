# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


from docplex.mp.solution import SolveSolution, SolveDetails
from docplex.mp.utils import DOcplexException
from docloud.status import JobSolveStatus
# gendoc: ignore


class ISolver(object):
    """
    The pure solving part
    """
    def can_solve(self):
        """
        :return: True if this engine class can truly solve
        """
        raise NotImplementedError  # pragma: no cover

    def connect_progress_listeners(self, listeners):
        """
        Connects progress listeners
        :param listeners:
        :return:
        """
        raise NotImplementedError  # pragma: no cover

    def solve(self, mdl, parameters):
        ''' Redefine this method for the real solve.
            Returns True if model is well-formed and a solution has been found.
        '''
        raise NotImplementedError  # pragma: no cover

    def solve_relaxed(self, mdl, relaxable_groups, optimize, limits, parameters=None):
        """
        Runs feasopt-like algorithm with a set of relaxable cts with preferences
        :param relaxable_groups:
        :param optimize:
        :return:
        """
        raise NotImplementedError  # pragma: no cover

    def get_solve_status(self):
        """  Return a DOcloud-style solve status.

        Possible enums are in docloud/status.py
        Default is UNKNOWN at this stage. Redefined for CPLEX and DOcloud engines.
        """
        return JobSolveStatus.UNKNOWN  # pragma: no cover

    def get_solutions(self, dvars):
        """
        Returns a dictionary of variable solution values.
        :param dvars:
        :return:
        """
        raise NotImplementedError  # pragma: no cover

    def get_cplex(self):
        """
        Returns the underlying CPLEX, if any. May raise an exception if not applicable.
        :return:
        """
        raise NotImplementedError  # pragma: no cover

    def has_cplex(self):  # pragma: no cover
        try:
            return self.get_cplex() is not None
        except DOcplexException:
            # some engine may raise an exception when accessing a cplex
            return False

    def set_parameter(self, parameter, value):
        """ Changes the parameter value.
        :param parameter:
        :param value:
        """
        raise NotImplementedError  # pragma: no cover

    def get_parameter(self, parameter):
        raise NotImplementedError  # pragma: no cover

    def get_solve_details(self):
        raise NotImplementedError  # pragma: no cover


# noinspection PyAbstractClass
class IEngine(ISolver):
    """ interface for all engine facades
    """

    def name(self):
        ''' Returns the code to be used in model'''
        raise NotImplementedError  # pragma: no cover

    def get_var_index(self, var):
        raise NotImplementedError  # pragma: no cover

    def get_ct_index(self, index):
        raise NotImplementedError  # pragma: no cover

    def get_infinity(self):
        raise NotImplementedError  # pragma: no cover

    def create_one_variable(self, vartype, lb, ub, name):
        raise NotImplementedError  # pragma: no cover

    def create_variables(self, keys, vartype, lb, ub, namer):
        raise NotImplementedError  # pragma: no cover

    def create_binary_linear_constraint(self, binaryct):
        raise NotImplementedError  # pragma: no cover

    def create_block_linear_constraints(self, ct_seq):
        raise NotImplementedError  # pragma: no cover

    def create_range_constraint(self, rangect):
        raise NotImplementedError  # pragma: no cover

    def create_indicator_constraint(self, ind):
        raise NotImplementedError  # pragma: no cover

    def remove_constraint(self, ct):
        raise NotImplementedError  # pragma: no cover

    def set_objective(self, sense, expr):
        raise NotImplementedError  # pragma: no cover

    def clear_objective(self, expr):
        raise NotImplementedError  # pragma: no cover

    def end(self):
        raise NotImplementedError  # pragma: no cover

    def notify_trace_output(self, out):
        raise NotImplementedError  # pragma: no cover

    def set_var_attribute(self, var, attr_name, attr_val):
        raise NotImplementedError  # pragma: no cover

    def get_var_attribute(self, var, attr_name):
        raise NotImplementedError  # pragma: no cover


# noinspection PyAbstractClass
class DummyEngine(IEngine):

    def create_range_constraint(self, rangect):
        return -1  # pragma: no cover

    def create_indicator_constraint(self, ind):
        return -1  # pragma: no cover

    def notify_trace_output(self, out):
        pass  # pragma: no cover

    def get_infinity(self):
        return 1e+20  # pragma: no cover

    def get_var_index(self, var):
        return -1  # pragma: no cover

    def get_ct_index(self, index):
        return -1  # pragma: no cover

    def create_one_variable(self, vartype, lb, ub, name):
        return -1  # pragma: no cover

    def create_variables(self, keys, vartype, lb, ub, namer):
        return [-1] * len(keys)  # pragma: no cover

    def set_var_attribute(self, dvar, attr_name, attr_val):
        return attr_val  # pragma: no cover

    def get_var_attribute(self, var, attr_name):  # pragma: no cover
        if "name" == attr_name:
            return var.name
        elif "lb" == attr_name:
            return var.lb
        elif "ub" == attr_name:
            return var.ub
        else:
            raise NotImplementedError

    def create_binary_linear_constraint(self, binaryct):
        return -1  # pragma: no cover

    def create_block_linear_constraints(self, ct_seq):
        return [-1] * len(ct_seq)  # pragma: no cover

    def remove_constraint(self, ct):
        pass  # pragma: no cover

    def set_objective(self, sense, expr):
        pass  # pragma: no cover

    def end(self):
        """ terminate the engine
        """
        pass  # pragma: no cover

    def connect_progress_listeners(self, listeners):
        pass  # pragma: no cover

    def can_solve(self):
        return False  # pragma: no cover

    def solve(self, mdl, parameters):
        return False  # pragma: no cover

    def get_solve_status(self):
        return JobSolveStatus.UNKNOWN  # pragma: no cover

    def solve_relaxed(self, mdl, relaxable_groups, optimize, limits, parameters=None):
        raise NotImplementedError  # pragma: no cover

    def get_solve_attribute(self, attr_name, indices):
        return {}  # pragma: no cover

    def get_solutions(self, *args):
        return {}  # pragma: no cover

    def clear_objective(self, expr):
        pass  # pragma: no cover

    def get_cplex(self):
        raise DOcplexException("No CPLEX is available.")  # pragma: no cover

    def get_solve_details(self):
        return SolveDetails.make_dummy()


# noinspection PyAbstractClass
class IndexerEngine(DummyEngine):
    """
    An abstract engine facade which generates unique indices for variables, constraints
    """

    def __init__(self, initial_index=0):
        DummyEngine.__init__(self)
        self._initial_index = initial_index  # CPLEX indices start at 0, not 1
        self.__var_counter = self._initial_index
        self.__ct_counter = self._initial_index

    def _incrementVars(self, size=1):
        self.__var_counter += size
        return self.__var_counter

    def _incrementCts(self, size=1):
        self.__ct_counter += size
        return self.__ct_counter

    def create_one_variable(self, vartype, lb, ub, name):
        old_count = self.__var_counter
        self._incrementVars(1)
        return old_count

    def create_variables(self, keys, vartype, lb, ub, namer):
        old_count = self.__var_counter
        new_count = self._incrementVars(len(keys))
        return list(range(old_count, new_count))

    def _create_one_ct(self):
        old_ct_count = self.__ct_counter
        self._incrementCts(1)
        return old_ct_count

    def create_binary_linear_constraint(self, binaryct):
        return self._create_one_ct()

    def create_block_linear_constraints(self, ct_seq):
        old_ct_count = self.__ct_counter
        self._incrementCts(len(ct_seq))
        return range(old_ct_count, self.__ct_counter)

    def create_range_constraint(self, rangect):
        return self._create_one_ct()

    def create_indicator_constraint(self, ind):
        return self._create_one_ct()

    def get_solve_attributes(self, attr_name, indices):
        # return empty dict
        return {}

    def dump(self, path):
        pass

    def set_objective(self, sense, expr):
        pass

    def clear_objective(self, expr):
        pass

    def set_parameter(self, parameter, value):
        """ Changes the parameter value in the engine.

        For this limited type of engine, nothing to do.

        """
        pass

    def get_parameter(self, parameter):
        """ Gets the current value of a parameter.

        Params:
         parameter: the parameter for which we query the value.

        """
        return parameter.get()


class NoSolveEngine(IndexerEngine):
    # INTERNAL: a dummy engine that cannot solve.

    def __init__(self, mdl, **kwargs):
        IndexerEngine.__init__(self)

    def name(self):
        return "local"

    def get_var_index(self, var):
        return var.index

    def get_ct_index(self, ct):
        return ct.index

    def can_solve(self):
        return False

    def solve(self, mdl, parameters):
        """
        This solver cannot solve. never ever.
        """
        mdl.fatal("No CPLEX DLL and no DOcloud credentials: model cannot be solved!")
        return None

    def solve_relaxed(self, mdl, relaxable_groups, optimize, limits, parameters=None):
        mdl.fatal("No CPLEX DLL: model cannot be relaxed!")
        return False, 0

    @staticmethod
    def make_from_model(mdl):
        eng = NoSolveEngine(mdl)
        eng._incrementVars(mdl.number_of_variables)
        eng._incrementCts(mdl.number_of_constraints)
        return eng


class ZeroSolveEngine(IndexerEngine):
    # INTERNAL: a dummy engine that says it can solve
    # but returns an all-zero solution.
    def __init__(self, mdl, **kwargs):
        IndexerEngine.__init__(self)  # pragma: no cover
        self._last_solved_parameters = None

    @property
    def last_solved_parameters(self):
        return self._last_solved_parameters

    @property
    def name(self):
        return "zero_solve"  # pragma: no cover

    def solve(self, mdl, parameters):
        # remember last solved params
        self._last_solved_parameters = parameters.clone() if parameters is not None else None
        # sets all variable values to zero
        # --- print parameters ---
        if parameters is None:
            print("DEBUG> parameters: None")
        else:
            if parameters.has_nondefaults():
                print("DEBUG> parameters:")
                parameters.print_information(indent_level=8)  #
            else:
                print("DEBUG> parameters: defaults")
        # ---
        # return a feasible value: max of zero and the lower bound
        zlb_map = {v: max(0, v.lb) for v in mdl.iter_variables() if v.lb != 0}
        obj = mdl.objective_expr.constant
        return SolveSolution(mdl, obj=obj, var_value_map=zlb_map, engine_name=self.name)  # pragma: no cover

    def can_solve(self):
        return True  # pragma: no cover

    def solve_relaxed(self, mdl, relaxable_groups, optimize, limits, parameters=None):
        return True, 0  # pragma: no cover

    def get_solve_details(self):
        # 1 is cplex status optimal...
        return SolveDetails(time=0, dettime=0, status=1, status_string="OPTIMAL")


class FakeFailEngine(IndexerEngine):
    # INTERNAL: a dummy engine that says it can solve
    # but always fail, and returns None.
    def __init__(self, mdl, **kwargs):
        IndexerEngine.__init__(self)  # pragma: no cover

    def name(self):
        return "no_solve_engine"  # pragma: no cover

    def solve(self, mdl, parameters):
        # solve fails equivalent to returning None
        return None  # pragma: no cover

    def can_solve(self):
        return True  # pragma: no cover

    def solve_relaxed(self, mdl, relaxable_groups, optimize, limits, parameters=None):
        return False, 0  # pragma: no cover

    def get_solve_status(self):
        return JobSolveStatus.INFEASIBLE_SOLUTION  # pragma: no cover

    def get_solve_details(self):
        return SolveDetails(time=0, dettime=0, status=3, status_string="infeasible")


class RaiseErrorEngine(IndexerEngine):
    # INTERNAL: a dummy engine that says it can solve
    # but always raises an exception, this is for testing

    def _simulate_error(self):
        raise DOcplexException("simulate exception")

    def __init__(self, mdl, **kwargs):
        IndexerEngine.__init__(self)  # pragma: no cover

    def name(self):
        return "raise_engine"  # pragma: no cover

    def solve(self, mdl, parameters):
        # solve fails equivalent to returning None
        self._simulate_error()
        return None  # pragma: no cover

    def can_solve(self):
        return True  # pragma: no cover

    def solve_relaxed(self, mdl, relaxable_groups, optimize, limits, parameters=None):
        self._simulate_error()
        return False, 0  # pragma: no cover

    def get_solve_status(self):
        return JobSolveStatus.INFEASIBLE_SOLUTION  # pragma: no cover
