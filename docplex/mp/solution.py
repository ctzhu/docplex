# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

from __future__ import print_function

import math

from six import iteritems, iterkeys

from docplex.mp.utils import is_iterable, is_number, is_string, StringIO, RedirectedOutputContext
from docloud.status import JobSolveStatus
from docplex.mp.linear import Var


# noinspection PyAttributeOutsideInit
class SolveSolution(object):
    """
    The :class:`SolveSolution` class holds the result of a solve.
    """

    # a symbolic value for no objective ?
    NO_OBJECTIVE_VALUE = -1e+75

    def __init__(self, model, obj=None, var_value_map=None, engine_name=None, keep_zeros=False, rounding=False):
        """ Creates a new solution object, associated to a a model.

        :param model: The model to which the solution is associated. This model cannot be changed.
        :param obj: The value of the objective in the solution. A value of None mans the objective is not defined at the
        time the solution is created, and will be set later.
        :param var_value_map:
        :param engine_name:
        :param keep_zeros:

        :return: A solution object.
        """
        assert model is not None
        assert engine_name is None or isinstance(engine_name, str)
        assert obj is None or is_number(obj)

        self.__model = model
        self._problem_name = model.name
        self._problem_objective_expr = model.objective_expr if model.has_objective() else None
        self.__objective__ = self.NO_OBJECTIVE_VALUE if obj is None else obj
        self.__engine_name = engine_name
        self.__var_value_map = {}
        self.__attr_map = {}
        self.__round_discrete = True
        self._solve_status = JobSolveStatus.UNKNOWN

        if var_value_map:
            self._store_var_value_map(var_value_map, keep_zeros=keep_zeros, rounding=rounding)

    def _get_var_by_name(self, varname):
        return self.__model.get_var_by_name(varname)

    def is_empty(self):
        """
        Checks whether the solution is empty.

        Returns:
            True if the solution is empty; in other words, it has no defined objective and no variable value.
        """
        return not self.has_objective() and not self.__var_value_map

    def add_var_value(self, var_key, value, keep_zero=True, rounding=False):
        """ Adds a new pair of (var, value) to this solution.

        Args:
            var_key: A decision variable (:class:`docplex.mp.linear.Var`) or a variable name (a string).
            value: A number, the value of the variable in the solution.
        """
        self_model = self.model
        if value != 0 or keep_zero:
            if isinstance(var_key, str):
                var = self._get_var_by_name(var_key)
                if not var:
                    self.error_handler.warning("No variable with name: %s - ignored", var_key)
            else:
                var = var_key
            if var:
                if rounding and var.is_discrete():
                    stored_value = self_model.round_nearest(value)
                else:
                    stored_value = value
                self.__var_value_map[var] = stored_value

    def is_attributes_fetched(self, attr_name):
        return attr_name and attr_name in self.__attr_map

    @property
    def model(self):
        """
        A property that gets the model associated with the solution.
        """
        return self.__model

    @property
    def error_handler(self):
        return self.__model.error_handler

    def get_objective_value(self):
        """
        Gets the objective value as defined in the solution.
        When the objective value has not been defined, a special value `NO_SOLUTION` is returned.
        You can check whether the objective has been set using :func:`has_objective`.

        Returns:
            The value of the objective as defined by the solution.
        """
        return self.__objective__

    def set_objective_value(self, obj):
        """
        Sets the objective value of the solution.
        
        Args:
            obj (float): The value of the objective in the solution (a floating-point number).
        """
        self.__objective__ = obj

    def has_objective(self):
        """

        :return: True if the solution defines an objective value.
        """
        return self.__objective__ != self.NO_OBJECTIVE_VALUE

    def _has_problem_objective(self):
        return self.model.has_objective()

    objective_value = property(get_objective_value, set_objective_value)

    @property
    def solve_status(self):
        return self._solve_status

    def _set_solve_status(self, new_status):
        # INTERNAL
        self._solve_status = new_status

    def _store_var_value_map(self, key_value_map, keep_zeros=False, rounding=False):
        for e, val in iteritems(key_value_map):
            self.add_var_value(var_key=e, value=val, keep_zero=keep_zeros, rounding=rounding)

    def _copy_var_values(self):
        # obsolete
        pass

    def _store_attribute_results(self, var_rc_results, ct_dual_results, ct_slacks_results):
        self._store_attribute_result("reduced_costs", var_rc_results, is_variable=True)
        self._store_attribute_result("duals", ct_dual_results, is_variable=False)
        self._store_attribute_result("slacks", ct_slacks_results, is_variable=False)

    def _store_attribute_result(self, attr_name, attr_idx_map, is_variable):
        """
        Stores attribute results from the engine; results are expected as a dictionary of index->float.
        :param attr_name: The name of the attribute (a string).
        :param attr_idx_map: The dicitonary of attribute values by index.
        :param is_variable: UGLY, used to determine which compartement of indices
        :return: None
        """
        if attr_idx_map:
            mdl = self.model
            obj_mapper = mdl.get_var_by_index if is_variable else mdl.get_constraint_by_index
            attr_obj_map = {obj_mapper(idx): attr_val
                            for idx, attr_val in iteritems(attr_idx_map)
                            if obj_mapper(idx) is not None and attr_val != 0}
        else:
            attr_obj_map = {}
        self.__attr_map[attr_name] = attr_obj_map

    def iter_var_values(self):
        """Iterates over the (variable, value) pairs in the solution.

        Returns:
            A dict-style iterator which returns a two-component tuple (variable, value)
            for all variables mentioned in the solution.
        """
        return iteritems(self.__var_value_map)

    def iter_variables(self):
        """Iterates over all variables mentioned in the solution.

        Returns:
           An iterator object.
        """
        return iterkeys(self.__var_value_map)

    def contains(self, dvar):
        """
        Checks whether or not a decision variable is mentioned in the solution.
        Note that this predicate can also be used in the form `var in solution`, because the
        :func:`__contains_` method has been redefined for this purpose.

        Args:
            dvar (:class:`docplex.mp.linear.Var`): The variable to check.

        Returns:
            True if the variable is mentioned in the solution.
        """
        return dvar in self.__var_value_map

    def __contains__(self, dvar):
        return self.contains(dvar)

    def get_value(self, dvar_arg):
        """
        Gets the value of a solution variable in a solution.
        If the variable is not mentioned in the solution,
        the method returns 0 and does not raise an exception.
        Note that this method can also be used as :func:`solution[dvar]`
        because the :func:`__getitem__` method has been overloaded.

        Args:
            dvar-arg: A decision variable (:class:`docplex.mp.linear.Var`) or a variable name (a string).

        Returns:
            A floating-point number, the value of the variable in the solution.
        """
        if isinstance(dvar_arg, Var):
            return self.__var_value_map.get(dvar_arg, 0)
        elif is_string(dvar_arg):
            dvar = self._get_var_by_name(dvar_arg)
            return self.__var_value_map.get(dvar, 0) if dvar is not None else 0
        else:
            self.__model.fatal("Solution.get_value accepts variables or names, got: {0!s}", dvar_arg)

    def __getitem__(self, dvar):
        return self.get_value(dvar)

    def __iter__(self):
        # INTERNAL: this is necessary to prevent solution from being an iterable.
        # as it follows getitem protocol, it can mistakenly be interpreted as an iterable
        raise TypeError

    def equals_solution(self, other, check_models=False, check_explicit=False, obj_precision=1e-3, var_precision=1e-6):
        if check_models and (self.model != other.model):
            return False

        if math.fabs(self.objective_value - other.objective_value) >= obj_precision:
            return False

        for dvar, val in self.iter_var_values():
            if check_explicit and not other.contains(dvar):
                return False
            this_val = self.get_value(dvar)
            other_val = other.get_value(dvar)
            if math.fabs(this_val - other_val) >= var_precision:
                return False

        for other_dvar, other_val in other.iter_var_values():
            if check_explicit and not self.contains(other_dvar):
                return False
            this_val = self.get_value(other_dvar)
            other_val = other.get_value(other_dvar)
            if math.fabs(this_val - other_val) >= var_precision:
                return False

        return True

    def get_attribute(self, mobjs, attr, default_attr_value=0):
        assert not self.is_empty()
        assert is_iterable(mobjs)
        if not mobjs:
            return []
        elif attr not in self.__attr_map:
            # warn
            return [0] * len(mobjs)
        else:
            attr_map = self.__attr_map[attr]
            return [attr_map.get(mobj, default_attr_value) for mobj in mobjs]

    def display_attributes(self):
        for attr_key in self.__attr_map:
            attr_value_map = self.__attr_map[attr_key]

            print("#{0}={1:d}".format(attr_key, len(attr_value_map)))
            for obj, attr_val in iteritems(attr_value_map):
                obj_qualifier = obj.name if obj.has_username() else str(obj)
                print(" {0}.{1} = {2}".format(obj_qualifier, attr_key, attr_val))

    def display(self,
                print_zeros=False,
                header_fmt="solution for: {0:s}",
                objective_fmt="{0}: {1:.{prec}f}",
                value_fmt="{varname:s} = {value:.{prec}f}",
                iter_vars=None):
        problem_name = self._problem_name
        if header_fmt and problem_name:
            print(header_fmt.format(problem_name))
        if self._problem_objective_expr is not None and objective_fmt and self.has_objective():
            obj_prec = self.model.objective_expr.float_precision
            obj_name = self._problem_objective_name()
            print(objective_fmt.format(obj_name, self.__objective__, prec=obj_prec))
        if iter_vars is None:
            iter_vars = self.model.iter_variables() if print_zeros else self.iter_variables()
        print_counter = 0
        for dvar in iter_vars:
            if not dvar.is_generated():
                var_value = self.get_value(dvar)
                if print_zeros or var_value != 0:
                    print_counter += 1
                    print(value_fmt.format(varname=dvar.name,
                                           value=var_value,
                                           prec=dvar.float_precision,
                                           counter=print_counter))

    def to_string(self):
        oss = StringIO()
        self.to_stringio(oss)
        return oss.getvalue()

    def _problem_objective_name(self, default_obj_name="objective"):
        # INTERNAL
        # returns the string used for displaying the objective
        # if the problem has an objective with a name, use it
        # else return the default (typically "objective"
        self_objective_expr = self._problem_objective_expr
        if self_objective_expr is not None and self_objective_expr.has_name():
            return self_objective_expr.name
        else:
            return default_obj_name

    def to_stringio(self, oss):
        problem_name = self._problem_name
        if problem_name:
            oss.write("solution for: %s\n" % problem_name)
        if self._problem_objective_expr is not None and self.has_objective():
            obj_name = self._problem_objective_name()
            oss.write("%s: %g\n" % (obj_name, self.__objective__))

        value_fmt="{varname:s}={value:.{prec}f}"
        for dvar, val in self.iter_var_values():
             if not dvar.is_generated():
                var_value = self.get_value(dvar)
                if var_value != 0:
                    oss.write(value_fmt.format(varname=dvar.name, value=var_value, prec=dvar.float_precision))
                    oss.write("\n")

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return "docplex.mp.solution.SolveSolution({0}, {1})".format(self.objective_value, len(self.__var_value_map))

    def print_mst(self):
        """
        Writes the solution in an output stream "out" (assumed to satisfy the file interface)
        in CPLEX MST format.
        """
        mst_start = """<?xml version = "1.0" standalone="yes"?>
<?xml-stylesheet href="https://www.ilog.com/products/cplex/xmlv1.0/solution.xsl" type="text/xsl"?>
<CPLEXSolution version="1.0">
"""
        mst_end = "</CPLEXSolution>"

        print(mst_start)
        # header
        print(" <header\n   problemName=\"{0}\"".format(self._problem_name))
        if self.has_objective():
            print("   objectiveValue=\"{0}:g\"".format(self.objective_value))
        print("  />")
        print(" <variables>")
        for dvar, val in self.iter_var_values():
            var_name = dvar.name
            var_value = self[dvar]
            var_index = dvar.index
            print("  <variable name=\"{0}\" index=\"{1}\" value=\"{2:g}\"/>".format(var_name, var_index, var_value))
        print(" </variables>")
        print(mst_end)

    mst_extension = ".mst"

    def print_mst_to_stream(self, out):
        if out is None:
            # prints on standard output
            self.print_mst()
        elif isinstance(out, str):
            # a string is interpreted as a path name
            path = out if out.endswith(self.mst_extension) else out + self.mst_extension
            with open(path) as of:
                self.print_mst_to_stream(of)
                # print("* file: %s overwritten" % path)
        else:
            try:
                with RedirectedOutputContext(out, self.model.error_handler):
                    self.print_mst()

            except AttributeError:  # pragma: no cover
                pass  # pragma: no cover
                # stringio will raise an attribute error here, due to with
                # print("Cannot use this an output: %s" % str(out))

    def check_as_mip_start(self, error_handler=None):
        """
        Checks that this solution is a valid MIP start. To be valid, it must have:
            * at least one discrete variable (integer or binary), and
            * the value for integer/binary variables should be consistent with the type.

        Args:
            error_handler: An instance of an error handler or None.
        """

        discrete_vars = (dv for dv in self.iter_variables() if dv.is_discrete())
        count_values = 0
        count_errors = 0
        for dv in discrete_vars:
            sol_value = self.get_value(dv)
            if not dv.typecheck_initial_value(sol_value):
                count_errors += 1
                if error_handler:
                    error_handler.error("Wrong initial value for variable {0}: {1}, type: {2!s}",   # pragma: no cover
                                        dv.name, sol_value, dv.vartype)                             # pragma: no cover
            else:
                count_values += 1
        if count_values == 0:
            if error_handler:
                error_handler.error("MIP start contains no discrete variable")  # pragma: no cover
            return False
        else:
            return count_errors == 0

    def as_dict(self, keep_zeros=False):
        var_value_dict = {}
        # INTERNAL: return a dictionary of variable_name: variable_value
        for dvar, dval in self.iter_var_values():
            if not dvar.has_automatic_name() and (keep_zeros or dval != 0):
                var_value_dict[dvar.name] = dval
        return var_value_dict



class SolveDetails(object):

    UNKNOWN_STATUS = "*unknown*"

    def __init__(self, time=0, dettime=0, status=-1, status_string=None):
        self._time = max(time, 0)
        self._dettime = max(dettime, 0)
        self._cpx_solve_status = status
        if status_string is None:
            self._cpx_solve_status_string = self.UNKNOWN_STATUS
        else:
            self._cpx_solve_status_string = status_string

    @staticmethod
    def from_json(json_details):
        if not json_details:
            return SolveDetails.make_dummy()

        details = SolveDetails()
        details._time = float(json_details["cplex.time"])
        details._dettime = float(json_details["cplex.dettime"])
        details._cpx_solve_status = int(json_details["cplex.status"])
        details._cpx_solve_status_string = json_details["cplex.statusstring"]
        return details

    @staticmethod
    def make_dummy():
        dummy_details = SolveDetails(status_string="dummy")
        return dummy_details

    def __nonzero__(self):
        return self._solution is not None

    def get_time(self):
        return self._time

    def get_dettime(self):
        return self._dettime

    def __repr__(self):
        return "docplex.mp.solution.SolveDetails(time={0:g}, dettime={1:g}, status={2:s})"\
            .format(self._time, self._dettime, self._cpx_solve_status_string)

    def print_information(self):
        print("status ={0}".format(self._cpx_solve_status_string))
        print("time   ={0} s.".format(self._time))
        #print("dettime={0}".format(self._dettime))

