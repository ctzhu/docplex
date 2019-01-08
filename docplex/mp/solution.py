# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

from __future__ import print_function

import json
import math
import six
import sys

from six import iteritems, iterkeys
from docloud.status import JobSolveStatus

from docplex.mp.compat23 import StringIO
from docplex.mp.utils import is_iterable, is_number, is_string
from docplex.mp.utils import make_output_path2
from docplex.mp.linear import Var


# noinspection PyAttributeOutsideInit
class SolveSolution(object):
    """
    The :class:`SolveSolution` class holds the result of a solve.
    """

    # a symbolic value for no objective ?
    NO_OBJECTIVE_VALUE = -1e+75

    @staticmethod
    def _is_discrete_value(v):
        return v == int(v)

    def __init__(self, model, var_value_map=None, obj=None, name=None, engine_name=None, keep_zeros=True,
                 rounding=False):
        """ SolveSolution(model, var_valeu_map, obj, name)

        Creates a new solution object, associated to a a model.

        Args:
            model: The model to which the solution is associated. This model cannot be changed.

            obj: The value of the objective in the solution. A value of None means the objective is not defined at the
                time the solution is created, and will be set later.

            var_value_map: a Python dictionary containing associtaions of variables to values.

            name: a name for the solution. The default is None, in which case the solution is named after the
                model name.

        :return: A solution object.
        """
        assert model is not None
        assert engine_name is None or is_string(engine_name)
        assert obj is None or is_number(obj)

        self.__model = model
        self._checker = model
        self._name = name
        self._problem_name = model.name
        self._problem_objective_expr = model.objective_expr if model.has_objective() else None
        self.__objective__ = self.NO_OBJECTIVE_VALUE if obj is None else obj
        self.__engine_name = engine_name
        self.__var_value_map = {}
        self.__attr_map = {}
        self.__round_discrete = rounding
        self._solve_status = JobSolveStatus.UNKNOWN

        if var_value_map is not None:
            self._store_var_value_map(var_value_map, keep_zeros=keep_zeros, rounding=rounding)

    @staticmethod
    def make_engine_solution(model, var_value_map=None, obj=None, engine_name=None):
        # INTERNAL
        sol = SolveSolution(model,
                            var_value_map=var_value_map, obj=obj,
                            engine_name=engine_name,
                            rounding=True,
                            keep_zeros=False)
        return sol

    def _get_var_by_name(self, varname):
        return self.__model.get_var_by_name(varname)

    def clear(self):
        """ Clears all solve result data.

        All data related to the model are left unchanged.
        """
        self.__var_value_map = {}
        self.__objective__ = self.NO_OBJECTIVE_VALUE
        self.__attr_map = {}
        self._solve_status = JobSolveStatus.UNKNOWN

    def is_empty(self):
        """
        Checks whether the solution is empty.

        Returns:
            Boolean: True if the solution is empty; in other words, the solution has no defined objective and no variable value.
        """
        return not self.has_objective() and not self.__var_value_map

    @property
    def problem_name(self):
        return self._problem_name

    def get_name(self):
        """ This property allows to get/set a name on the solution.

        In some cases , it might be interesting to build different solutions for the same model,
        in this case, use the name property to disinguish them.

        """
        return self._name

    def set_name(self, solution_name):
        self._checker.typecheck_string(solution_name, accept_empty=False, accept_none=True)
        self._name = solution_name

    name = property(get_name, set_name)


    def _resolve_var(self, var_key, do_raise):
        # INTERNAL: accepts either strings or variable objects
        # returns a variable or None
        if isinstance(var_key, Var):
            var = var_key
        elif is_string(var_key):
            var = self._get_var_by_name(var_key)
            # var might be None here if the name is unknown
        else:
            var = None
        # --
        if var is None:
            if do_raise:
                self.model.fatal("Expecting variable or name, got: {0!r}", var_key)
            else:
                self.model.warning("Expecting variable or name, got: {0!r} - ignored", var_key)
        return var

    def _typecheck_var_key_value(self, var_key, value, caller):
        # INTERNAL
        self._checker.typecheck_num(value, caller=caller)
        if not is_string(var_key) and not isinstance(var_key, Var):
            self.model.fatal("{0} expects either Var or string, got: {1!r}", caller, var_key)

    def add_var_value(self, var_key, value):
        """ Adds a new (variable, value) pair to this solution.

        Args:
            var_key: A decision variable (:class:`docplex.mp.linear.Var`) or a variable name (string).
            value (number): The value of the variable in the solution.
        """
        self._typecheck_var_key_value(var_key, value, caller="Solution.add_var_value")
        self._set_var_value(var_key, value, keep_zero=True, rounding=False, do_warn_on_non_discrete=True)

    def set_var_value(self, var_key, value, keep_zero, rounding, do_warn_on_rounding):
        # INTERNAL
        self._typecheck_var_key_value(var_key, value, caller="Solution.add_var_value")
        self._set_var_value(var_key, value, keep_zero, rounding, do_warn_on_rounding)

    def _set_var_value(self, var_key, value, keep_zero, rounding, do_warn_on_non_discrete):
        # INTERNAL: no checks done.
        if value != 0 or keep_zero:
            var = self._resolve_var(var_key, do_raise=False)
            if var is not None:
                self._set_var_value_internal(var, value, rounding, do_warn_on_non_discrete=do_warn_on_non_discrete)

    def _set_var_value_internal(self, var, value, rounding, do_warn_on_non_discrete):
        # INTERNAL, no check
        stored_value = value
        if var.is_discrete():
            if not self._is_discrete_value(value):
                if rounding:
                    stored_value = self.model.round_nearest(value)
                if do_warn_on_non_discrete:
                    if rounding:
                        self.error_handler.warning(
                            "Trying to assign non-discrete value: {1} to discrete variable {0} - rounded to {2}",
                            (var, value, stored_value))
                    else:
                        self.error_handler.warning(
                            "Discrete variable {0!r} has been assigned non-discrete value: {1}",
                            (var, value))
        # ---
        self.__var_value_map[var] = stored_value

    def is_attributes_fetched(self, attr_name):
        return attr_name and attr_name in self.__attr_map

    @property
    def model(self):
        """
        This property returns the model associated with the solution.
        """
        return self.__model

    @property
    def error_handler(self):
        return self.__model.error_handler

    def get_objective_value(self):
        """
        Gets the objective value as defined in the solution.
        When the objective value has not been defined, a special value `NO_SOLUTION` is returned.
        To check whether the objective has been set, use :func:`has_objective`.

        Returns:
            float: The value of the objective as defined by the solution.
        """
        return self.__objective__

    def set_objective_value(self, obj):
        """
        Sets the objective value of the solution.
        
        Args:
            obj (float): The value of the objective in the solution.
        """
        self.__objective__ = obj

    def has_objective(self):
        """
        Checks whether or not the objective has been set.

        Returns:
            Boolean: True if the solution defines an objective value.
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
            # need to check var_keys and values
            self.set_var_value(var_key=e, value=val, keep_zero=keep_zeros, rounding=rounding, do_warn_on_rounding=False)

    def _store_attribute_results(self, var_rc_results, ct_dual_results, ct_slacks_results):
        self._store_attribute_result("reduced_costs", var_rc_results, is_variable=True)
        self._store_attribute_result("duals", ct_dual_results, is_variable=False)
        self._store_attribute_result("slacks", ct_slacks_results, is_variable=False)

    def _store_attribute_result(self, attr_name, attr_idx_map, is_variable):
        """
        Stores attribute results from the engine; results are expected as a dictionary of index->float.
        :param attr_name: The name of the attribute (a string).
        :param attr_idx_map: The dicitonary of attribute values by index.
        :param is_variable: UGLY, used to determine which comportement of indices
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
            iterator: A dict-style iterator which returns a two-component tuple (variable, value)
            for all variables mentioned in the solution.
        """
        return iteritems(self.__var_value_map)

    def iter_variables(self):
        """Iterates over all variables mentioned in the solution.

        Returns:
           iterator: An iterator object over all variables mentioned in the solution.
        """
        return iterkeys(self.__var_value_map)

    def contains(self, dvar):
        """
        Checks whether or not a decision variable is mentioned in the solution.

        This predicate can also be used in the form `var in solution`, because the
        :func:`__contains_` method has been redefined for this purpose.

        Args:
            dvar (:class:`docplex.mp.linear.Var`): The variable to check.

        Returns:
            Boolean: True if the variable is mentioned in the solution.
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
            dvar_arg: A decision variable (:class:`docplex.mp.linear.Var`) or a variable name (string).

        Returns:
            float: The value of the variable in the solution.
        """
        dvar = self._resolve_var(dvar_arg, do_raise=True)
        return self.__var_value_map.get(dvar, 0) if dvar is not None else 0

    @property
    def number_of_var_values(self):
        """ This property returns the number of variable values stored in this solution.

        """
        return len(self.__var_value_map)


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
                print_zeros=True,
                header_fmt="solution for: {0:s}",
                objective_fmt="{0}: {1:.{prec}f}",
                value_fmt="{varname:s} = {value:.{prec}f}",
                iter_vars=None,
                **kwargs):
        print_generated = kwargs.get("print_generated", False)
        problem_name = self._problem_name
        if header_fmt and problem_name:
            print(header_fmt.format(problem_name))
        if self._problem_objective_expr is not None and objective_fmt and self.has_objective():
            obj_prec = self.model.objective_expr.float_precision
            obj_name = self._problem_objective_name()
            print(objective_fmt.format(obj_name, self.__objective__, prec=obj_prec))
        if iter_vars is None:
            iter_vars = self.iter_variables()
        print_counter = 0
        for dvar in iter_vars:
            if print_generated or not dvar.is_generated():
                var_value = self.get_value(dvar)
                if print_zeros or var_value != 0:
                    print_counter += 1
                    print(value_fmt.format(varname=dvar.name,
                                           value=var_value,
                                           prec=dvar.float_precision,
                                           counter=print_counter))

    def to_string(self, print_zeros=True):
        oss = StringIO()
        self.to_stringio(oss, print_zeros=print_zeros)
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

    def to_stringio(self, oss, print_zeros=True):
        problem_name = self._problem_name
        if problem_name:
            oss.write("solution for: %s\n" % problem_name)
        if self._problem_objective_expr is not None and self.has_objective():
            obj_name = self._problem_objective_name()
            oss.write("%s: %g\n" % (obj_name, self.__objective__))

        value_fmt = "{varname:s}={value:.{prec}f}"
        for dvar, val in self.iter_var_values():
            if not dvar.is_generated():
                var_value = self.get_value(dvar)
                if print_zeros or var_value != 0:
                    oss.write(value_fmt.format(varname=dvar.name, value=var_value, prec=dvar.float_precision))
                    oss.write("\n")

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        if self.has_objective():
            s_obj = "obj={0:g}".format(self.objective_value)
        else:
            s_obj = "obj=N/A"
        s_values = ",".join(["{0}:{1:g}".format(var.name, val) for var, val in iteritems(self.__var_value_map)])
        return "docplex.mp.solution.SolveSolution({0},values={{{1}}})".format(s_obj, s_values)

    def print_mst(self):
        """
        Writes the solution in an output stream "out" (assumed to satisfy the file interface)
        in CPLEX MST format.
        """
        SolutionMSTPrinter.print_one_solution(sol=self, out=sys.stdout)

    def print_mst_to_stream(self, out):
        SolutionMSTPrinter.print_to_stream(self, out)

    def export_as_mst_string(self):
        return SolutionMSTPrinter.print_to_string(self)

    def export_as_mst(self, path=None, basename=None):
        """ Exports a solution to a file in CPLEX mst format.

        Args:
            basename: Controls the basename with which the solution is printed.
                Accepts None, a plain string, or a string format.
                If None, the model's name is used.
                If passed a plain string, the string is used in place of the model's name.
                If passed a string format (either with %s or {0}), this format is used to format the
                model name to produce the basename of the written file.

            path: A path to write the file, expects a string path or None.
                Can be a directory, in which case the basename
                that was computed with the basename argument is appended to the directory to produce
                the file.
                If given a full path, the path is directly used to write the file, and
                the basename argument is not used.
                If passed None, the output directory will be ``tempdir.gettempdir()``.

        Example:
            Assuming the solution has the name "prob":

            ``sol.export_as_mst()`` will write file prob.mst in a temporary directory.

            ``sol.export_as_mst(path="c:/temp/myprob1.mst")`` will write file "c:/temp/myprob1.mst".

            ``sol.export_as_mst(basename="my_%s_mipstart", path ="z:/home/")`` will write "z:/home/my_prob_mipstart.mst".

        """
        mst_path = make_output_path2(actual_name=self._problem_name,
                                     extension=SolutionMSTPrinter.mst_extension,
                                     path=path,
                                     basename_arg=basename)
        if mst_path:
            self.print_mst_to_stream(mst_path)

    def get_printer(self, key):
        printers = {'json': SolutionJSONPrinter,
            'xml': SolutionMSTPrinter
            }
        printer = printers.get(key.lower())
        if not printer:
            raise ValueError("format must be one of {}".format(self.printers.keys()))
        return printer


    def export(self, file_or_filename, format="json"):
        """ Export this solution.
        
        Args:
            file_or_filename: If ``file_or_filename`` is a string, this argument contains the filename to
                write to. If this is a file object, this argument contains the file object to write to.
            format: The format of the solution. The format can be:
                - json
                - xml
        """

        printer = self.get_printer(format)

        if isinstance(file_or_filename, six.string_types):
            fp = open(file_or_filename, "w")
            close_fp = True
        else:
            fp = file_or_filename
            close_fp = False
        try:
            printer.print_to_stream(self, fp)
        finally:
            if close_fp:
                fp.close()

    def export_as_string(self, format="json"):
        oss = StringIO()
        self.export(oss, format=format)
        return oss.getvalue()


    def check_as_mip_start(self, error_handler=None):
        """Checks that this solution is a valid MIP start.

        To be valid, it must have:

            * at least one discrete variable (integer or binary), and
            * the values for decision variables should be consistent with the type.

        Args:
            error_handler: An instance of an error handler or None.

        Returns:
            Boolean: True if this solution is a valid MIP start.
        """
        if 0 == len(self.__var_value_map):
            if error_handler:
                error_handler.error("MIP start solution is empty, provide at least one intere/boolean variable value")
            return False

        discrete_vars = (dv for dv in self.iter_variables() if dv.is_discrete())
        count_values = 0
        count_errors = 0
        for dv in discrete_vars:
            sol_value = self.get_value(dv)
            if not dv.typecheck_initial_value(sol_value):
                count_errors += 1
                if error_handler:
                    error_handler.error("Wrong initial value for variable {0}: {1}, type: {2!s}",  # pragma: no cover
                                        dv.name, sol_value, dv.vartype)  # pragma: no cover
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


class SolutionMSTPrinter(object):
    # header containsthe final newline
    mst_header = """<?xml version = "1.0" standalone="yes"?>
<?xml-stylesheet href="https://www.ilog.com/products/cplex/xmlv1.0/solution.xsl" type="text/xsl"?>

"""
    mst_extension = ".mst"

    one_solution_start_tag = "<CPLEXSolution version=\"1.0\">"
    one_solution_end_tag = "</CPLEXSolution>"

    # used when several solutions are present
    many_solution_start_tag = "<CPLEXSolutions version=\"1.0\">"
    many_solution_end_tag = "</CPLEXSolutions>"

    @staticmethod
    def print_signature(out):
        from docplex.version import docplex_version_string

        out.write("<!-- This file has been generated by DOcplex version {}  -->\n".format(docplex_version_string))

    @classmethod
    def print(cls, out, solutions):
        # solutions can be either a plain solution or a sequence or an iterator
        if not is_iterable(solutions):
            cls.print_one_solution(solutions, out)
        else:
            sol_seq = list(solutions)
            nb_solutions = len(sol_seq)
            assert nb_solutions > 0
            if 1 == nb_solutions:
                cls.print_one_solution(sol_seq[0], out)
            else:
                cls.print_many_solutions(sol_seq, out)

    @classmethod
    def print_one_solution(cls, sol, out, print_header=True):
        if print_header:
            out.write(cls.mst_header)
            cls.print_signature(out)
        # <CPLEXSolution version="1.0">
        out.write(cls.one_solution_start_tag)
        out.write("\n")

        # <header
        # problemName="foo"
        # objectiveValue="42"
        # />
        out.write(" <header\n   problemName=\"{0}\"\n".format(sol.problem_name))
        if sol.has_objective():
            out.write("   objectiveValue=\"{0}:g\"\n".format(sol.objective_value))
        out.write("  />\n")

        #  <variables>
        #    <variable name="x1" index ="1" value="3.14"/>
        #  </variables>
        out.write(" <variables>\n")
        for dvar, val in sol.iter_var_values():
            var_name = dvar.name
            var_value = sol[dvar]
            var_index = dvar.index
            out.write("  <variable name=\"{0}\" index=\"{1}\" value=\"{2:g}\"/>\n"
                      .format(var_name, var_index, var_value))
        out.write(" </variables>\n")

        #  </CPLEXSolution version="1.0">
        out.write(cls.one_solution_end_tag)
        out.write("\n")

    @classmethod
    def print_many_solutions(cls, sol_seq, out):
        out.write(cls.mst_header)
        cls.print_signature(out)
        # <CPLEXSolutions version="1.0">
        out.write(cls.many_solution_start_tag)
        out.write("\n")

        for sol in sol_seq:
            cls.print_one_solution(sol, out, print_header=False)

        # <CPLEXSolutions version="1.0">
        out.write(cls.many_solution_end_tag)
        out.write("\n")

    @classmethod
    def print_to_stream(cls, solutions, out, extension=mst_extension):
        if out is None:
            # prints on standard output
            cls.print(sys.stdout, solutions)
        elif isinstance(out, str):
            # a string is interpreted as a path name
            path = out if out.endswith(extension) else out + extension
            with open(path, "w") as of:
                cls.print_to_stream(solutions, of)
                # print("* file: %s overwritten" % path)
        else:
            try:
                cls.print(out, solutions)

            except AttributeError:  # pragma: no cover
                pass  # pragma: no cover
                # stringio will raise an attribute error here, due to with
                # print("Cannot use this an output: %s" % str(out))

    @classmethod
    def print_to_string(cls, solutions):
        oss = StringIO()
        cls.print_to_stream(solutions, out=oss)
        return oss.getvalue()


from json import JSONEncoder


class SolutionJSONEncoder(JSONEncoder):
    def default(self, solution):
        n = {'CPLEXSolution': self.encode_solution(solution)}
        return n

    def encode_solution(self, solution):
        n = {}
        n["version"] = "1.0"
        n["header"] = self.encode_header(solution)
        n["variables"] = self.encode_variables(solution)
        return n

    def encode_header(self, solution):
        n = {}
        n["problemName"] = solution.problem_name
        if solution.has_objective():
            n["objectiveValue"] = "{}".format(solution.objective_value)
        return n

    def encode_variables(self, sol):
        n = []
        for dvar, val in sol.iter_var_values():
            v = {"index": "{}".format(dvar.index),
                 "name": dvar.name,
                 "value": "{}".format(sol[dvar])}
            n.append(v)
        return n


class SolutionJSONPrinter(object):
    json_extension = ".json"

    @classmethod
    def print(cls, out, solutions, indent=None):
        # solutions can be either a plain solution or a sequence or an iterator
        if not is_iterable(solutions):
            cls.print_one_solution(solutions, out, indent=indent)
        else:
            sol_seq = list(solutions)
            nb_solutions = len(sol_seq)
            assert nb_solutions > 0
            if 1 == nb_solutions:
                cls.print_one_solution(sol_seq[0], out, indent=indent)
            else:
                cls.print_many_solutions(sol_seq, out, indent=indent)

    @classmethod
    def print_one_solution(cls, sol, out, indent=None):
        out.write(json.dumps(sol, cls=SolutionJSONEncoder, indent=indent))

    @classmethod
    def print_many_solutions(cls, sol_seq, out, indent=None):
        encoder = SolutionJSONEncoder()
        n = {"CPLEXSolutions": [encoder.default(sol) for sol in sol_seq]}
        out.write(json.dumps(n, indent=indent))

    @classmethod
    def print_to_stream(cls, solutions, out, extension=json_extension, indent=None):
        if out is None:
            # prints on standard output
            cls.print(sys.stdout, solutions, indent=indent)
        elif isinstance(out, str):
            # a string is interpreted as a path name
            path = out if out.endswith(extension) else out + extension
            with open(path, "w") as of:
                cls.print_to_stream(solutions, of, indent=indent)
                # print("* file: %s overwritten" % path)
        else:
            try:
                cls.print(out, solutions, indent=indent)

            except AttributeError:  # pragma: no cover
                pass  # pragma: no cover
                # stringio will raise an attribute error here, due to with
                # print("Cannot use this an output: %s" % str(out))

    @classmethod
    def print_to_string(cls, solutions, indent=None):
        oss = StringIO()
        cls.print_to_stream(solutions, out=oss, indent=indent)
        return oss.getvalue()


