# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

import sys

from docplex.mp.engine import DummyEngine
from docplex.mp.utils import is_number, is_int, is_iterable, generate_constant
from docplex.mp.compat23 import izip

from docplex.mp.vartype import BinaryVarType, IntegerVarType, ContinuousVarType
from docplex.mp.linear import LinearConstraintType
from docplex.mp.basic import ObjectiveSense
from docplex.mp.linear import AbstractLinearConstraint, IndicatorConstraint
from docplex.mp.progress import ProgressData
from docplex.mp.solution import SolveSolution
from docplex.mp.sdetails import SolveDetails
import cplex

from six import iteritems
from contextlib import contextmanager

from enum import Enum
# CHECK THIS
# noinspection PyProtectedMember
from cplex._internal import _subinterfaces
from cplex.callbacks import MIPInfoCallback

# noinspection PyProtectedMember
import cplex._internal._constants as cpx_cst
from cplex.exceptions import CplexError

from docplex.mp.compat23 import fast_range

# gendoc: ignore


class ConnectListenersCallback(MIPInfoCallback):
    RELATIVE_EPS = 1e-5
    ABS_EPS = 1e-4

    # noinspection PyAttributeOutsideInit
    def initialize(self, listeners, node_period=-1):
        self.__listeners = listeners
        self.__pdata = ProgressData()
        self._start_time = -1
        self._start_dettime = -1
        # subset of listeners which listen to intermediate solutions.
        self.__solution_listeners = [l for l in listeners if l.requires_solution()]

    @property
    def last_incumbent(self):
        return self.__pdata.current_objective

    @property
    def last_best_bound(self):
        return self.__pdata.best_bound

    @property
    def last_nb_nodes(self):
        return self.__pdata.current_nb_nodes

    def __call__(self):
        has_incumbent = self.has_incumbent()

        if self._start_time < 0:
            self._start_time = self.get_start_time()
        if self._start_dettime < 0:
            self._start_dettime = self.get_start_dettime()

        pdata = self.__pdata
        pdata.has_incumbent = has_incumbent
        if has_incumbent:
            pdata.current_objective = self.get_incumbent_objective_value()
        pdata.best_bound = self.get_best_objective_value()
        pdata.mip_gap = self.get_MIP_relative_gap()
        pdata.current_nb_nodes = self.get_num_nodes()
        pdata.remaining_nb_nodes = self.get_num_remaining_nodes()
        pdata.time = self.get_time() - self._start_time
        pdata.det_time = self.get_dettime() - self._start_dettime

        for l in self.__listeners:
            l.notify_progress(pdata)
        if has_incumbent:
            # get incumbent values as a list of values (value[v] at position index[v])
            cpx_incumbent_values = self.get_incumbent_values()
            for sl in self.__solution_listeners:
                sl.notify_solution(cpx_incumbent_values)


# internal
class _CplexSyncMode(Enum):
    InSync, InResync, OutOfSync = [1, 2, 3]


class CplexIndexMode(Enum):
    # enumerated value for different ways to handle cplex indices.
    # Query is old-way: ask Cplex by name, very slow.
    # Guess means compute guessed indices, do not query
    #   UseReturn is new API: assume "add" returns a range and use it.
    Query, Guess, UseReturn = 1, 2, 3

    _mode2string = {"quey": Query, "guess": Guess, "return": UseReturn}

    @staticmethod
    def parse(text, default_mode):
        if not text:
            return default_mode
        else:
            return CplexIndexMode._mode2string.get(text.lower(), default_mode)


class _CpxWithParamsExecutionContext(object):
    # internal context manager to handle forcing parameters during relaxation.

    def __init__(self, cplex_to_use, parameters_to_use):
        self._cplex = cplex_to_use
        # store current parameters
        self._saved_params = cplex_to_use.parameters
        # store params to use
        self._parameters_to_use = parameters_to_use

    def __enter__(self):
        # replace parameters if needed
        if self._parameters_to_use is not None:
            self._cplex.parameters = self._parameters_to_use
        # return the Cplex instance with the overwritten parameters.
        return self._cplex

    def __exit__(self, exc_type, exc_val, exc_tb):
        # whatever happened, restore saved parameter values.
        self._cplex.parameters = self._saved_params


# noinspection PyProtectedMember
class CplexEngine(DummyEngine):
    """
        CPLEX engine wrapper.
    """
    CPX_RANGE_SYMBOL = 'R'

    def __init__(self, mdl, **kwargs):
        """
        INTERNAL
        :param mdl: the model
        :param index_mode: a string describing how cplex indices are to be managed
        :return:
        """
        DummyEngine.__init__(self)
        cpx = cplex.Cplex()

        # resetting DATACHECK to 0 has no measurable effect
        # cpx.parameters._set(1056, 0)

        index_mode = None
        if 'index_mode' in kwargs:
            index_mode = kwargs['index_mode']

        self._model = mdl
        self._saved_log_output = True  # initialization from model is deferred (pickle)
        self._index_mode = CplexIndexMode.parse(index_mode, CplexIndexMode.Guess)

        # deferred bounds changes, as dicts {var: num}
        self._var_lb_changed = {}
        self._var_ub_changed = {}

        self.__cplex = cpx

        self._solve_count = 0
        self._last_solve_status = False
        self._last_solve_details = None

        # for unpickling, remember to resync with model
        self._resync = _CplexSyncMode.InSync

        # remember truly allocated indices
        self._last_used_ct_index = -1
        self._last_used_var_index = -1

        self._cplex_vartype_map = {BinaryVarType: 'B',
                                   IntegerVarType: 'I',
                                   ContinuousVarType: 'C'}
        self._cplex_cttype_map = {LinearConstraintType.EQ: 'E',
                                  LinearConstraintType.GE: 'G',
                                  LinearConstraintType.LE: 'L'}

        self._cplex_objsense_map = {ObjectiveSense.Minimize: self.__cplex.objective.sense.minimize,
                                    ObjectiveSense.Maximize: self.__cplex.objective.sense.maximize}

    def _mark_as_out_of_sync(self):
        self._resync = _CplexSyncMode.OutOfSync

    def _notify_linear_constraint_index(self, linct_idx):
        if is_iterable(linct_idx):
            idx = linct_idx[-1]
        else:
            idx = linct_idx
        if idx > self._last_used_ct_index:
            self._last_used_ct_index = idx

    def _notify_linear_constraint_deleted(self, linct_idx):
        if linct_idx >= 0:
            self._last_used_ct_index -= 1

    def _get_guessed_var_index(self):
        return self._last_used_var_index + 1

    def _notify_var_index(self, idx):
        if idx >= 0:
            self._last_used_var_index = max(self._last_used_var_index, idx)

    def _get_guessed_ct_index(self):
        """
        Makes a guess on the next value of constraint indices,
        when a block with size constraints ha sbeen allocated.

        Note: this comprises both linear constraints and range c onstraints
        :return:
        """
        return self._last_used_ct_index + 1

    def _guessed_ct_index_range(self, block_size):
        first_index = self._get_guessed_ct_index()
        return fast_range(first_index, first_index + block_size)

    def _set_trace_output(self, ofs):
        cpx = self.__cplex
        cpx.set_log_stream(ofs)
        cpx.set_results_stream(ofs)
        cpx.set_error_stream(ofs)
        cpx.set_warning_stream(ofs)

    def notify_trace_output(self, out):
        self_log_output = self._saved_log_output
        if self_log_output != out:
            self._set_trace_output(out)
            self._saved_log_output = out

    def get_var_index(self, dvar):
        self._resync_if_needed()
        dvar_name = dvar.name
        if not dvar_name:
            self.error_handler.fatal("cannot query index for anonymous object: {0!s}", dvar)
        else:
            return self.__cplex.variables.get_indices(dvar_name)

    def get_ct_index(self, ct):
        self._resync_if_needed()
        ctname = ct.name
        if not ctname:
            self.error_handler.fatal("cannot query index for anonymous constraint: {0!s}", ct)
        self_cplex = self.__cplex
        if isinstance(ct, AbstractLinearConstraint):
            return self_cplex.linear_constraints.get_indices(ctname)
        elif isinstance(ct, IndicatorConstraint):
            return self_cplex.indicator_constraints.get_indices(ctname)
        else:
            self.error_handler.fatal("unrecognized constraint to query index: {0!s}", ct)

    @property
    def error_handler(self):
        return self._model.error_handler

    def get_cplex(self):
        """
        Returns the underlying CPLEX object
        :return:
        """
        return self.__cplex

    def get_infinity(self):
        return cplex.infinity

    def _convert_model_value_to_cplex_value(self, type_descr, model_to_cplex_map, model_value):
        """
        :param type_descr: a string e.g. "variable type" describing the type being mapped
        :param model_to_cplex_map: a dictionary from model values to CPLEX values
        :param model_value: a value from modeling layer, to be translated
        :return:
        """
        res = model_to_cplex_map.get(model_value)
        if res is None:
            self.error_handler.fatal("Unexpected {0}: {1!s}", (type_descr, model_value))  # pragma: no cover
        return res

    def _vartype2cplextype(self, vartype_type):
        return self._convert_model_value_to_cplex_value("vartype", self._cplex_vartype_map, vartype_type)

    def _create_cpx_vartype_list(self, vartype, size):
        """ FIXME: Mega Hack here: setting an explicit continuous type
            will lead CPLEX to interpret the problem as a MIP
            got a 1017 error on the production sample for this...
        """
        vartype_type = type(vartype)
        if vartype_type == ContinuousVarType:
            return ''
        else:
            cpx_vartype = self._vartype2cplextype(vartype_type)
            if size == 1:
                return cpx_vartype
            else:
                return [cpx_vartype] * size

    def cttype2cplextype(self, cttype):
        return self._convert_model_value_to_cplex_value("constraint type", self._cplex_cttype_map, cttype)

    def objsense2cplexobjsense(self, sense):
        return self._convert_model_value_to_cplex_value("objective sense", self._cplex_objsense_map, sense)

    def _guess_index_range(self, names):
        # take the "real" last index from CPLEX, says 99
        actual_last_index = self.__cplex.variables.get_indices(names[-1])
        # if nb vars = 100, first index is 99-100 + 1= 0 cqfd
        guessed_first_index = actual_last_index - len(names) + 1
        index_range = range(guessed_first_index, actual_last_index + 1)
        return index_range

    # @profile
    def _internal_create_variables(self, allnames, alltypes, alllbs, allubs):
        # INTERNAL
        ret_add = self.__cplex.variables.add(names=allnames, types=alltypes, lb=alllbs, ub=allubs)

        if self._index_mode is CplexIndexMode.UseReturn:
            all_indices = ret_add
        elif self._index_mode is CplexIndexMode.Guess:
            # guess mode.
            all_indices = self._guess_index_range(allnames)
        else:
            # QueryByName mode, safe but verrry slow, no name hashing
            all_indices = self.__cplex.variables.get_indices(allnames)
        self._notify_var_index(all_indices[-1])
        return all_indices

    def create_one_variable(self, vartype, lb, ub, name):
        self._resync_if_needed()
        alltypes = self._create_cpx_vartype_list(vartype, size=1)
        allnames = [name] if name is not None else []
        alllbs = [lb] if lb is not None else []
        allubs = [ub] if ub is not None else []
        ret_val = self.__cplex.variables.add(names=allnames, types=alltypes, lb=alllbs, ub=allubs)
        index_mode = self._index_mode
        if index_mode is CplexIndexMode.UseReturn:
            idx = ret_val[0]
        elif index_mode is CplexIndexMode.Guess:
            idx = self._get_guessed_var_index()
        else:
            idx = self.__cplex.variables.get_indices(name)

        self._notify_var_index(idx)
        return idx

    # @profile
    def create_variables(self, keys, vartype, lbs, ubs, names):
        if keys:
            self._resync_if_needed()
            nb_vars = len(keys)
            alltypes = self._create_cpx_vartype_list(vartype, nb_vars)
            all_indices = self._internal_create_variables(names, alltypes, lbs, ubs)
            return all_indices
        else:
            return []

    def _apply_var_fn(self, var, args, setter_fn, getter_fn=None):
        cpxvars = self.__cplex.variables
        is_var_iterable = is_iterable(var)
        is_arg_iterable = is_iterable(args) and not isinstance(args, str)
        if not is_var_iterable and is_arg_iterable:
            self.error_handler.fatal("Single var requires a numeric argument, not iterable")
        if is_var_iterable:
            indices = [_v.get_index() for _v in var]
            list_args = args if is_arg_iterable else generate_constant(args, len(indices))
            setter_fn(cpxvars, izip(indices, list_args))
            if getter_fn:
                return getter_fn(cpxvars, indices)
            else:
                return None
        else:
            # newLb assumed to be NOT iterable at this point.
            var_idx = var.safe_index
            applied_arg = args
            setter_fn(cpxvars, var.safe_index, applied_arg)
            if getter_fn:
                return getter_fn(cpxvars, var_idx)
            else:
                return None

    _getset_map = {"lb": (cplex._internal._subinterfaces.VariablesInterface.set_lower_bounds,
                          cplex._internal._subinterfaces.VariablesInterface.get_lower_bounds),
                   "ub": (cplex._internal._subinterfaces.VariablesInterface.set_upper_bounds,
                          cplex._internal._subinterfaces.VariablesInterface.get_upper_bounds),
                   "name": (cplex._internal._subinterfaces.VariablesInterface.set_names,
                            cplex._internal._subinterfaces.VariablesInterface.get_names)}


    def rename_var(self, dvar, new_name):
        var_index = dvar.get_index()
        cpxvars = self.__cplex.variables
        cpxvars.set_names([(var_index, new_name)])


    def set_var_lb(self, var_lbs):
        self._resync_if_needed()
        self_var_lbs = self._var_lb_changed
        if isinstance(var_lbs, tuple):
            dv, lb = var_lbs
            self_var_lbs[dv] = lb
        else:
            for dv, lb in var_lbs:
                self_var_lbs[dv] = lb


    def set_var_ub(self, var_ubs):
        self._resync_if_needed()
        self_var_ubs = self._var_ub_changed
        if isinstance(var_ubs, tuple):
            dv, ub = var_ubs
            self_var_ubs[dv] = ub
        else:
            for dv, ub in var_ubs:
                self_var_ubs[dv] = ub

    def get_var_attribute(self, dvar, attr_name):
        self._resync_if_needed()
        getset_tuple = self._getset_map.get(attr_name)
        if not getset_tuple:
            self.error_handler.warning("unsupported attribute: {0}", attr_name)
        else:
            getter_fn = getset_tuple[1]
            return getter_fn(self.__cplex.variables, dvar.index)


    def get_solve_attribute(self, attr, index_seq):
        ''' Returns a sequence of attributes from the engine'''
        self._check_is_solved_ok()

        indices = list(index_seq)
        if attr == "slacks":
            all_attributes = self.__cplex.solution.get_linear_slacks(indices)
        elif attr == "duals":
            all_attributes = self.__cplex.solution.get_dual_values(indices)
        elif attr == "reduced_costs":
            all_attributes = self.__cplex.solution.get_reduced_costs(indices)
        else:
            self.error_handler.error('*unexpected attribute name: {0!s}', attr)
            return {}
        assert len(indices) == len(all_attributes)
        filtered_attr_map = {indices[i]: all_attributes[i] for i in range(len(indices)) if all_attributes[i]}
        return filtered_attr_map

    def _linexpr_to_cplex(self, expr):
        ''' convert a linear expression into a list of two lists (of the same length)
           1. a list of indices taken from previous post to CPLEX
           2. a list of coefs
           note the trick to iterate the expression only once:
           build a zipped list of tuples, then unzip it and return thr list of lists.
        '''
        all_indices_coefs = [(dv._index, float(k)) for dv, k in expr.iter_terms()]
        if all_indices_coefs:
            zipped = list(zip(*all_indices_coefs))
            return [zipped]
        else:
            return []

    # the returned list MUST be of size 2 otherwise the wrapper will crash.
    _trivial_linexpr = [[], []]

    # @profile
    def _binaryct_to_cplex(self, linear_ct):
        """
        Builds two lists, one for indices, one for coefs, of variables
        in lef_expr - right_expr.
        Keep the ordering: left variables first, then those from right_expr that are not in left_expr

        :param linear_ct: a linear constraint of type: expr1 OP expr2
        :return: either [] or a list of two lists indices and coefs
        """
        # noinspection PyPep8
        left_expr  = linear_ct.left_expr
        right_expr = linear_ct.right_expr
        if right_expr.is_constant():
            all_indices_coefs = [(dv._index, float(k)) for dv, k in left_expr.iter_terms()]

        elif left_expr.is_constant():
            all_indices_coefs = [(dv._index, -float(k)) for dv, k in right_expr.iter_terms()]
        else:
            all_indices_coefs = [(dv._index, float(k)) for dv, k in linear_ct._generate_net_coefs()]

        if all_indices_coefs:
            return list(zip(*all_indices_coefs))
        else:
            # the returned list MUST be of size 2 otherwise the wrapper will crash.
            return self._trivial_linexpr

    def __index_problem_stop_here(self):
        #  put a breakpoint here if index problems occur
        pass  # pragma: no cover

    # @profile
    def _make_cplex_linear_ct(self, cpx_lin_expr, ctype, rhs, name):
        """
        INTERNAL. fundamental way to post a linear ct to CPLEX.
        :param cpx_lin_expr:
        :param ctype:
        :param rhs:
        :param name:
        :return:
        """
        cpx_rhs = [float(rhs)]  # if not a float, cplex crashes baaaadly
        cpx_type = self.cttype2cplextype(ctype)
        cpxnames = [name] if name else []

        linearcts = self.__cplex.linear_constraints
        ret_add = linearcts.add(lin_expr=cpx_lin_expr, senses=cpx_type, rhs=cpx_rhs, names=cpxnames)

        self_index_mode = self._index_mode
        if self_index_mode is CplexIndexMode.Guess:
            cpx_ct_index = self._get_guessed_ct_index()
            self._notify_linear_constraint_index(cpx_ct_index)
        elif self_index_mode is CplexIndexMode.UseReturn:
            cpx_ct_index = ret_add[0]
        else:  # self_index_mode is CplexIndexMode.Query:
            if name:
                cpx_ct_index = linearcts.get_indices(name)
                if cpx_ct_index != self._last_used_ct_index + 1:
                    self.__index_problem_stop_here()
                self._notify_linear_constraint_index(cpx_ct_index)
            else:
                self.error_handler.trace("Cannot get index for anonymous constraint: {0!s}", ())
                # there is nothing to notify
                return -1

        return cpx_ct_index

    # @profile
    def create_binary_linear_constraint(self, binaryct):
        cpx_linexp1 = self._binaryct_to_cplex(binaryct)
        # wrap one more time
        cpx_linexp = [cpx_linexp1] if cpx_linexp1 else []
        # returns a number
        num_rhs = binaryct.rhs()
        return self._make_cplex_linear_ct(cpx_lin_expr=cpx_linexp,
                                          ctype=binaryct.type,
                                          rhs=num_rhs, name=binaryct.name)

    def create_block_linear_constraints(self, linct_seq):
        self._resync_if_needed()
        block_size = len(linct_seq)
        # need to force float() for numpy num types will crash CPLEX
        cpx_rhss = [float(ct.rhs()) for ct in linct_seq]
        cpx_senses = [self.cttype2cplextype(ct.type) for ct in linct_seq]
        cpx_names = [ct.get_name() for ct in linct_seq]
        cpx_linexprs = [self._binaryct_to_cplex(ct) for ct in linct_seq]

        cpx_linear = self.__cplex.linear_constraints
        ret_add = cpx_linear.add(lin_expr=cpx_linexprs, senses=cpx_senses, rhs=cpx_rhss, names=cpx_names)

        # -- diffent ways to query the range of new indices ---
        self_index_mode = self._index_mode
        if self_index_mode is CplexIndexMode.Guess:
            cpx_ct_indices = self._guessed_ct_index_range(block_size)
            self._notify_linear_constraint_index(cpx_ct_indices)
        elif self_index_mode is CplexIndexMode.UseReturn:
            cpx_ct_indices = ret_add  # returns a range
        else:  # self_index_mode is CplexIndexMode.Query:
            if cpx_names:
                cpx_ct_indices = cpx_linear.get_indices(cpx_names)
                self._notify_linear_constraint_index(cpx_ct_indices[-1])  # max is last?
            else:
                self.error_handler.trace("Cannot get index for anonymous constraint: {0!s}", ())
                # there is nothing to notify
                return -1

        return cpx_ct_indices

    def create_range_constraint(self, range_ct):
        """
        Post a range constraint to CPLEX
        :param range_ct:
        :return:
        """
        self._resync_if_needed()
        linearcts = self.__cplex.linear_constraints
        expr = range_ct.expr
        offset = expr.constant
        lhs = range_ct.lb
        rhs = range_ct.ub
        cpx_lin_expr = self._linexpr_to_cplex(expr)

        cpx_rhs = [rhs - offset]
        cpx_range_values = [lhs - rhs]  # should be negative ???
        cpx_type = CplexEngine.CPX_RANGE_SYMBOL
        ctname = range_ct.name
        cpxnames = [ctname] if ctname else []
        ret_add = linearcts.add(lin_expr=cpx_lin_expr,
                                senses=cpx_type, rhs=cpx_rhs,
                                range_values=cpx_range_values,
                                names=cpxnames)
        self_index_mode = self._index_mode
        if self_index_mode is CplexIndexMode.Guess:
            cpx_ct_index = self._get_guessed_ct_index()
            self._notify_linear_constraint_index(cpx_ct_index)
        elif self_index_mode is CplexIndexMode.Query:
            if ctname:
                cpx_ct_index = linearcts.get_indices(ctname)
                if cpx_ct_index != self._get_guessed_ct_index():
                    pass
                self._notify_linear_constraint_index(cpx_ct_index)
            else:
                cpx_ct_index = -1
        elif self_index_mode is CplexIndexMode.UseReturn:
            # first element of range
            cpx_ct_index = ret_add[0]
        else:
            cpx_ct_index = -1
            self.error_handler.fatal("Unexpected index mode: {0!s}", self_index_mode)
        return cpx_ct_index

    def create_indicator_constraint(self, indicator_ct):
        """
        Post an indicator ct to CPLEX
        :param indicator_ct:
        :return:
        """
        self._resync_if_needed()
        linear_ct = indicator_ct.linear_constraint
        ct_name = indicator_ct.name
        active_value = 1 - indicator_ct.active_value
        binary_var = indicator_ct.indicator_var
        binary_index = binary_var.safe_index

        # the linear ct is not posted to CPLEX,
        # but we need to convert it to linexpr
        cpx_linexpr = self._binaryct_to_cplex(linear_ct)
        rhs = linear_ct.rhs()
        cpx_name = ct_name or ''
        cpx_sense = self.cttype2cplextype(linear_ct.type)

        cpx_indicators = self.__cplex.indicator_constraints
        cpx_complemented = active_value
        ret_add = cpx_indicators.add(cpx_linexpr, cpx_sense, rhs, binary_index, cpx_complemented, cpx_name)
        if self._index_mode is CplexIndexMode.UseReturn:
            cpx_indicator_index = ret_add  # for indicators, CPLEX returns the index, not a range...
        else:
            cpx_indicator_index = cpx_indicators.get_indices(ct_name) if ct_name else -1
        return cpx_indicator_index

    def remove_constraint(self, ct):
        self._resync_if_needed()
        ct_index = ct.safe_index
        # we have a safe index
        self.__cplex.linear_constraints.delete(ct_index)
        self._notify_linear_constraint_deleted(ct_index)

    def set_objective(self, sense, expr):
        self._resync_if_needed()
        cpx_objective = self.__cplex.objective
        # --- set sense
        cpx_obj_sense = self.objsense2cplexobjsense(sense)
        cpx_objective.set_sense(cpx_obj_sense)
        # --- set coefficients
        if expr.is_quad_expr():
            cvq, cvv = expr.compute_separable_convexity()
            if cvv is not None and cvq < 0:
                self._model.warning("Quadratic objective is separable and non-convex, term: {0}{1!s}^2", cvq, cvv)

            self._set_quadratic_objective_coefs(cpx_objective, quad_expr=expr)
            self._set_linear_objective_coefs(cpx_objective, expr.linear_part)
        else:
            self._set_linear_objective_coefs(cpx_objective, linexpr=expr)

    def _set_linear_objective_coefs(self, cpx_objective, linexpr):
        # NOTE: convert to float as numpy doubles will crash cplex....
        index_coef_seq = [(dv._index, float(k)) for dv, k in linexpr.iter_terms()]
        if index_coef_seq:
            cpx_objective.set_linear(index_coef_seq)

    def _set_quadratic_objective_coefs(self, cpx_objective, quad_expr):
        for qv1, qv2, qk in quad_expr.iter_quad_triplets():
            fqk = float(qk)   # same as above: beware of numpy floats
            if qv1 is qv2:
                # diagonal term in the Q matrix
                qvi = qv1._index
                cpx_objective.set_quadratic_coefficients(qvi, qvi, 2 * fqk)
            else:
                # a triangular term in the Q matrix.
                cpx_objective.set_quadratic_coefficients(qv1._index, qv2._index, fqk)


    def clear_objective(self, expr):
        """
        Do not send an empty list otherwise a crash occurs.
        :param expr:
        :return:
        """
        self._resync_if_needed()
        if expr.is_constant():
            pass   # do nothing
        elif expr.is_quad_expr():
            # 1. reset quad part
            cpx_objective = self.__cplex.objective
            # -- set quad coeff to 0 for all quad variable pairs
            for qv1, qv2, _ in expr.iter_quad_triplets():
                cpx_objective.set_quadratic_coefficients(qv1, qv2, 0.)
            # 2. reset linear part
            self._clear_linear_objective(expr.linear_part)
        else:
            self._clear_linear_objective(expr)


    def _clear_linear_objective(self, linexpr):
        var_zero_seq = [(var._index, 0) for var in linexpr.iter_variables()]
        self.__cplex.objective.set_linear(var_zero_seq)
        # set_linear() does NOT reset the objective!
        # IndexError: tuple index out of range

    @staticmethod
    def status2string(cpx_status):
        ''' Converts a CPLEX integer status value to a string'''
        return _subinterfaces.SolutionInterface.status.__getitem__(cpx_status)

    def get_status_as_string(self):
        ''' Returns the solve status as a string.'''
        raw_status = self.__cplex.solution.get_status()
        return self.status2string(raw_status)

    @staticmethod
    def _is_relaxed_status_ok(status):
        # __CPLEX_RELAX_OK_STATUSES = frozenset([126, 16, 18, 14, 17, 19, 15])
        # list all status values for which there is a relaxed solution.
        # include QUAD values for the future, though for now the modeling doe snot support quads
        __CPLEX_RELAX_OK_STATUSES = {cpx_cst.CPXMIP_OPTIMAL_RELAXED_INF,
                                     cpx_cst.CPXMIP_OPTIMAL_RELAXED_SUM,
                                     cpx_cst.CPXMIP_OPTIMAL_RELAXED_QUAD,
                                     cpx_cst.CPXMIP_FEASIBLE_RELAXED_INF,
                                     cpx_cst.CPXMIP_FEASIBLE_RELAXED_QUAD,
                                     cpx_cst.CPXMIP_FEASIBLE_RELAXED_SUM,
                                     cpx_cst.CPX_STAT_FEASIBLE_RELAXED_SUM,
                                     cpx_cst.CPX_STAT_FEASIBLE_RELAXED_INF,
                                     cpx_cst.CPX_STAT_FEASIBLE_RELAXED_QUAD,
                                     cpx_cst.CPX_STAT_OPTIMAL_RELAXED_INF,
                                     cpx_cst.CPX_STAT_OPTIMAL_RELAXED_SUM
                                     }
        return status in __CPLEX_RELAX_OK_STATUSES

    __CPLEX_SOLVE_OK_STATUSES = {1,    # CPX_STAT_OPTIMAL
                                 6,    # CPX_STAT_NUM_BEST: solution exists but numerical issues
                                 24,   # CPX_STAT_FIRSTORDER: stting optimlaitytarget to 2
                                 101,  # CPXMIP_OPTIMAL
                                 102,  # CPXMIP_OPTIMAL_TOL
                                 104,  # CPXMIP_SOL_LIM
                                 105,  # CPXMPI_NODE_LIM_FEAS
                                 107,  # CPXMIP_TIME_LIM_FEAS
                                 109,  # CPXMIP_FAIL_FEAS : what is this ??
                                 111,  # CPXMIP_MEM_LIM_FEAS
                                 113,  # CPXMIP_ABORT_FEAS
                                 116,  # CPXMIP_FAIL_FEAS_NO_TREE : integer sol exists (????)
                                 129,  # CPXMIP_OPTIMAL_POPULATED
                                 130   # CPXMIP_OPTIMAL_POPULATED_TOL
                                 }

    @staticmethod
    def _is_solve_status_ok(status, all_ok_codes=__CPLEX_SOLVE_OK_STATUSES):
        # Converts a raw CPLEX status to a boolean
        return status in all_ok_codes

    def can_solve(self):
        return True

    @property
    def name(self):
        return 'cplex'

    def _sol_to_cpx(self, solution):
        l = [(dv.get_index(), val) for dv, val in solution.iter_var_values()]
        ul = zip(*l)
        # py3 zip() returns a generator, not a list, and CPLEX needs a list!
        return list(ul)

    def _sync_bounds(self, verbose=False):
        self_var_lbs = self._var_lb_changed
        if self_var_lbs:
            lb_vars, lb_values = zip(*iteritems(self_var_lbs))
            self._apply_var_fn(var=lb_vars, args=lb_values,
                               setter_fn=cplex._internal._subinterfaces.VariablesInterface.set_lower_bounds)
            if verbose:
                print("* synced {} var lower bounds".format(len(self._var_lb_changed)))

        self_var_ubs = self._var_ub_changed
        if self_var_ubs:
            ub_vars, ub_values = zip(*iteritems(self_var_ubs))
            self._apply_var_fn(var=ub_vars, args=ub_values,
                               setter_fn=cplex._internal._subinterfaces.VariablesInterface.set_upper_bounds)
            if verbose:
                print("* synced {} var upper bounds".format(len(self._var_ub_changed)))

    def solve(self, mdl, parameters=None):
        self._resync_if_needed()

        self._sync_bounds()

        cpx = self.__cplex
        # keep this line until RTC28217 is solved and closed !!! ----------------
        # see RTC 28217 item #18 for details
        cpx.get_problem_name()  # workaround from Ryan
        # -----------------------------------------------------------------------
        self._solve_count += 1
        solve_time_start = cpx.get_time()
        cpx_status = -1
        cpx_status_string = "*unknown*"
        cpx_miprelgap = None
        linear_nonzeros = -1
        nb_columns = 0
        cpx_probtype = None
        # print("--> starting CPLEX solve #", self.__solveCount)
        cpx_status_string = None
        try:
            # --- mipstart block ---
            mip_starts = mdl.mip_starts
            effort_level = cpx.MIP_starts.effort_level.repair
            for mp in mip_starts:
                if not isinstance(mp, SolveSolution):
                    self.error_handler.error("mip_starts expects Solution, got: {0!r} - ignored", (mp,))
                elif mp.check_as_mip_start():
                    # convert explicit values as tuples of (index, value)
                    cpx_sol = self._sol_to_cpx(mp)
                    # all_indices = [dv.get_index() for dv, _ in mp.iter_var_values()]
                    # all_values = [val for _, val in mp.iter_var_values()]
                    # cpx.MIP_starts.add([all_indices, all_values], effort_level)
                    cpx.MIP_starts.add(cpx_sol, effort_level)
                else:
                    pass
            # --- end of mipstart block ---

            linear_nonzeros = cpx.linear_constraints.get_num_nonzeros()
            nb_columns = cpx.variables.get_num()
            cpx_probtype = cpx.problem_type[cpx.get_problem_type()]
            cpx.solve()  # returns nothing in Python
            cpx_status = cpx.solution.get_status()
            cpx_status_string = self.__cplex.solution.get_status_string(cpx_status)

            solve_ok = self._is_solve_status_ok(cpx_status)
            if solve_ok:
                if cpx._is_MIP():
                    cpx_miprelgap = cpx.solution.MIP.get_mip_relative_gap()

        except cplex.exceptions.CplexSolverError as cpx_s:
            cpx_code = cpx_s.args[2]
            if 5002 == cpx_code:
                # we are in the notorious "non convex" case.
                # provide a meaningful status string for the solve details
                cpx_status = 5002  # famous error code...
                cpx_status_string = "QP with non-convex objective"
            self.error_handler.error("CPLEX Error: {0!s}, code={1}",
                                     (cpx_s.args[0], cpx_code))  # tuples required here...
            solve_ok = False

        except cplex.exceptions.CplexError as cpx_e:
            self.error_handler.error("CPLEX error: {0}", cpx_e.message)
            solve_ok = False

        finally:
            solve_time = cpx.get_time() - solve_time_start

            details = SolveDetails(solve_time,
                                   cpx_status, cpx_status_string,
                                   cpx_probtype,
                                   nb_columns, linear_nonzeros,
                                   cpx_miprelgap)
            self._last_solve_details = details


        # clear bound change requests
        self._var_lb_changed = {}
        self._var_ub_changed = {}

        self._last_solve_status = solve_ok
        new_solution = None
        if solve_ok:
            # compute correct objective including constant term
            obj_expr = mdl.objective_expr
            full_obj = cpx.solution.get_objective_value() + obj_expr.constant
            rounded_obj = mdl.round_objective_if_discrete(full_obj)
            # we need to build this list (maybe cache it?)
            all_var_indices = [dvar.index for dvar in mdl.iter_variables()]
            if all_var_indices:
                # do not query values on an empty model...
                all_var_values = self.__cplex.solution.get_values(all_var_indices)
                var_value_map = dict(zip(mdl.iter_variables(), all_var_values))
            else:
                var_value_map = {}

            new_solution = SolveSolution(mdl, obj=rounded_obj,
                                         var_value_map=var_value_map,
                                         engine_name=self.name,
                                         keep_zeros=False,
                                         rounding=True)
            new_solution._set_solve_status(self.get_solve_status())
            # cache attributes?
        else:
            mdl.notify_solve_failed()
        if cpx_status_string:
            mdl.error_handler.trace("CPLEX solve returns with status: {0}", (cpx_status_string,))
        return new_solution


    def _run_cpx_op_with_details(self, cpx_fn, *args):
        cpx = self.__cplex
        cpx_time_start = cpx.get_time()
        cpx_status = -1
        cpx_status_string = "*unknown*"
        cpx_miprelgap = None
        linear_nonzeros = -1
        nb_columns = 0
        cpx_probtype = None
        try:
            linear_nonzeros = cpx.linear_constraints.get_num_nonzeros()
            nb_columns = cpx.variables.get_num()
            cpx_fn(*args)
            cpx_status = cpx.solution.get_status()
            cpx_probtype = cpx.problem_type[cpx.get_problem_type()]
            cpx_status_string = self.__cplex.solution.get_status_string(cpx_status)
            solve_ok = self._is_solve_status_ok(cpx_status)
            if solve_ok:
                if cpx._is_MIP():
                    cpx_miprelgap = cpx.solution.MIP.get_mip_relative_gap()

        except cplex.exceptions.CplexSolverError as cpx_s:
            self.error_handler.error("CPLEX Error: {0!s}, code={1}",
                                     (cpx_s.args[0], cpx_s.args[2]))  # tuples required here...

        except cplex.exceptions.CplexError as cpx_e:
            self.error_handler.error("CPLEX error: {0}", cpx_e.message)

        finally:
            cpx_time = cpx.get_time() - cpx_time_start

        details = SolveDetails(cpx_time,
                               cpx_status, cpx_status_string,
                               cpx_probtype,
                               nb_columns, linear_nonzeros,
                               cpx_miprelgap)
        return details

    def _check_is_solved_ok(self):
        """
        INTERNAL: checks the engine has recently been solved ok.
        Either raise an exception or returns None.
        :return:
        """
        self._check_is_solved()
        self.error_handler.ensure(self._last_solve_status, "Last solve failed")

    def _check_is_solved(self):
        # INTERNAL: check the engine has been solved
        self.error_handler.ensure(self._solve_count > 0, "Model is not solved yet")

    def get_solve_details(self):
        # must be solved but not necessarily ok
        return self._last_solve_details

    def to_index(self, arg):
        if is_int(arg):
            return arg
        else:
            try:
                return arg.index
            except AttributeError:
                self.error_handler.fatal("Cannot extract indeex from: {0!s}", arg)


    def get_solutions(self, dvars):
        # self._check_is_solved_ok()
        if not dvars:
            return {}

        # arguments can be either indices or objects with "index" attribute, else crash...
        indices = [self.to_index(v) for v in dvars]
        all_values = self.__cplex.solution.get_values(indices)
        return dict(zip(indices, all_values))

    def solve_relaxed(self, mdl, relaxable_groups, optimize, limits, parameters=None):
        """ Runs feasopt with a set of relaxable cts and numerical  preferences.

        Args:
            mdl: the model being relaxed
            relaxable_groups:
            optimize: True if the model has a non-numeric objective
            limits: a tuple with limits on the execution of each pass.
            parameters: parameters to use instead of the model parameters

        Returns:
            a solution object, or None.
        """
        self._resync_if_needed()
        relax_gap, relax_max_nb_sol, relax_pass_time_limit = limits
        cplex_params = self.__cplex.parameters
        cplex_feasopt_mode_param = cplex_params.feasopt.mode
        # which forced mode for feasopt? switch this flag for testing
        # with 12.67.2 and nurse, INF is very slow...
        use_sum_or_nb = True
        if use_sum_or_nb:
            if optimize:
                new_mode = cplex_params.feasopt.mode.values.opt_sum
            else:
                new_mode = cplex_params.feasopt.mode.values.min_sum
        else:
            if optimize:
                new_mode = cplex_params.feasopt.mode.values.opt_inf
            else:
                new_mode = cplex_params.feasopt.mode.values.min_inf

        overwritten_params = {cplex_feasopt_mode_param: new_mode}
        if relax_gap > 0:
            overwritten_params[cplex_params.mip.tolerances.mipgap] = relax_gap
        if relax_max_nb_sol >= 1:
            overwritten_params[cplex_params.mip.limits.solutions] = relax_max_nb_sol
        if relax_pass_time_limit >= 1:  # no less than 1s.
            overwritten_params[cplex_params.timelimit] = relax_pass_time_limit

        class _CpxOvwerwriteParamsExecutionContext(object):
            # internal context manager to handle forcing parameters during relaxation.

            def __init__(self, cplex_to_overwrite, overwrite_param_dict):
                assert isinstance(overwrite_param_dict, dict)
                self._cplex = cplex_to_overwrite
                self._overwrite_param_dict = overwrite_param_dict
                # store current values
                self._saved_param_values = {p: p.get() for p in overwrite_param_dict}
                # for p, v in iteritems(overwrite_param_dict):
                #     print("-- overwriting parameter {0} with value: {1!s}".format(p._name, v))

            def __enter__(self):
                # force overwrite values.
                for p, v in iteritems(self._overwrite_param_dict):
                    p.set(v)
                # return the Cplex instance with the overwritten parameters.
                return self._cplex

            def __exit__(self, exc_type, exc_val, exc_tb):
                # whatever happened, restore saved parameter values.
                for param, old_value in iteritems(self._saved_param_values):
                    param.set(old_value)

        cpx_relax_groups = []
        with _CpxOvwerwriteParamsExecutionContext(self.__cplex, overwritten_params) as cpx:
            cpx_feasopt = cpx.feasopt
            all_indices = []
            all_cts = []
            feasopt_count = 1
            for (pref, group_cts) in relaxable_groups:
                self.error_handler.ensure(is_number(pref) and pref > 0,
                                          "Relaxation preference must be strict positive number", pref)
                self.error_handler.ensure(is_iterable(group_cts), "Non-iterable relaxable cts", group_cts)
                if not group_cts:
                    continue

                group_indices = [ct.safe_index for ct in group_cts]
                cpx_relax_groups.append(cpx_feasopt.linear_constraints(pref, group_indices))
                all_indices.append(group_indices)
                all_cts.append(group_cts)
                # at this stage, we have a list of groups
                # each group is itself a list
                # the first item is a number, th epreference
                # the second item is a list of indices.
                self._last_solve_details = self._run_cpx_op_with_details(cpx_feasopt, *cpx_relax_groups)
                feasopt_count += 1

            # feasopt state is restored by now
            cpx_solution = cpx.solution
            feas_status = cpx_solution.get_status()
            feas_ok = self._is_relaxed_status_ok(feas_status)
            feas_obj = cpx_solution.get_objective_value() if feas_ok else 0
            # feas_status_string = CplexEngine.status2string(feas_status)
            # print("* feasopt returns with status: %s, obj=%g"%(feas_status_string, feas_obj))
            return feas_ok, feas_obj

    def get_infeasibilities(self, cts):
        indices = [ct.index for ct in cts]
        # PCO: Daniel Junglas confirms using [] uses the last solution vector
        return self.__cplex.solution.infeasibility.linear_constraints([], indices)

    def dump(self, path):
        self._resync_if_needed()
        if path.find('.') > 0:
            self.__cplex.write(path)
        else:
            self.__cplex.write(path, filetype="lp")

    def get_problem_type(self):
        """ CPLEX wrapper returns an integer."""
        return self.__cplex.get_problem_type()

    def end(self):
        """ terminate the engine, cannot find this in the doc.
        """
        del self.__cplex
        self.__cplex = None

    # noinspection PyProtectedMember
    def is_mip(self):
        cpx = self.__cplex
        _all_mip_problems = frozenset({'MIP', 'MILP', 'fixedMILP', 'MIQP', 'fixedMIQP'})
        cpx_problem_type = cpx.problem_type[cpx.get_problem_type()]
        return cpx_problem_type in _all_mip_problems

    def connect_progress_listeners(self, progress_listener_list):
        if not progress_listener_list:
            self.error_handler.info("No progress listeners to connect")
        elif self.is_mip():
            ccb = self.__cplex.register_callback(ConnectListenersCallback)
            ccb.initialize(progress_listener_list)

    def set_parameter(self, parameter, value):
        # value check is up to the caller.
        try:
            self.__cplex._env.parameters._set(parameter.cpx_id, value)
        except CplexError as cpx_e:
            cpx_msg = str(cpx_e)
            if cpx_msg.startswith("Bad parameter identifier"):
                self.error_handler.warning("Parameter \"{0}\" is not recognized",  (parameter.qualified_name,))
            else:
                self.error_handler.error("Error setting parameter {0} to value {1}"
                                         .format(parameter.short_name, value))

    def set_parameter_block(self, parameters):
        if parameters:
            for param in parameters:
                self.set_parameter(param, param.current_value)

    def get_parameter(self, parameter):
        try:
            return self.__cplex._env.parameters._get(parameter.cpx_id)
        except CplexError:
            return parameter.default_value

    def get_solve_status(self):
        from docloud.status import JobSolveStatus
        # In this function we try to do the exact same mappings as the IloCplex C++ and Java classes.
        # However, this is not always possible since the C++ and Java implementations are not consistent
        # and sometimes they are even in error (see RTC-21923).
        cpx_status = self.__cplex.solution.get_status()
        if cpx_status in {cpx_cst.CPXMIP_ABORT_FEAS,
                          cpx_cst.CPXMIP_DETTIME_LIM_FEAS,
                          cpx_cst.CPXMIP_FAIL_FEAS,
                          cpx_cst.CPXMIP_FAIL_FEAS_NO_TREE,
                          cpx_cst.CPXMIP_MEM_LIM_FEAS,
                          cpx_cst.CPXMIP_NODE_LIM_FEAS,
                          cpx_cst.CPXMIP_TIME_LIM_FEAS
                          }:
            return JobSolveStatus.FEASIBLE_SOLUTION

        elif cpx_status in {cpx_cst.CPXMIP_ABORT_INFEAS,
                            cpx_cst.CPXMIP_DETTIME_LIM_INFEAS,
                            cpx_cst.CPXMIP_FAIL_INFEAS,
                            cpx_cst.CPXMIP_FAIL_INFEAS_NO_TREE,
                            cpx_cst.CPXMIP_MEM_LIM_INFEAS,
                            cpx_cst.CPXMIP_NODE_LIM_INFEAS,
                            cpx_cst.CPXMIP_TIME_LIM_INFEAS
                            }:
            # Hit a limit without a feasible solution: We don't know anything about the solution.
            return JobSolveStatus.UNKNOWN
        elif cpx_status in {cpx_cst.CPXMIP_OPTIMAL,
                            cpx_cst.CPXMIP_OPTIMAL_TOL}:
            return JobSolveStatus.OPTIMAL_SOLUTION
        elif cpx_status is cpx_cst.CPXMIP_SOL_LIM:
            #  return hasSolution(env, lp) ? JobSolveStatus.FEASIBLE_SOLUTION : JobSolveStatus.UNKNOWN;
            return JobSolveStatus.FEASIBLE_SOLUTION
        elif cpx_status is cpx_cst.CPXMIP_INForUNBD:
            return JobSolveStatus.INFEASIBLE_OR_UNBOUNDED_SOLUTION
        elif cpx_status in {cpx_cst.CPXMIP_UNBOUNDED,
                            cpx_cst.CPXMIP_ABORT_RELAXATION_UNBOUNDED}:
            return JobSolveStatus.UNBOUNDED_SOLUTION
        elif cpx_status is cpx_cst.CPXMIP_INFEASIBLE:  # proven infeasible
            return JobSolveStatus.INFEASIBLE_SOLUTION
        elif cpx_status is cpx_cst.CPXMIP_OPTIMAL_INFEAS:  # optimal with unscaled infeasibilities
            # DANIEL: What exactly do we return here? There is an optimal solution but that solution is
            # infeasible after unscaling.
            return JobSolveStatus.OPTIMAL_SOLUTION

        #  feasopt status values
        elif cpx_status in frozenset({
            cpx_cst.CPXMIP_ABORT_RELAXED,  # relaxed solution is available and can be queried
            cpx_cst.CPXMIP_FEASIBLE  # problem feasible after phase I and solution available
        }):
            return JobSolveStatus.FEASIBLE_SOLUTION
        elif cpx_status in {cpx_cst.CPXMIP_FEASIBLE_RELAXED_INF,
                            cpx_cst.CPXMIP_FEASIBLE_RELAXED_QUAD,
                            cpx_cst.CPXMIP_FEASIBLE_RELAXED_SUM
                            }:
            return JobSolveStatus.UNKNOWN
        elif cpx_status in {cpx_cst.CPXMIP_OPTIMAL_RELAXED_INF,
                            cpx_cst.CPXMIP_OPTIMAL_RELAXED_QUAD,
                            cpx_cst.CPXMIP_OPTIMAL_RELAXED_SUM
                            }:
            return JobSolveStatus.INFEASIBLE_SOLUTION

        # populate status values
        elif cpx_status in {cpx_cst.CPXMIP_OPTIMAL_POPULATED
                            # ,cpx_cst.CPXMIP_OPTIMAL_POPULATED_TO
                            }:
            return JobSolveStatus.OPTIMAL_SOLUTION
        elif cpx_status is cpx_cst.CPXMIP_POPULATESOL_LIM:
            # minimal value for CPX_PARAM_POPULATE_LIM is 1! So there must be a solution
            return JobSolveStatus.FEASIBLE_SOLUTION

        elif cpx_status is cpx_cst.CPX_STAT_OPTIMAL:
            return JobSolveStatus.OPTIMAL_SOLUTION

        elif cpx_status is cpx_cst.CPX_STAT_INFEASIBLE:
            return JobSolveStatus.INFEASIBLE_SOLUTION

        # cpx_cst.CPX_STAT_ABORT_USER:
        # cpx_cst.CPX_STAT_ABORT_DETTIME_LIM:
        # cpx_cst.CPX_STAT_ABORT_DUAL_OBJ_LIM:
        # cpx_cst.CPX_STAT_ABORT_IT_LIM:
        # cpx_cst.CPX_STAT_ABORT_PRIM_OBJ_LIM:
        # cpx_cst.CPX_STAT_ABORT_TIME_LIM:
        #   switch (primalDualFeasible(env, lp)) {
        #   case PRIMAL_FEASIBLE: return JobSolveStatus.FEASIBLE_SOLUTION
        #   case PRIMAL_DUAL_FEASIBLE: return JobSolveStatus.OPTIMAL_SOLUTION
        #   case DUAL_FEASIBLE: return JobSolveStatus.UNKNOWN;
        #   default: return JobSolveStatus.UNKNOWN;
        #   }
        #
        # cpx_cst.CPX_STAT_ABORT_OBJ_LIM:
        #   /** DANIEL: Our Java API returns ERROR here while the C++ API returns Feasible if primal feasible
        #    *         and Unknown otherwise. Since we don't have ERROR in IloSolveStatus we emulate the
        #    *         C++ behavior (this is more meaningful anyway). In the long run we should make sure
        #    *         all the APIs behave in the same way.
        #    */
        #   switch (primalDualFeasible(env, lp)) {
        #   case PRIMAL_FEASIBLE: return JobSolveStatus.FEASIBLE_SOLUTION
        #   case PRIMAL_DUAL_FEASIBLE: return JobSolveStatus.FEASIBLE_SOLUTION
        #   default: return JobSolveStatus.UNKNOWN;
        #   }
        #
        # cpx_cst.CPX_STAT_FIRSTORDER:
        #   // See IloCplexI::CplexToAlgorithmStatus()
        #   return primalFeasible(env, lp) ? JobSolveStatus.FEASIBLE_SOLUTION : JobSolveStatus.UNKNOWN;

        elif cpx_status is cpx_cst.CPX_STAT_CONFLICT_ABORT_CONTRADICTION:
            # Numerical trouble in conflict refiner.
            #  DANIEL: C++ and Java both return Error here although a conflict is
            #          available (but nor proven to be minimal). This looks like a bug
            #          since no exception is thrown there. In IloSolveStatus we don't
            #          have ERROR, so we return UNKNOWN instead. This is fine for now
            #          since we do not support the conflict refiner anyway.
            #
            return JobSolveStatus.UNKNOWN

        elif cpx_status in {

            cpx_cst.CPX_STAT_CONFLICT_ABORT_DETTIME_LIM,
            cpx_cst.CPX_STAT_CONFLICT_ABORT_IT_LIM,
            cpx_cst.CPX_STAT_CONFLICT_ABORT_MEM_LIM,
            cpx_cst.CPX_STAT_CONFLICT_ABORT_NODE_LIM,
            cpx_cst.CPX_STAT_CONFLICT_ABORT_OBJ_LIM,
            cpx_cst.CPX_STAT_CONFLICT_ABORT_TIME_LIM,
            cpx_cst.CPX_STAT_CONFLICT_ABORT_USER
        }:
            # /** DANIEL: C++ and Java return Error here. This is almost certainly wrong.
            # *         Docs say "a conflict is available but not minimal".
            #  *         This is particularly erroneous if no exception gets thrown.
            #  *         See RTC-21923.
            #  *         In IloSolveStatus we don't have ERROR, so we return UNKNOWN instead.
            #  *         This should not be a problem since right now we don't support the
            #  *         conflict refiner anyway.
            #  */
            return JobSolveStatus.UNKNOWN
        elif cpx_status is cpx_cst.CPX_STAT_CONFLICT_FEASIBLE:
            return JobSolveStatus.FEASIBLE_SOLUTION
        elif cpx_status is cpx_cst.CPX_STAT_CONFLICT_MINIMAL:
            return JobSolveStatus.INFEASIBLE_SOLUTION
        elif cpx_status in {cpx_cst.CPX_STAT_FEASIBLE_RELAXED_INF,
                            cpx_cst.CPX_STAT_FEASIBLE_RELAXED_QUAD,
                            cpx_cst.CPX_STAT_FEASIBLE_RELAXED_SUM,
                            }:
            return JobSolveStatus.UNKNOWN

        elif cpx_status is cpx_cst.CPX_STAT_FEASIBLE:
            return JobSolveStatus.FEASIBLE_SOLUTION
        elif cpx_status in {cpx_cst.CPX_STAT_OPTIMAL_RELAXED_INF,
                            cpx_cst.CPX_STAT_OPTIMAL_RELAXED_QUAD,
                            cpx_cst.CPX_STAT_OPTIMAL_RELAXED_SUM}:
            return JobSolveStatus.INFEASIBLE_SOLUTION

        elif cpx_status is cpx_cst.CPX_STAT_NUM_BEST:
            #  Solution available but not proved optimal (due to numeric difficulties)
            # assert(hasSolution(env, lp));
            return JobSolveStatus.UNKNOWN

        elif cpx_status is cpx_cst.CPX_STAT_OPTIMAL_INFEAS:  # infeasibilities after unscaling
            # assert(hasSolution(env, lp));
            return JobSolveStatus.OPTIMAL_SOLUTION

        elif cpx_status is cpx_cst.CPX_STAT_INForUNBD:  # Infeasible or unbounded in presolve.
            return JobSolveStatus.INFEASIBLE_OR_UNBOUNDED_SOLUTION
        elif cpx_status is cpx_cst.CPX_STAT_OPTIMAL_FACE_UNBOUNDED:
            #    unbounded optimal face (barrier only)
            # // CPX_STAT_OPTIMAL_FACE_UNBOUNDED is explicitly an error in Java and implicitly (fallthrough)
            # // an error in C++. So it should be fine to produce an error here as well.
            # // In IloSolveStatus we don't have ERROR, so we return UNKNOWN instead.
            # // In case of ERROR we should have seen a non-zero status anyway and the
            # // user should not care too much about the returned status.
            return JobSolveStatus.UNKNOWN
        elif cpx_status is cpx_cst.CPX_STAT_UNBOUNDED:
            # definitely unbounded
            return JobSolveStatus.UNBOUNDED_SOLUTION
        else:
            return JobSolveStatus.UNBOUNDED_SOLUTION

    def _resync_if_needed(self):
        if self._resync is _CplexSyncMode.OutOfSync:
            # print("-- resync cplex from model...")
            # send whole model to engine.
            try:
                self._resync = _CplexSyncMode.InResync
                self._model.resync()
            finally:
                self._resync = _CplexSyncMode.InSync


@contextmanager
def overload_cplex_parameter_values(cpx_engine, overload_dict):
    old_values = {p: p.get() for p in overload_dict}
    try:
        yield cpx_engine
    finally:
        # restore params
        for p, saved_value in iteritems(old_values):
            p.set(saved_value)


from docplex.mp.compat23 import copyreg
from docplex.mp.environment import Environment
from docplex.mp.engine import NoSolveEngine


def unpickle_cplex_engine(mdl, is_traced):
    #  INTERNAL
    unpicking_env = Environment()
    if unpicking_env.has_cplex:
        cplex_engine = CplexEngine(mdl)
        cplex_engine.notify_trace_output(sys.stdout if is_traced else None)  # what to do if file??
        # mark to be resync'ed
        cplex_engine._mark_as_out_of_sync()
        return cplex_engine
    else:
        return NoSolveEngine.make_from_model(mdl)

unpickle_cplex_engine.__safe_for_unpickling__ = True


def pickle_cplex_engine(cplex_engine):
    model = cplex_engine._model
    return unpickle_cplex_engine, (model, model.is_logged())

copyreg.pickle(CplexEngine, pickle_cplex_engine)
