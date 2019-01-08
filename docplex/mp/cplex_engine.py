# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

import sys

from collections import defaultdict

from docplex.mp.engine import DummyEngine
from docplex.mp.utils import is_iterable, generate_constant, DOcplexException, str_holo
from docplex.mp.compat23 import izip

from docplex.mp.vartype import ContinuousVarType
from docplex.mp.constants import ConflictStatus
from docplex.mp.constr import IndicatorConstraint, QuadraticConstraint, LinearConstraint
from docplex.mp.progress import ProgressData
from docplex.mp.solution import SolveSolution
from docplex.mp.sdetails import SolveDetails
from docplex.mp.conflict_refiner import TConflictConstraint, VarLbConstraintWrapper, VarUbConstraintWrapper
import cplex

from six import iteritems
from contextlib import contextmanager

import numbers
from enum import Enum
# CHECK THIS
# noinspection PyProtectedMember
from cplex._internal import _subinterfaces
from cplex.callbacks import MIPInfoCallback

# noinspection PyProtectedMember
import cplex._internal._constants as cpx_cst
from cplex.exceptions import CplexError, CplexSolverError

from docplex.mp.compat23 import fast_range

from docplex.mp.compat23 import copyreg
from docplex.mp.environment import Environment
from docplex.mp.engine import NoSolveEngine

# gendoc: ignore


class ConnectListenersCallback(MIPInfoCallback):
    RELATIVE_EPS = 1e-5
    ABS_EPS = 1e-4

    # noinspection PyAttributeOutsideInit
    def initialize(self, listeners):
        self.__listeners = listeners
        self.__pdata = ProgressData()
        self._start_time = -1
        self._start_dettime = -1
        # subset of listeners which listen to intermediate solutions.
        self.__solution_listeners = [l for l in listeners if l.requires_solution()]
        for l in listeners:
            l.connect_cb(self)

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

    _mode2string = {"query": Query, "guess": Guess, "return": UseReturn}

    @staticmethod
    def parse(text, default_mode):
        if not text:
            return default_mode
        else:
            return CplexIndexMode._mode2string.get(text.lower(), default_mode)


class _CplexOverwriteParametersCtx(object):
    # internal context manager to handle forcing parameters during relaxation.

    def __init__(self, cplex_to_overwrite, overwrite_param_dict):
        assert isinstance(overwrite_param_dict, dict)
        self._cplex = cplex_to_overwrite
        self._overwrite_param_dict = overwrite_param_dict
        # store current values
        cplex_params = self._cplex._env.parameters
        self._saved_param_values = {p.cpx_id: cplex_params._get(p.cpx_id) for p in overwrite_param_dict}

    def __enter__(self):
        # force overwrite values.
        cplex_params = self._cplex._env.parameters
        for p, v in iteritems(self._overwrite_param_dict):
            cplex_params._set(p.cpx_id, v)
        # return the Cplex instance with the overwritten parameters.
        return self._cplex

    # noinspection PyUnusedLocal
    def __exit__(self, exc_type, exc_val, exc_tb):
        # whatever happened, restore saved parameter values.
        cplex_params = self._cplex._env.parameters
        for pid, saved_v in iteritems(self._saved_param_values):
            cplex_params._set(pid, saved_v)


class IndexScope(object):
    def __init__(self, name):
        self._name = name
        self._index = -1

    def clear(self):
        self._index = -1

    def new_index(self):
        self._index += 1
        return self._index

    def new_index_range(self, size):
        first = self._index + 1
        last = first + size
        self._index += size
        return fast_range(first, last)

    def notify_deleted(self, deleted_index):
        if deleted_index >= 0:
            self._index -= 1

    def notify_deleted_block(self, deleted_indices):
        self._index -= len(deleted_indices)

    def __str__(self):  # pragma: no cover
        return 'IndexScope({0}}[{1}]'.format(self._name, self._index)


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
        self._lincts_scope = IndexScope(name='lincts')
        self._indcst_scope = IndexScope(name='indicators')
        self._quadcst_scope = IndexScope(name='quadcts')
        self._vars_scope = IndexScope(name='vars')

    def _mark_as_out_of_sync(self):
        self._resync = _CplexSyncMode.OutOfSync

    def _allocate_one_index(self, ret_value, scope):
        self_index_mode = self._index_mode
        if self_index_mode is CplexIndexMode.UseReturn:
            return ret_value
        elif self_index_mode is CplexIndexMode.Guess:
            return scope.new_index()
        else:  # pragma: no cover
            raise ValueError

    def _allocate_range_index(self, size, ret_value, scope):
        self_index_mode = self._index_mode
        if self_index_mode is CplexIndexMode.UseReturn:
            return ret_value
        elif self_index_mode is CplexIndexMode.Guess:
            return scope.new_index_range(size)
        else:  # pragma: no cover
            raise ValueError

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

    def get_var_index(self, dvar):  # pragma: no cover
        self._resync_if_needed()
        dvar_name = dvar.name
        if not dvar_name:
            self.error_handler.fatal("cannot query index for anonymous object: {0!s}", (dvar,))
        else:
            return self.__cplex.variables.get_indices(dvar_name)

    def get_ct_index(self, ct):  # pragma: no cover
        self._resync_if_needed()
        ctname = ct.name
        if not ctname:
            self.error_handler.fatal("cannot query index for anonymous constraint: {0!s}", (ct,))
        self_cplex = self.__cplex
        if ct.is_linear():
            return self_cplex.linear_constraints.get_indices(ctname)
        elif isinstance(ct, IndicatorConstraint):
            return self_cplex.indicator_constraints.get_indices(ctname)
        elif isinstance(ct, QuadraticConstraint):
            return self_cplex.quadratic_constraints.get_indices(ctname)

        else:
            self.error_handler.fatal("unrecognized constraint to query index: {0!s}", ct)

    def _sync_constraint_indices(self, ct_iter):
        for ct in ct_iter:
            if ct.name is None:
                # TODO: for anonymous constraints, check identity between cplex constraint with same index.
                continue
            else:
                model_index = ct.get_index()
                if model_index >= 0:
                    cpx_index = self.get_ct_index(ct)
                    if model_index != cpx_index:  # pragma: no cover
                        self._model.error("indices differ, obj: {0!s}, docplex={1}, CPLEX={2}", ct, model_index,
                                          cpx_index)

    def _sync_var_indices(self, iter_dvars):
        for dvar in iter_dvars:
            # assuming dvar has a name
            model_index = dvar.get_index()
            cpx_index = self.get_var_index(dvar)
            if model_index != cpx_index:  # pragma : nocover
                self._model.error("indices differ, obj: {0!s}, docplex={1}, CPLEX={2}", dvar, model_index,
                                  cpx_index)

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

    def _create_cpx_vartype_list(self, vartype, size):
        """ FIXME: Mega Hack here: setting an explicit continuous type
            will lead CPLEX to interpret the problem as a MIP
            got a 1017 error on the production sample for this...
        """
        vartype_type = type(vartype)
        if vartype_type == ContinuousVarType:
            return ''
        else:
            cpx_vartype = vartype.get_cplex_typecode()
            if size == 1:
                return cpx_vartype
            else:
                return [cpx_vartype] * size

    def create_one_variable(self, vartype, lb, ub, name):
        self._resync_if_needed()
        alltypes = self._create_cpx_vartype_list(vartype, size=1)
        allnames = [name] if name is not None else []
        alllbs = [lb] if lb is not None else []
        allubs = [ub] if ub is not None else []
        ret_val = self.__cplex.variables.add(names=allnames, types=alltypes, lb=alllbs, ub=allubs)
        return self._allocate_one_index(ret_value=ret_val, scope=self._vars_scope)

    def create_variables(self, keys, vartype, lbs, ubs, names):
        self._resync_if_needed()
        nb_vars = len(keys)
        alltypes = self._create_cpx_vartype_list(vartype, nb_vars)
        ret_add = self.__cplex.variables.add(names=names, types=alltypes, lb=lbs, ub=ubs)
        return self._allocate_range_index(size=nb_vars, ret_value=ret_add, scope=self._vars_scope)


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

    def set_var_type(self, dvar, newtype):  # pragma: no cover
        var_index = dvar.get_index()
        cpxvars = self.__cplex.variables
        cpx_newtype = newtype.get_cplex_typecode()
        cpxvars.set_types([(var_index, cpx_newtype)])

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
        left_expr = linear_ct.left_expr
        right_expr = linear_ct.right_expr
        if right_expr.is_constant():
            all_indices_coefs = [(dv._index, float(k)) for dv, k in left_expr.iter_terms()]

        elif left_expr.is_constant():
            all_indices_coefs = [(dv._index, -float(k)) for dv, k in right_expr.iter_terms()]
        else:
            all_indices_coefs = [(dv._index, float(k)) for dv, k in linear_ct._generate_net_linear_coefs()]

        # all_indices_coefs is a list of  (index, coef) 2-tuples
        if all_indices_coefs:
            # CPLEX requires two lists: one for indices, one for coefs
            # we use zip to unzip the tuples
            return list(izip(*all_indices_coefs))
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
        cpx_type = ctype._cplex_code
        cpxnames = [name] if name else []

        linearcts = self.__cplex.linear_constraints
        ret_add = linearcts.add(lin_expr=cpx_lin_expr, senses=cpx_type, rhs=cpx_rhs, names=cpxnames)
        return self._allocate_one_index(ret_value=ret_add, scope=self._lincts_scope)

    # @profile
    def create_binary_linear_constraint(self, binaryct):
        self._resync_if_needed()
        cpx_linexp1 = self._binaryct_to_cplex(binaryct)
        # wrap one more time
        cpx_linexp = [cpx_linexp1] if cpx_linexp1 else []
        # returns a number
        num_rhs = binaryct.rhs()
        return self._make_cplex_linear_ct(cpx_lin_expr=cpx_linexp,
                                          ctype=binaryct.type,
                                          rhs=num_rhs, name=binaryct.get_name())

    def create_block_linear_constraints(self, linct_seq):
        self._resync_if_needed()
        block_size = len(linct_seq)
        # need to force float() for numpy num types will crash CPLEX
        # noinspection PyPep8
        cpx_rhss = [float(ct.rhs()) for ct in linct_seq]
        cpx_senses = [ct.type._cplex_code for ct in linct_seq]
        cpx_names = [ct._get_safe_name() for ct in linct_seq]
        cpx_linexprs = [self._binaryct_to_cplex(ct) for ct in linct_seq]

        cpx_linear = self.__cplex.linear_constraints
        ret_add = cpx_linear.add(lin_expr=cpx_linexprs, senses=cpx_senses, rhs=cpx_rhss, names=cpx_names)

        return self._allocate_range_index(size=block_size, ret_value=ret_add, scope=self._lincts_scope)

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
        return self._allocate_one_index(ret_value=ret_add, scope=self._lincts_scope)

    def create_indicator_constraint(self, indicator_ct):
        """
        Post an indicator ct to CPLEX
        :param indicator_ct:
        :return:
        """
        self._resync_if_needed()
        linear_ct = indicator_ct.linear_constraint
        ct_name = indicator_ct.get_name()
        active_value = 1 - indicator_ct._active_value
        binary_var = indicator_ct.indicator_var
        binary_index = binary_var.get_index()

        # the linear ct is not posted to CPLEX,
        # but we need to convert it to linexpr
        cpx_linexpr = self._binaryct_to_cplex(linear_ct)
        rhs = linear_ct.rhs()
        cpx_name = ct_name or ''
        cpx_sense = linear_ct.type._cplex_code

        cpx_indicators = self.__cplex.indicator_constraints
        cpx_complemented = active_value
        ret_add = cpx_indicators.add(cpx_linexpr, cpx_sense, rhs, binary_index, cpx_complemented, cpx_name)
        return self._allocate_one_index(ret_value=ret_add, scope=self._indcst_scope)

    def create_quadratic_constraint(self, qct):
        self._resync_if_needed()
        # ---
        self_cplex = self.__cplex
        float_rhs = float(qct.rhs())  # if not a float, cplex crashes baaaadly
        cpx_sense = qct.type._cplex_code
        # see RTC-31772, None is accepted from 12.6.3.R0 onward. use get_name() when dropping compat for 12.6.2
        qctname = qct._get_safe_name()

        # linear part
        net_linears = [(lv._index, float(lk)) for lv, lk in qct.iter_net_linear_coefs()]
        list_linears = list(izip(*net_linears))
        if not list_linears:
            list_linears = [(0,), (0.0,)]  # always non empty
        # build a list of three lists: [qv1.index], [qv2.index], [qk..]

        net_quad_triplets = [(qvp[0]._index, qvp[1]._index, float(qk)) for qvp, qk in qct.iter_net_quads()]
        if net_quad_triplets:
            list_quad_triplets = list(izip(*net_quad_triplets))
            ret_add = self_cplex.quadratic_constraints.add(lin_expr=list_linears,
                                                           quad_expr=list_quad_triplets,
                                                           sense=cpx_sense,
                                                           rhs=float_rhs,
                                                           name=qctname)
            return self._allocate_one_index(ret_value=ret_add, scope=self._quadcst_scope)
        else:
            # actually a linear constraint
            return self._make_cplex_linear_ct(cpx_lin_expr=[list_linears], ctype=qct.type, rhs=float_rhs, name=qctname)

    def remove_constraint(self, ct):
        self._resync_if_needed()
        doomed_index = ct.safe_index
        # we have a safe index
        if isinstance(ct, QuadraticConstraint):
            self.__cplex.quadratic_constraints.delete(doomed_index)
            self._quadcst_scope.notify_deleted(doomed_index)
        elif isinstance(ct, IndicatorConstraint):
            self.__cplex.indicator_constraints.delete(doomed_index)
            self._indcst_scope.notify_deleted(doomed_index)
        elif isinstance(ct, LinearConstraint):
            self.__cplex.linear_constraints.delete(doomed_index)
            self._lincts_scope.notify_deleted(doomed_index)
        else:
            raise TypeError

    def remove_constraints(self, cts):
        self._resync_if_needed()
        if cts is None:
            self.__cplex.linear_constraints.delete()
            self._lincts_scope.clear()
            self.__cplex.quadratic_constraints.delete()
            self._quadcst_scope.clear()
            self.__cplex.indicator_constraints.delete()
            self._indcst_scope.clear()
        else:
            doomed_linears = [c.safe_index for c in cts if c.is_linear()]
            doomed_quadcts = [c.safe_index for c in cts if isinstance(c, QuadraticConstraint)]
            dooomed_indcts = [c.safe_index for c in cts if isinstance(c, IndicatorConstraint)]
            if doomed_linears:
                self.__cplex.linear_constraints.delete(doomed_linears)
                self._lincts_scope.notify_deleted_block(doomed_linears)
            if doomed_quadcts:
                self.__cplex.quadratic_constraints.delete(doomed_quadcts)
                self._quadcst_scope.notify_deleted_block(doomed_quadcts)
            if dooomed_indcts:
                self.__cplex.indicator_constraints.delete(dooomed_indcts)
                self._indcst_scope.notify_deleted(dooomed_indcts)

    def set_objective(self, sense, expr):
        self._resync_if_needed()
        # old objective
        old_objective = self._model.objective_expr
        self._clear_objective(old_objective)
        # --
        cpx_objective = self.__cplex.objective
        # --- set sense
        cpx_obj_sense = 1 if sense.is_minimize() else -1
        cpx_objective.set_sense(cpx_obj_sense)
        # --- set offset
        cpx_objective.set_offset(expr.get_constant())
        # --- set coefficients
        if expr.is_quad_expr():
            # cvq, cvv = expr.compute_separable_convexity()
            # if cvv is not None and cvq < 0:
            #     self._model.warning(
            #         "Objective is separable with negative coefficients, therefore non-convex. See term(s): {0}{1!s}^2",
            #         cvq, cvv)

            self._set_quadratic_objective_coefs(cpx_objective, quad_expr=expr)
            self._set_linear_objective_coefs(cpx_objective, expr.linear_part)
        else:
            self._set_linear_objective_coefs(cpx_objective, linexpr=expr)

    def _set_linear_objective_coefs(self, cpx_objective, linexpr):
        # NOTE: convert to float as numpy doubles will crash cplex....
        index_coef_seq = [(dv._index, float(k)) for dv, k in linexpr.iter_terms()]
        if index_coef_seq:
            # if list is empty, cplex will crash.
            cpx_objective.set_linear(index_coef_seq)

    def _set_quadratic_objective_coefs(self, cpx_objective, quad_expr):
        quad_obj_triplets = [(qv1._index, qv2._index, 2 * qk if qv1 is qv2 else qk) for qv1, qv2, qk in quad_expr.iter_quad_triplets()]
        if quad_obj_triplets:
            # if list is empty, cplex will crash.
            cpx_objective.set_quadratic_coefficients(quad_obj_triplets)

    def _clear_objective(self, expr):
        """
        Do not send an empty list otherwise a crash occurs.
        :param expr:
        :return:
        """
        self._resync_if_needed()
        if expr.is_constant():
            pass  # do nothing
        elif expr.is_quad_expr():
            # 1. reset quad part
            cpx_objective = self.__cplex.objective
            # -- set quad coeff to 0 for all quad variable pairs
            quad_reset_triplets = [(qvp.first._index, qvp.second._index, 0) for qvp, qk in expr.iter_quads()]
            if quad_reset_triplets:
                cpx_objective.set_quadratic_coefficients(quad_reset_triplets)
            # 2. reset linear part
            self._clear_linear_objective(expr.linear_part)
        else:
            self._clear_linear_objective(expr)

    def _clear_linear_objective(self, linexpr):
        if not linexpr.is_constant():
            var_zero_seq = [(var._index, 0) for var in linexpr.iter_variables()]
            self.__cplex.objective.set_linear(var_zero_seq)
            # set_linear() does NOT reset the objective!
            # IndexError: tuple index out of range

    @staticmethod
    def status2string(cpx_status):
        ''' Converts a CPLEX integer status value to a string'''
        return _subinterfaces.SolutionInterface.status.__getitem__(cpx_status)

    __CPLEX_SOLVE_OK_STATUSES = {1,  # CPX_STAT_OPTIMAL
                                 6,  # CPX_STAT_NUM_BEST: solution exists but numerical issues
                                 24,  # CPX_STAT_FIRSTORDER: stting optimlaitytarget to 2
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
                                 130  # CPXMIP_OPTIMAL_POPULATED_TOL
                                 }

    def _is_solve_status_ok(self, status, all_ok_codes=__CPLEX_SOLVE_OK_STATUSES):
        # Converts a raw CPLEX status to a boolean
        return status in all_ok_codes

    __CPLEX_RELAX_OK_STATUSES = frozenset({cpx_cst.CPX_STAT_FEASIBLE,
                                           cpx_cst.CPXMIP_OPTIMAL_RELAXED_INF,
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
                                           })

    def _is_relaxed_status_ok(self, status):
        # list all status values for which there is a relaxed solution.
        # also consider solve statuses in case  the model is indeed feasible
        return status in CplexEngine.__CPLEX_RELAX_OK_STATUSES or self._is_solve_status_ok(status)

    def can_solve(self):
        return True

    @property
    def name(self):
        return 'cplex'

    def _location(self):
        # INTERNAL
        return 'cplex_local'

    def _sol_to_cpx(self, model, mipstart):
        l = mipstart._to_tuple_list(model)
        ul = zip(*l)
        # py3 zip() returns a generator, not a list, and CPLEX needs a list!
        return list(ul)

    def _sync_var_bounds(self, verbose=False):
        self_var_lbs = self._var_lb_changed
        if self_var_lbs:
            lb_vars, lb_values = zip(*iteritems(self_var_lbs))
            self._apply_var_fn(var=lb_vars, args=lb_values,
                               setter_fn=cplex._internal._subinterfaces.VariablesInterface.set_lower_bounds)
            if verbose:  # pragma: no cover
                print("* synced {} var lower bounds".format(len(self._var_lb_changed)))

        self_var_ubs = self._var_ub_changed
        if self_var_ubs:
            ub_vars, ub_values = zip(*iteritems(self_var_ubs))
            self._apply_var_fn(var=ub_vars, args=ub_values,
                               setter_fn=cplex._internal._subinterfaces.VariablesInterface.set_upper_bounds)
            if verbose:  # pragma: no cover
                print("* synced {} var upper bounds".format(len(self._var_ub_changed)))

    def _apply_sos(self, mdl):
        # INTERNAL
        cpx_sos = self.__cplex.SOS
        # start by deleting all SOS: du passe faisons table rase....
        cpx_sos.delete()
        for sos_set in mdl.iter_sos():
            cpx_sos_type = sos_set.sos_type._cpx_sos_type()
            indices = [dv.index for dv in sos_set.iter_variables()]
            weights = sos_set.get_ranks()
            # do NOT pass None to cplex/swig here --> crash
            cpx_sos_name = sos_set._get_safe_name()
            # call cplex...
            sos_index = cpx_sos.add(type=cpx_sos_type, SOS=cplex.SparsePair(ind=indices, val=weights),
                                    name=cpx_sos_name)

    def _format_cplex_message(self, cpx_msg):
        if 'CPLEX' not in cpx_msg:
            cpx_msg = 'CPLEX: %s' % cpx_msg
        return cpx_msg.rstrip(' .\n')

    def clean_before_solve(self):
        # INTERNAL
        # delete all infos that were left by the previous solve
        self.__cplex.MIP_starts.delete()

    def solve(self, mdl, parameters=None):
        self._resync_if_needed()

        cpx = self.__cplex
        # keep this line until RTC28217 is solved and closed !!! ----------------
        # see RTC 28217 item #18 for details
        cpx.get_problem_name()  # workaround from Ryan
        # -----------------------------------------------------------------------
        self._solve_count += 1
        solve_time_start = cpx.get_time()
        cpx_status = -1
        cpx_miprelgap = None
        cpx_bestbound = None
        linear_nonzeros = -1
        nb_columns = 0
        cpx_probtype = None
        # print("--> starting CPLEX solve #", self.__solveCount)
        cpx_status_string = None
        try:
            # keep this in the protected  block...
            self._sync_var_bounds()
            self._apply_sos(mdl)

            # --- mipstart block ---
            mip_starts = mdl.mip_starts
            effort_level = cpx.MIP_starts.effort_level.repair
            for mp in mip_starts:
                if not isinstance(mp, SolveSolution):
                    mdl.fatal("mip_starts expects Solution, got: {0!r} - ignored", mp)
                cpx_sol = self._sol_to_cpx(mdl, mp)
                cpx.MIP_starts.add(cpx_sol, effort_level)

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
                    cpx_bestbound = cpx.solution.MIP.get_best_objective()

        except cplex.exceptions.CplexSolverError as cpx_s:  # pragma: no cover
            cpx_code = cpx_s.args[2]
            if 5002 == cpx_code:
                # we are in the notorious "non convex" case.
                # provide a meaningful status string for the solve details
                cpx_status = 5002  # famous error code...

                if self._model.has_quadratic_constraint():
                    cpx_status_string = "Non-convex QCP"
                    self._model.error('Model is non-convex')
                else:
                    cpx_status_string = "QP with non-convex objective"
                    self._model.error('Model has non-convex objective: {0!s}', str_holo(self._model.objective_expr, 60))
            solve_ok = False

        except cplex.exceptions.CplexError as cpx_e:  # pragma: no cover
            self.error_handler.error("CPLEX error: {0}", self._format_cplex_message(cpx_e.message))
            solve_ok = False

        except Exception as pe:  # pragma: no cover
            solve_ok = False
            self.error_handler.error('Internal error in CPLEX solve: {0!s}'.format(pe))

        finally:
            solve_time = cpx.get_time() - solve_time_start

            details = SolveDetails(solve_time,
                                   cpx_status, cpx_status_string,
                                   cpx_probtype,
                                   nb_columns, linear_nonzeros,
                                   cpx_miprelgap, cpx_bestbound)
            self._last_solve_details = details

        # clear bound change requests
        self._var_lb_changed = {}
        self._var_ub_changed = {}

        self._last_solve_status = solve_ok
        new_solution = None
        if solve_ok:
            new_solution = self._make_solution(mdl, self.get_solve_status())
            # cache attributes?
        else:
            mdl.notify_solve_failed()
        if cpx_status_string:
            mdl.error_handler.trace("CPLEX solve returns with status: {0}", (cpx_status_string,))
        return new_solution

    def _make_solution(self, mdl, solve_status):
        cpx = self.__cplex
        full_obj = cpx.solution.get_objective_value()
        rounded_obj = mdl.round_objective_if_discrete(full_obj)

        if mdl.number_of_variables > 0:
            all_var_indices = [dvar.get_index() for dvar in mdl.iter_variables()]
            # do not query values on an empty model...
            all_var_values = cpx.solution.get_values(all_var_indices)
            var_value_map = dict(izip(mdl.iter_variables(), all_var_values))
        else:
            var_value_map = {}

        solution = SolveSolution.make_engine_solution(model=mdl,
                                                      var_value_map=var_value_map,
                                                      obj=rounded_obj,
                                                      location=self._location(),
                                                      solve_status=solve_status)
        return solution

    def _run_cpx_op_with_details(self, cpx_fn, *args):
        cpx = self.__cplex
        cpx_time_start = cpx.get_time()
        cpx_status = -1
        cpx_status_string = "*unknown*"
        cpx_miprelgap = None
        cpx_bestbound = None
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
                    cpx_bestbound = cpx.solution.MIP.get_best_objective()

        except cplex.exceptions.CplexSolverError as cpx_s:  # pragma: no cover
            self.error_handler.error("CPLEX Error: {0!s}, code={1}",
                                     (cpx_s.args[0], cpx_s.args[2]))  # tuples required here...

        except cplex.exceptions.CplexError as cpx_e:  # pragma: no cover
            self.error_handler.error("CPLEX error: {0}", cpx_e.message)

        finally:
            cpx_time = cpx.get_time() - cpx_time_start

        details = SolveDetails(cpx_time,
                               cpx_status, cpx_status_string,
                               cpx_probtype,
                               nb_columns, linear_nonzeros,
                               cpx_miprelgap,
                               cpx_bestbound)
        return details

    def _check_is_solved_ok(self):
        """
        INTERNAL: checks the engine has recently been solved ok.
        Either raise an exception or returns None.
        :return:
        """
        if 0 == self._solve_count:
            self._model.fatal("Model {0} is not solved yet", self._model.name)
        if not self._last_solve_status:
            self._model.fatal("Last solve failed")

    def get_solve_details(self):
        # must be solved but not necessarily ok
        return self._last_solve_details

    def _make_groups(self, relaxable_groups):
        cpx_feasopt = self.__cplex.feasopt
        all_groups = []
        for (pref, group_cts) in relaxable_groups:
            if pref > 0 and group_cts:
                linears = []
                quads = []
                inds = []
                for ct in group_cts:
                    ctindex = ct.index
                    if ct.is_linear():
                        linears.append(ctindex)
                    elif isinstance(ct, IndicatorConstraint):
                        inds.append(ctindex)
                    elif isinstance(ct, QuadraticConstraint):
                        quads.append(ctindex)
                    else:
                        self.error_handler.error('cannot relax this: {0!s}'.format(ct))

                if linears:
                    all_groups.append(cpx_feasopt.linear_constraints(pref, linears))
                if quads:
                    all_groups.append(cpx_feasopt.quadratic_constraints(pref, quads))
                if inds:
                    all_groups.append(cpx_feasopt.indicator_constraints(pref, inds))
        return all_groups

    ct_linear = cplex._internal._subinterfaces.FeasoptConstraintType.linear
    ct_quadratic = cplex._internal._subinterfaces.FeasoptConstraintType.quadratic
    ct_indicator = cplex._internal._subinterfaces.FeasoptConstraintType.indicator

    infeasibility_resolver_map = {ct_linear: cplex.Cplex.solution.infeasibility.linear_constraints,
                                  ct_quadratic: cplex.Cplex.solution.infeasibility.quadratic_constraints,
                                  ct_indicator: cplex.Cplex.solution.infeasibility.indicator_constraints
                                  }

    _scope_resolver_map = {ct_linear: lambda m: m._linct_scope,
                           ct_quadratic: lambda m: m._quadct_scope,
                           ct_indicator: lambda m: m._indct_scope
                           }

    def _decode_infeasibilities(self, cpx, model, cpx_relax_groups, model_scope_resolver=_scope_resolver_map):
        resolver_map = {self.ct_linear: cpx.solution.infeasibility.linear_constraints,
                        self.ct_quadratic: cpx.solution.infeasibility.quadratic_constraints,
                        self.ct_indicator: cpx.solution.infeasibility.indicator_constraints
                        }
        cpx_sol_values = cpx.solution.get_values()
        cts_by_type = defaultdict(list)
        # split and group indices by type (cplex groups are ugly!!!)
        for g in cpx_relax_groups:
            # gp is a list of tuples (pref, ctype, index)
            for t in g._gp:
                ctype, ct_index = t[1][0]
                cts_by_type[ctype].append(ct_index)

        infeas_map = {}
        for ctype, indices in iteritems(cts_by_type):
            if indices:
                resolver_fn = resolver_map[ctype]
                ctype_infeas = resolver_fn(cpx_sol_values, indices)
                mscope = model_scope_resolver[ctype](model)
                assert mscope
                for ct_index, ct_infeas in izip(indices, ctype_infeas):
                    ct = mscope.get_object_by_index(ct_index)
                    if ct is not None:
                        infeas_map[ct] = ct_infeas
        return infeas_map

    def solve_relaxed(self, mdl, prio_name, relaxable_groups, relax_mode, parameters=None):
        """ Runs feasopt with a set of relaxable cts and numerical  preferences.

        Args:
            mdl: the model being relaxed
            relaxable_groups: a sequence of tuples (pref, cts) where
                cts is a sequence of constraints and pref is the preference
                for the relaxation of this group


        Returns:
            a solution object, or None.
        """
        self._resync_if_needed()
        self._sync_var_bounds()

        self_cplex = self.__cplex
        cpx_relax_groups = self._make_groups(relaxable_groups)

        feasopt_parameters = parameters or mdl.parameters
        feasopt_override_params = {feasopt_parameters.feasopt.mode: relax_mode.value}
        with _CplexOverwriteParametersCtx(self_cplex, feasopt_override_params) as cpx:
            # at this stage, we have a list of groups
            # each group is itself a list
            # the first item is a number, the preference
            # the second item is a list of constraint indices.
            self._last_solve_details = self._run_cpx_op_with_details(cpx.feasopt, *cpx_relax_groups)

        # feasopt state is restored by now
        cpx_solution = self_cplex.solution
        feas_status = cpx_solution.get_status()
        if self._is_relaxed_status_ok(feas_status):
            infeas_map = self._decode_infeasibilities(self_cplex, mdl, cpx_relax_groups)

            # all_ct_indices = []
            # index_extend = all_ct_indices.extend
            # for _, g in relaxable_groups:
            #     index_extend(ct.safe_index for ct in g)
            # raw_infeasibilities = cpx_solution.infeasibility.linear_constraints([], all_ct_indices)
            # infeas_map = {mdl.get_constraint_by_index(ctx): raw_infeasibilities[c] for c, ctx in enumerate(all_ct_indices)}
            relaxed_sol = self._make_solution(mdl, self.get_solve_status())
            relaxed_sol.store_infeasibilities(infeas_map)
            return relaxed_sol
        else:
            return None

    # def get_infeasibilities(self, cts):
    #     indices = [ct.index for ct in cts]
    #     # PCO: Daniel Junglas confirms using [] uses the last solution vector
    #     return self.__cplex.solution.infeasibility.linear_constraints([], indices)

    def refine_conflict(self, mdl, preferences=None, groups=None, parameters=None):
        """ Starts conflict refiner on the model.

        Args:
            mdl: The model for which conflict refinement is performed.
            preferences: a dictionary defining constraints preferences.
            groups: a list of ConstraintsGroup.

        Returns:
            A list of "TConflictConstraint" namedtuples, each tuple corresponding to a constraint that is
            involved in the conflict.
            The fields of the "TConflictConstraint" namedtuple are:
                - the name of the constraint or None if the constraint corresponds to a variable lower or upper bound
                - a reference to the constraint or to a wrapper representing a Var upper or lower bound
                - an :enum:'docplex.mp.constants.ConflictStatus' object that indicates the
                conflict status type (Excluded, Possible_member, Member...)
            This list is empty if no conflict is found by the conflict refiner.
        """

        try:
            # sync parameters
            mdl._sync_parameters_to_engine(parameters)

            cpx = self.__cplex

            if groups is None or groups == []:
                all_constraints = cpx.conflict.all_constraints()
                weighted_groups = self._build_weighted_constraints(mdl, all_constraints._gp, preferences)
                cpx.conflict.refine(*weighted_groups)
            else:
                groups_def = [self._build_group_definition_with_index(grp) for grp in groups]
                cpx.conflict.refine(*groups_def)

            return self._get_conflicts_local(mdl, cpx)

        except DOcplexException as docpx_e:
            mdl._set_solution(None)
            raise docpx_e

    def _build_group_definition_with_index(self, cts_group):
        return cts_group.preference, tuple([(self._get_constraint_type(ct), ct.index)
                                            for ct in cts_group.get_group_constraints()])

    @staticmethod
    def _get_constraint_type(ct):
        if isinstance(ct, LinearConstraint):
            return cpx_cst.CPX_CON_LINEAR
        if isinstance(ct, IndicatorConstraint):
            return cpx_cst.CPX_CON_INDICATOR
        if isinstance(ct, QuadraticConstraint):
            return cpx_cst.CPX_CON_QUADRATIC
        if isinstance(ct, VarLbConstraintWrapper):
            return cpx_cst.CPX_CON_LOWER_BOUND
        if isinstance(ct, VarUbConstraintWrapper):
            return cpx_cst.CPX_CON_UPPER_BOUND
        ct.model.fatal("Type unknown (or not supported yet) for constraint: " + repr(ct))

    def _build_weighted_constraints(self, mdl, groups, preferences=None):
        weighted_groups = []
        for (pref, seq) in groups:
            for (_type, _id) in seq:
                if _type == cpx_cst.CPX_CON_LOWER_BOUND or _type == cpx_cst.CPX_CON_UPPER_BOUND:
                    # Keep default preference
                    weighted_groups.append((pref, ((_type, _id),)))
                else:
                    ct = mdl.get_constraint_by_index(_id)
                    if preferences is not None:
                        new_pref = preferences.get(ct, None)
                        if new_pref is not None and isinstance(new_pref, numbers.Number):
                            pref = new_pref
                    weighted_groups.append((pref, ((_type, _id),)))
        return weighted_groups

    def _get_conflicts_local(self, mdl, cpx):
        # Build var by idx dict
        vars_by_index = mdl._build_index_dict(mdl._Model__allvars)

        try:
            conflicts = cpx.conflict.get()
            groups = cpx.conflict.get_groups()
        except CplexSolverError:
            # Return an empty list if no conflict is available
            return []

        result = []
        for (pref, seq), status in zip(groups, conflicts):
            if status == cpx_cst.CPX_CONFLICT_EXCLUDED:
                continue
            c_status = ConflictStatus(status)
            for (_type, _id) in seq:
                """
                Possible values for elements of grptype:
                    CPX_CON_LOWER_BOUND 	1 	variable lower bound
                    CPX_CON_UPPER_BOUND 	2 	variable upper bound
                    CPX_CON_LINEAR 	        3 	linear constraint
                    CPX_CON_QUADRATIC 	    4 	quadratic constraint
                    CPX_CON_SOS 	        5 	special ordered set
                    CPX_CON_INDICATOR 	    6 	indicator constraint
                """
                if _type == cpx_cst.CPX_CON_LOWER_BOUND:
                    result.append(TConflictConstraint(None, VarLbConstraintWrapper(vars_by_index[_id]), c_status))

                if _type == cpx_cst.CPX_CON_UPPER_BOUND:
                    result.append(TConflictConstraint(None, VarUbConstraintWrapper(vars_by_index[_id]), c_status))

                if _type == cpx_cst.CPX_CON_LINEAR:
                    ct = mdl.get_constraint_by_index(_id)
                    result.append(TConflictConstraint(ct.name, ct, c_status))

                if _type == cpx_cst.CPX_CON_QUADRATIC:
                    ct = mdl.get_quadratic_by_index(_id)
                    result.append(TConflictConstraint(ct.name, ct, c_status))

                if _type == cpx_cst.CPX_CON_SOS:
                    # TODO: DO NOT CREATE A FATAL ERROR: return a counter with a warning
                    mdl.fatal("Special Ordered Set constraints are not implemented (yet)")

                if _type == cpx_cst.CPX_CON_INDICATOR:
                    ct = mdl.get_indicator_by_index(_id)
                    result.append(TConflictConstraint(ct.name, ct, c_status))

        return result

    def dump(self, path):
        self._resync_if_needed()
        try:
            if path.find('.') > 0:
                self.__cplex.write(path)
            else:
                self.__cplex.write(path, filetype="lp")

        except CplexSolverError as cpx_se:  # pragma: no cover
            if cpx_se.args[2] == 1422:
                raise IOError("SAV export cannot open file: {}".format(path))
            else:
                raise DOcplexException("CPLEX error in SAV export: {0!s}", cpx_se)

    # def get_problem_type(self):
    #     """ CPLEX wrapper returns an integer."""
    #     return self.__cplex.get_problem_type()

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

    def connect_progress_listeners(self, progress_listeners):
        if not progress_listeners:  # pragma: no cover
            self.error_handler.info("No progress listeners to connect")
        elif self.is_mip():
            ccb = self.__cplex.register_callback(ConnectListenersCallback)
            ccb.initialize(progress_listeners)

    def sync_parameters(self, parameters):
        # INTERNAL
        # parameters is a root parameter group from DOcplex
        if parameters:
            cpx_params = self.__cplex._env.parameters
            for param in parameters:
                try:
                    cpx_params.set(param.cpx_id, param.current_value)
                except CplexError as cpxe:
                    cpx_msg = str(cpxe)
                    if cpx_msg.startswith("Bad parameter identifier"):
                        self.error_handler.warning("Parameter \"{0}\" is not recognized",
                                                   (param.qualified_name,))
                    else:
                        self.error_handler.error("Error setting parameter {0} to value {1}"
                                                 .format(param.short_name, param.current_value))

    def set_parameter(self, parameter, value):
        # value check is up to the caller.
        # parameter is a DOcplex parameter object
        try:
            self.__cplex._env.parameters._set(parameter.cpx_id, value)
        except CplexError as cpx_e:
            cpx_msg = str(cpx_e)
            if cpx_msg.startswith("Bad parameter identifier"):
                self.error_handler.warning("Parameter \"{0}\" is not recognized", (parameter.qualified_name,))
            else:  # pragma: no cover
                self.error_handler.error("Error setting parameter {0} to value {1}"
                                         .format(parameter.short_name, value))

    def get_parameter(self, parameter):
        try:
            return self.__cplex._env.parameters._get(parameter.cpx_id)
        except CplexError:  # pragma: no cover
            return parameter.default_value

    def get_solve_status(self):  # pragma: no cover
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

        # feasopt status values
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
                self._model._resync()
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
