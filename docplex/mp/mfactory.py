# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore

from docplex.mp.linear import Var, LinearExpr, _ZeroExpr, MonomialExpr
from docplex.mp.linear import _DummyFeasibleConstraint, _DummyInfeasibleConstraint
from docplex.mp.linear import LinearConstraintType, LinearConstraint, RangeConstraint, IndicatorConstraint

from docplex.mp.functional import FunctionalExpr, MaximumExpr, MinimumExpr, AbsExpr

from docplex.mp.utils import *


class _AbstractModelFactory(object):
    def __init__(self, model, engine):
        self._model = model
        self._engine = engine
        self._error_handler = model.error_handler


class ModelFactory(object):
    def is_free_lb(self, var_lb):
        return var_lb <= - self.infinity

    def is_free_ub(self, var_ub):
        return var_ub >= self.infinity

    def __init__(self, model, engine):
        self.__model = model
        self.__engine = engine
        self.__error_handler = model.error_handler
        self.infinity = engine.get_infinity()
        self.zero_expr = 0  # assigned to an expr later on.

    def init(self):
        model = self.__model
        self.zero_expr = LinearExpr(model, 0.0)
        self.unique_zero_expr = _ZeroExpr(model)

    def new_trivial_feasible_ct(self):
        return _DummyFeasibleConstraint(self.__model, self.zero_expr)

    def new_trivial_infeasible_ct(self):
        return _DummyInfeasibleConstraint(self.__model, self.zero_expr)

    def fatal(self, msg, *args):
        self.__error_handler.fatal(msg, args)

    def warning(self, msg, *args):
        self.__error_handler.warning(msg, args)

    def update_engine(self, engine):
        # the model has already disposed the old engine, if any
        self.__engine = engine
        self.infinity = engine.get_infinity()

    def new_var(self, vartype, lb=None, ub=None, varname=None):
        self_model = self.__model
        actual_name = varname or self_model._create_automatic_varname()
        var = Var(self_model, vartype, actual_name, lb, ub, is_automatic_name=not bool(varname))
        idx = self.__engine.create_one_variable(vartype, var.lb, var.ub, actual_name)
        self_model._register_one_var(var, idx)
        return var

    def _expand_bounds(self, keys, var_bound, default_bound, size, is_lb_or_ub):
        ''' Converts raw bounds data (either LB or UB) to CPLEX-compatible bounds list.
            If lbs is None, this is the default, return [].
            If lbs is [] take the default again.
            If it is a number, build a list of size <size> with this number.
            If it is a list, use it if size ok (check numbers??),
            else try it as a function over keys.
        '''
        if var_bound is None:
            # default lb is zero, default ub is infinity
            return []

        elif is_number(var_bound):
            if is_lb_or_ub:
                if var_bound == default_bound:
                    return []
                else:
                    return [var_bound] * size
            else:
                # ub
                if var_bound >= default_bound:
                    return []
                else:
                    return [var_bound] * size

        elif isinstance(var_bound, str):
            self._bad_bounds_fatal(var_bound)

        elif isinstance(var_bound, list):
            nb_bounds = len(var_bound)
            if nb_bounds == 0:
                return None  # use defaults
            elif nb_bounds < size:
                # see how we can use defaults for those missing bounds
                self.fatal("Variable bounds list is too small, expecting: %d, got: %d" % (size, nb_bounds))
            else:
                if nb_bounds > size:
                    self.warning("Variable bounds list is too large, required: %d, got: %d." % (size, nb_bounds))
                for b in range(size):
                    b_value = var_bound[b]
                    if not is_number(b_value):
                        self.fatal("Variable bounds list expects numbers, got: {0!s} (pos: #{1})",
                                   b_value, b)
                return var_bound

        elif is_iterator(var_bound):
            # unfold the iterator, as CPLEX needs a list
            return list(var_bound)

        elif isinstance(var_bound, dict):
            return [var_bound.get(k, default_bound) for k in keys]
        else:
            # try a function?
            try:
                _computed_bounds = [var_bound(k) for k in keys]
                if not is_iterable(_computed_bounds):
                    self._bad_bounds_fatal(var_bound)
                elif _computed_bounds:
                    for b in _computed_bounds:
                        if not is_number(b):
                            self.fatal("computed bound expects a number, got: {0!s}", b)
                return _computed_bounds
            except TypeError:
                self._bad_bounds_fatal(var_bound)
            except Exception as e:
                self.fatal("error calling function model bounds: {0!s}, error: {1!s}", var_bound, e)

    def _bad_bounds_fatal(self, bad_bound):
        self.fatal("unexpected variable bound: {0!s}, expecting: None|number|function|iterable", bad_bound)

    # @profile
    def new_var_list(self, var_container,
                     keys, vartype,
                     lb=None, ub=None,
                     name=str,
                     arity=1, key_format=None,
                     allow_empty_keys=True):
        if not keys:
            if allow_empty_keys:
                return []
            else:
                self.fatal("No keys to index the variables.")
        else:
            if any((k is None for k in keys)):
                self.fatal("A variable key cannot be None, see: {0!s}", keys)

        mdl = self.__model
        default_naming_fn = mdl._create_automatic_varname
        actual_naming_fn = compile_naming_function(keys, name, default_naming_fn, arity, key_format)

        # compute defaults once
        default_lb = vartype.default_lb
        default_ub = vartype.default_ub
        number_of_vars = len(keys)
        xlbs = self._expand_bounds(keys, lb, default_lb, number_of_vars, is_lb_or_ub=True)
        xubs = self._expand_bounds(keys, ub, default_ub, number_of_vars, is_lb_or_ub=False)
        # at this point both list are either [] or have size numberOfVars
        use_default_lbs = False
        use_default_ubs = False
        nb_lbs = len(xlbs)
        if 0 == nb_lbs:
            use_default_lbs = True
        elif number_of_vars == nb_lbs:
            pass
        else:
            mdl.fatal("Internal error: bad lbs size, got: {0}, expecting: {1}", nb_lbs, number_of_vars)
        nb_ubs = len(xubs)
        if 0 == nb_ubs:
            use_default_ubs = True
        elif number_of_vars == nb_ubs:
            pass
        else:
            mdl.fatal("Internal error: bad ubs size, got: {0}, expecting: {1}", nb_ubs, number_of_vars)


        is_safe = use_default_ubs and use_default_lbs
        is_auto = name is None  # not bool(name)

        allvars = [Var(mdl, vartype,
                       actual_naming_fn(key),
                       xlbs[k] if xlbs else default_lb,
                       xubs[k] if xubs else default_ub,
                       _safe_domain=is_safe,
                       is_automatic_name=is_auto,
                       container=var_container) for k, key in enumerate(keys)]

        # query the engine for a list of indices.
        indices = self.__engine.create_variables(keys, vartype, xlbs, xubs, actual_naming_fn)
        mdl._register_block_vars(allvars, indices)
        return allvars

    def constant_expr(self, cst):
        if 0 == cst:
            return self.unique_zero_expr
        else:
            return LinearExpr(self.__model, e=cst)

    def linear_expr(self, e=0, constant=0, name=None):
        # handle here the special case for 0.
        expr = LinearExpr(self.__model, e, constant, name)
        return expr

    def _new_zero_expr(self):
        return self.linear_expr()

    def scal_prod(self, dvars, coefs=1.0):
        # Testing anumpy array for its logical value will not work:
        # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        # we would have to trap the test for ValueError then call any()
        #
        if is_number(coefs):
            if 0 == coefs:
                return self._new_zero_expr()
            else:
                sum_expr = self.sum(dvars)
                return sum_expr * coefs
        else:
            self.__model.typecheck_iterable(coefs)

        if not is_iterable(dvars):
            dvars = [dvars]
        else:
            # iterable
            pass

        if has_len(coefs) and 0 == len(coefs):
            skip = True
        elif has_len(dvars) and 0 == len(dvars):
            skip = True
        else:
            skip = False
        if skip:
            return self._new_zero_expr()
        else:
            return self._scal_prod(dvars, coefs)

    def _scal_prod(self, dvars, coefs, cc_type=LinearExpr.counter_type):
        """
        INTERNAL, dvars is not empty.
        :param dvars:
        :param coefs:
        :return:
        """
        total_num = 0
        fcc = cc_type()

        normalizer = 0
        for item, coef in zip(dvars, coefs):
            if 0 == coef:
                pass
            elif isinstance(item, Var):
                fcc.update_from_item_value(item, coef)
                if coef < 0:
                    normalizer = coef

            elif isinstance(item, LinearExpr):
                fcc.update_from_scaled_dict(item._get_terms_dict(), coef)
                normalizer = -999

            elif isinstance(item, MonomialExpr):
                m_coef = item.coef
                fcc.update_from_item_value(item.var, m_coef * coef)
                if m_coef < 0:
                    normalizer = m_coef

            elif is_number(item):
                if item:
                    total_num += coef * item
            else:
                self.fatal("scal_prod accepts variables, expressions, numbers, not: {0!s}", item)
        # pass
        if normalizer < 0:  # normalize only if we saw a negative coeff
            pass
            fcc.normalize()

        res_dict = self._sort_terms_if_needed(fcc)
        scalprod_expr = LinearExpr(self.__model, e=res_dict, safe=True)
        # scalprod_expr._assign_terms(res_dict, is_safe=True, assume_normalized=True)
        return scalprod_expr

    def sum(self, sum_args):
        if is_iterable(sum_args):
            if is_iterator(sum_args):
                return self._sum_with_iter(sum_args)
            if has_len(sum_args) and 0 == len(sum_args):
                return self.linear_expr()
            elif isinstance(sum_args, dict):
                # handle dict: sum all values
                return self._sum_with_seq(sum_args.values())
            elif is_indexable(sum_args):
                first = sum_args[0]
                if self.__model._is_operand(first):
                    return self._sum_with_seq(sum_args)
                elif is_numpy_ndarray(sum_args):
                    return self._sum_with_iter(sum_args.flat)
                else:
                    self.fatal("cannot handle sequence with type: {0!s}", type(sum_args))
            else:
                return self._sum_with_seq(sum_args)
        elif is_number(sum_args):
            return sum_args
        else:
            return self.__model._to_linear_expr(sum_args)

    def _sort_terms_if_needed(self, counter, term_dict_type=LinearExpr.term_dict_type):
        if not self.__model._keep_ordering:
            return counter
        elif isinstance(counter, term_dict_type):
            return counter
        else:
            # normalize by sorting variables by increasing indices
            sorted_items = sorted(counter.items(), key=lambda vk: vk[0].get_index())
            od = term_dict_type(sorted_items)
            return od

    # Hi @profile
    def _sum_with_iter(self, args, cctype=LinearExpr.counter_type):
        """
        x-seq is an iterator so can be used only once.
        :param args:
        :return:
        """
        accumulated_ct = 0
        # do we really need to sort variables here??
        acc = cctype()
        for item in args:
            if isinstance(item, Var):
                acc.update_from_item(item)
            elif isinstance(item, MonomialExpr):
                acc.update_from_item_value(item._dvar, item._coef)
            elif isinstance(item, LinearExpr):
                acc.update(item._get_terms_dict())
                accumulated_ct += item.constant
            elif isinstance(item, FunctionalExpr):
                acc.update_from_item(item.functional_var)
            elif isinstance(item, _ZeroExpr):
                pass
            else:
                accumulated_ct += item

        res_terms = self._sort_terms_if_needed(acc)
        sum_x = LinearExpr(self.__model, e=res_terms, constant=accumulated_ct, safe=True)

        return sum_x

    def _varlist_to_terms(self, var_list,
                          cc_type=LinearExpr.counter_type,
                          term_dict_type=LinearExpr.term_dict_type):
        # INTERNAL: converts a sum of vars to a dict, sorting if needed.
        if self.__model._keep_ordering:
            varsum_terms = term_dict_type([(v, 1) for v in var_list])
        else:
            varsum_terms = cc_type(var_list)
        return varsum_terms

    # @profile
    def _sum_with_seq(self, x_list):
        for z in x_list:
            if not isinstance(z, Var):
                x_seq_all_variables = False
                break
        else:
            x_seq_all_variables = True

        if x_seq_all_variables:
            sumvars_terms = self._varlist_to_terms(x_list)
            return LinearExpr(self.__model, e=sumvars_terms, safe=True)
        else:
            return self._sum_with_iter(args=x_list)

    def _new_binary_constraint(self, lhs, ctype, rhs, name=None):
        ct = LinearConstraint(self.__model, lhs, ctype, rhs, name)
        return ct

    def new_le_constraint(self, e, rhs, ctname=None):
        return self._new_binary_constraint(e, LinearConstraintType.LE, rhs, name=ctname)

    def new_eq_constraint(self, e, rhs, ctname=None):
        return self._new_binary_constraint(e, LinearConstraintType.EQ, rhs, name=ctname)

    def new_ge_constraint(self, e, rhs, ctname=None):
        return self._new_binary_constraint(e, LinearConstraintType.GE, rhs, name=ctname)

    def new_range_constraint(self, lb, expr, rhs, ctname=None):
        # INTERNAL
        rng = RangeConstraint(self.__model, expr, lb, rhs, ctname)
        return rng

    def new_indicator_constraint(self, binary_var, linear_ct, active_value=1, ctname=None):
        # INTERNAL
        indicator_ct = IndicatorConstraint(self.__model, binary_var, linear_ct, active_value, ctname)
        return indicator_ct

    def new_max_expr(self, *args):
        nb_args = len(args)
        if 0 == nb_args:
            return - self.infinity
        elif 1 == nb_args:
            return args[0]
        else:
            return MaximumExpr(self.__model, args)

    def new_min_expr(self, *args):
        nb_args = len(args)
        if 0 == nb_args:
            return self.infinity
        elif 1 == nb_args:
            return args[0]
        else:
            return MinimumExpr(self.__model, args)

    def new_abs_expr(self, e):
        if is_number(e):
            return abs(e)
        else:
            self_model = self.__model
            return AbsExpr(self_model, self_model._to_linear_expr(e))

    def resync_whole_model(self):
        self_model = self.__model
        self_engine = self.__engine

        for var in self_model.iter_variables():
            # do not call create_one_var public API
            # or resync would loop
            idx = self_engine.create_one_variable(var.vartype, var.lb, var.ub, var.name)
            if idx != var.get_index():
                print("index discrepancy: {0!s}, new index= {1}, old index={2}".format(var, idx,
                                                                                       var.get_index()))  # pragma: no cover

        for ct in self_model.iter_constraints():
            if isinstance(ct, LinearConstraint):
                self_engine.create_binary_linear_constraint(ct)
            elif isinstance(ct, RangeConstraint):
                self_engine.create_range_constraint(ct)
            elif isinstance(ct, IndicatorConstraint):
                self_engine.create_indicator_constraint(ct)
            else:
                self_model.error("Unexpected constraint type: {0!s} - ignored", type(ct))  # pragma: no cover

        # send objective
        self_engine.set_objective(self_model.objective_sense, self_model.objective_expr)
