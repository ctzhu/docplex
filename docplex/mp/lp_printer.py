# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


from __future__ import print_function
# from six import iteritems, itervalues

import re

from docplex.mp.linear import *
from docplex.mp.mprinter import TextModelPrinter

from docplex.mp.utils import StringIO

from docplex.mp.format import LP_format

# gendoc: ignore


class LPModelPrinter(TextModelPrinter):

    _lp_re = re.compile(r"[a-df-zA-DF-Z!#$%&()/,;?@_`'{}|\"][a-zA-Z0-9!#$%&()/.,;?@_`'{}|\"]*")

    def __init__(self, hide_user_names=False, indentLevel=1, wrap_lines=True):
        TextModelPrinter.__init__(self,
                                  indent=indentLevel,
                                  comment_start='\\',
                                  hide_user_names=hide_user_names,
                                  nb_digits_for_floats=9,
                                  wrap_lines=wrap_lines)

        # symbols for constraints, not including whitespace 
        self.ct_symbol_map = {LinearConstraintType.EQ: "=",  # BEWARE NOT ==
                              LinearConstraintType.LE: "<=",
                              LinearConstraintType.GE: ">="}

        self.wrapper = self.create_wrapper(line_width=79, initial_indent=1)

        self._noncompliant_varname = None

    def get_format(self):
        return LP_format

    def encrypt_user_names(self):
        """
        Encrypt either if expliclity asked for or if some name is not LP-compliant
        :return:
        """
        return TextModelPrinter.encrypt_user_names(self) or self._noncompliant_varname

    def _print_ct_name(self, oss, ct):
        lp_ctname = self.ct_print_name(ct)
        indented = self._indent_level
        if lp_ctname is not None:
            oss.write(lp_ctname)
            oss.write(": ")
            indented += (len(lp_ctname) + 1)
        return indented

    def _print_binary_ct(self, oss, num_printer, var_name_map, binary_ct):
        # ensure consistent ordering: left termes then right terms
        iter_diff_coeffs = binary_ct._iter_net_coeffs(ordering=True)
        self._print_expr_iter(oss, num_printer, var_name_map, iter_diff_coeffs, constant_to_print=None)
        oss.write(' %s ' % self.ct_symbol_map[binary_ct.type])
        num_printer.to_stringio(oss, binary_ct.rhs())

    def _print_ranged_ct(self, oss, num_printer, var_name_map, ranged_ct):
        exp = ranged_ct.expr
        (varname, rhs, _) = self._rangeData[ranged_ct]
        self._print_expr(oss, num_printer, var_name_map, exp, False)
        oss.write(' - ')
        oss.write(varname)
        oss.write(' = ')
        oss.write(self._num_to_string(rhs))

    def _print_indicator_ct(self, oss, num_printer, var_name_map, indicator_ct):
        """
        Prints an indicator ct in LP
        :param oss:
        :param indicator_ct:
        :return:
        """
        INDICATOR_SYMBOL = " -> "
        binary_var = indicator_ct.indicator_var

        oss.write(var_name_map[binary_var])
        oss.write(" = ")
        oss.write("%d" % indicator_ct.logical_rhs)
        oss.write(INDICATOR_SYMBOL)
        self._print_binary_ct(oss, num_printer, var_name_map, indicator_ct.linear_constraint)

    def _print_constraint(self, oss, num_printer, var_name_map, ct):
        indented = 0
        if isinstance(ct, LinearConstraint):
            if not ct.is_trivial_feasible():
                indented = 0 if self._hide_user_names else self._print_ct_name(oss, ct)
                self._print_binary_ct(oss, num_printer, var_name_map, ct)
        elif isinstance(ct, RangeConstraint):
            indented = 0 if self._hide_user_names else self._print_ct_name(oss, ct)
            self._print_ranged_ct(oss, num_printer, var_name_map, ct)
        elif isinstance(ct, IndicatorConstraint):
            indented = 0 if self._hide_user_names else self._print_ct_name(oss, ct)
            self._print_indicator_ct(oss, num_printer, var_name_map, ct)
        else:
            ct.error("ERROR: unexpected constraint not printed: {0!s}".format(ct))

        self.wrap_and_print(oss, self.wrapper, subsequent_level=indented)

    def _print_expr(self, oss, num_printer, var_name_map, expr, print_constant=True):
        """ prints an expr to a string io-like object."""
        printed_constant = expr._get_constant() if print_constant else None
        term_iter = expr.iter_terms()
        self._print_expr_iter(oss, num_printer, var_name_map, term_iter, printed_constant)

    #  @profile
    def _print_expr_iter(self, oss, num_printer, var_name_map, expr_iter, constant_to_print):
        c = 0
        for (v, coeff) in expr_iter:
            if 0 == coeff:
                continue

            # 1 separator
            if c > 0:
                oss.write(' ')

            wrote_sign = False
            if coeff < 0 or c > 0:
                oss.write('-' if coeff < 0 else '+')
                wrote_sign = True
            # sign has been taken care of, drop it
            if coeff < 0:
                coeff = -coeff

            if 1 != coeff:
                if wrote_sign:
                    oss.write(' ')
                num_printer.to_stringio(oss, coeff)
            if wrote_sign or 1 != coeff:
                oss.write(' ')
            oss.write(var_name_map[v])
            c += 1

        if constant_to_print is not None:
            if constant_to_print != 0:
                if c == 0:
                    num_printer.to_stringio(oss, constant_to_print)
                else:
                    if constant_to_print < 0:
                        sign = ' - '
                        constant_to_print = -constant_to_print
                    else:
                        sign = ' + '
                    oss.write(sign)
                    num_printer.to_stringio(oss, constant_to_print)
            else:
                pass

        elif not c:
            # expr is empty, we must print something, so 0
            oss.write("0")

    def _print_var_block(self, iter_vars, header):
        printed_header = False
        oss = None
        c = 0
        for v in iter_vars:
            lp_name = self._var_print_name(v)
            if oss is None:
                oss = StringIO()
            if not printed_header:
                print('\n', header, sep='')
                printed_header = True
            if c > 0:
                oss.write(' ')
            oss.write(lp_name)
            c += 1
        if printed_header:
            self.wrap_and_print(oss, self.wrapper, subsequent_level=self._indent_level)

    def _print_var_bounds(self, varname, lb, ub):
        LE = "<="
        FREE = "Free"
        VARNAME_INDENT = 4 * ' '

        if lb is None and ub is None:
            # try to indent with space of '0 <= ', that is 5 space
            print(VARNAME_INDENT, varname, FREE)
        elif lb is None:
            print(varname, LE, self._num_to_string(ub))
        elif ub is None:
            print(self._num_to_string(lb), LE, varname)
        elif lb == ub:
            print(VARNAME_INDENT, varname, '=', self._num_to_string(lb))
        else:
            print(self._num_to_string(lb), LE, varname, LE, self._num_to_string(ub))

    TRUNCATE = 200

    def fix_name(self, mobj, prefix, local_index_map):
        raw_name = mobj.name

        # anonymous constraints must be named in a LP (we follow CPLEX here)
        if self.encrypt_user_names() or not raw_name or self._is_automatic_name(raw_name):
            return self._make_prefix_name(mobj, prefix, local_index_map, offset=1)
        else:
            # swap blanks with underscores
            fixed_name = self.translate_chars(raw_name)
            # truncate if necessary, again this does nothing if name is too short
            return fixed_name[:self.TRUNCATE]

    def _print_model_name(self, model):
        printed_name = model.name or 'CPLEX'
        print("\\Problem name: %s\n" % printed_name)

    @staticmethod
    def _is_lp_compliant(name):
        if name is None:
            return True
        # PUT THIS SOMEWHERE ELSE
        fixed_name = LPModelPrinter.fix_whitespace(name)
        #lp_re = re.compile(r"[a-df-zA-DF-Z!#$%&()/,;?@_`'{}|\"][a-zA-Z0-9!#$%&()/.,;?@_`'{}|\"]*")
        lp_match = LPModelPrinter._lp_re.match(fixed_name)
        if lp_match is None:
            return False
        elif lp_match.start() != 0:
            return False
        elif lp_match.end() != len(fixed_name):
            return False
        else:
            return True

    @staticmethod
    def _is_injective(name_map):
        nb_keys = len(name_map)
        nb_different_names = len(set(name_map.values()))
        return nb_different_names == nb_keys

    #@profile
    def printModelInternal(self, model):
        # first , check that all fixed names are lp compliant
        for dv in model.iter_variables():
            if dv.has_user_name() and not self._is_lp_compliant(dv.name):
                self._noncompliant_varname = dv.name
                break
        else:
            self._noncompliant_varname = None

        #self._noncompliant_varname = [dv for dv in model.iter_variables() if not self._is_lp_compliant(dv.name)]
        if self._noncompliant_varname:
            model.error_handler.info("#non-compliant LP name: |{0}|\n".format(self._noncompliant_varname))

        if not self._is_injective(self._var_name_map):
            # use indices to differentiate names
            sys.__stdout__.write("-- refining variable names\n")
            k = 0
            for dv, lp_name in iteritems(self._var_name_map):
                refined_name = "%s#%d" % (lp_name, k)
                self._var_name_map[dv] = refined_name
                k += 1
            #assert self._is_injective(self._var_name_map)

        TextModelPrinter.prepare(self, model)
        self_num_printer = self._num_printer
        var_name_map = self._var_name_map

        self._print_signature()
        self._print_encoding()
        self._print_model_name(model)

        # print objective
        print(model.objective_sense.name)
        objexpr = model.objective_expr
        obj_constant = objexpr.constant
        # the new dummy var has name 'x' + (max_index+2
        # why +2 because name indices start from 1 and CPLEX start from 0

        obj_constant_term_varname = self.get_extra_var_name(model, pattern='x%d')
        oss = StringIO()
        oss.write('obj: ')
        if not objexpr.is_constant():
            self._print_expr(oss, self_num_printer, var_name_map, objexpr, print_constant=False)
            oss.write(' + ')
        oss.write(obj_constant_term_varname)
        # 6 is len("obj: ") + 1
        self.wrap_and_print(oss, self.wrapper, subsequent_level=6)
        print("Subject To")

        for ct in model.iter_constraints():
            oss = StringIO()
            self._print_constraint(oss, self_num_printer, var_name_map, ct)

        print("\nBounds")
        for dvar in model.iter_variables():
            lp_name = self._var_print_name(dvar)
            if dvar.is_binary():
                self._print_var_bounds(lp_name, 0, 1)
            else:
                free_lb = dvar.has_free_lb()
                free_ub = dvar.has_free_ub()
                if free_lb and free_ub:
                    self._print_var_bounds(lp_name, None, None)
                elif free_ub:
                    self._print_var_bounds(lp_name, dvar.lb, None)
                elif free_lb:
                    self._print_var_bounds(lp_name, None, dvar.ub)
                else:
                    self._print_var_bounds(lp_name, dvar.lb, dvar.ub)
        # add constant term
        self._print_var_bounds(obj_constant_term_varname, obj_constant, obj_constant)
        # add ranged cts vars
        for rng in model.iter_range_constraints():
            (varname, _, ub) = self._rangeData[rng]
            self._print_var_bounds(varname, 0, ub)

        self._print_var_block(model.iter_binary_vars(), 'Binaries')
        self._print_var_block(model.iter_integer_vars(), 'Generals')

        print("End")
