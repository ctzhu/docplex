# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore

import operator
import os

# docplex
from docplex.mp.model import Model, Environment
from docplex.mp.utils import DOcplexException
from docplex.mp.params.cplex_params import get_params_from_cplex_version
# cplex
from cplex import Cplex
from cplex._internal._subinterfaces import ObjSense
from cplex.exceptions import CplexError, CplexSolverError

class _CplexReaderFileContext(object):
    def __init__(self, filename, read_method=None):
        self._cplex = None
        self._filename = filename
        self._read_method = read_method or ["read"]

    def __enter__(self):
        cpx = Cplex()
        # no output from cplex
        cpx.set_results_stream(None)
        cpx.set_log_stream(None)
        cpx.set_warning_stream(None)
        cpx.set_error_stream(None)
        self_read_fn = cpx
        for m in self._read_method:
            self_read_fn = self_read_fn.__getattribute__(m)

        try:
            self_read_fn(self._filename)
            self._cplex = cpx
            return cpx

        except CplexError as cpx_e:
            print("*CPLEX error {0!s} reading file {1} - exiting".format(cpx_e, self._filename))
            del cpx
            return None

    # noinspection PyUnusedLocal
    def __exit__(self, exc_type, exc_val, exc_tb):
        cpx = self._cplex
        if cpx is not None:
            del cpx
            self._cplex = None



class ModelReader(object):
    # INTERNAL

    sense_map = {'G': operator.ge, 'L': operator.le, 'E': operator.eq}

    # internal class to store range data
    # canno use tuples as they are immutable
    class _RangeData(object):
        def __init__(self, var_index, var_name, lb=0, ub=1e+75):
            self.var_index = var_index
            self.var_name = var_name
            self.lb = lb
            self.ub = ub

    @staticmethod
    def _cplex_ctsense_to_python_op(cpx_sense, _sense_map=sense_map):
        return _sense_map[cpx_sense]

    @staticmethod
    def _build_linear_expr_from_sparse_pair(mdl, var_map, cpx_sparsepair):
        expr = mdl.linear_expr()
        for cpx_index, cpx_val in zip(cpx_sparsepair.ind, cpx_sparsepair.val):
            expr.add_term(var_map[cpx_index], cpx_val)
        return expr

    def __init__(self, use_block_cts=True):
        self._use_block_constraints = use_block_cts

    def read_prm(self, filename):
        """ Reads a CPLEX PRM file.

        Reads a CPLEX file with parameters, and returns a DOcplex parameter group
        instance. This parameter object can be used in a solve().

        Args:
            filename: a path string

        Returns:
            A RootParameterGroup object , if the read operation succeeds, else None.
        """
        with _CplexReaderFileContext(filename, read_method=["parameters", "read_file"]) as cpx:
            if cpx:
                # raw parameters
                params = get_params_from_cplex_version(cpx.get_version())
                for param in params:
                    try:
                        cpx_value = cpx._env.parameters._get(param.cpx_id)
                        if cpx_value != param.default_value:
                            param.set(cpx_value)
                    except CplexError:
                        pass
                return params
            else:
                return None

    def read_model(self, filename, model_name=None, verbose=True, **kwargs):
        """ Reads a model from a CPLEX export file.

        Accepts all formats exported by CPLEX: LP, SAV, MPS...

        If an error occurs while reading the file, the message of the exception
        is printed and the function returns None.

        Args:
            file: a path string
            model_name: an optional name for the newly created model. If None,
                the model name will be the path basename.
            kwargs: a dict of keyword-based arguments, that are used when creating the model
                instance.

        Example:
            m = read_model("c:/temp/foo.mps", model_name="docplex_foo", solver_agent="docloud", output_level=100)


        Returns:
            an instance of Model, or None if an exception is raised.

        """
        if not os.path.exists(filename):
            print("* file not found: {0}".format(filename))
            return None

        # extract pure basename
        if model_name:
            name_to_use = model_name
        else:
            basename = os.path.basename(filename)
            dotpos = basename.find(".")
            if dotpos > 0:
                name_to_use = basename[:dotpos]
            else:
                name_to_use = basename

        if 0 == os.stat(filename).st_size:
            print("* file is empty: {0} - exiting".format(filename))
            return Model(name=name_to_use, **kwargs)


        # print("-> start reading file: {0}".format(filename))
        cpx = Cplex()
        # no warnings
        cpx.set_results_stream(None)
        cpx.set_log_stream(None)
        cpx.set_warning_stream(None)
        cpx.set_error_stream(None)  # remove messages about names
        try:
            cpx.read(filename)
        except CplexError as cpx_e:
            if verbose:
                print("*CPLEX error {0!s} reading file {1} - exiting".format(cpx_e, filename))
            return None

        range_map = {}
        final_output_level = kwargs.get("output_level", "info")

        #  print("-> end CPLEX read file: {0}".format(filename))
        try:

            mdl = Model(name=name_to_use, **kwargs)
            mdl.set_quiet()  # output level set to ERROR
            vartype_cont = mdl.continuous_vartype
            vartype_map = {'B': mdl.binary_vartype, 'I': mdl.integer_vartype, 'C': mdl.continuous_vartype}
            # 1 upload variables
            nb_vars = cpx.variables.get_num()
            all_names = cpx.variables.get_names()
            all_types = cpx.variables.get_types() if cpx._is_MIP() else []
            all_lbs = cpx.variables.get_lower_bounds()
            all_ubs = cpx.variables.get_upper_bounds()
            idx_to_var_map = {}
            # vars
            for v in range(nb_vars):
                varname = all_names[v]

                cpx_vtype = all_types[v] if all_types else 'C'
                vartype = vartype_map.get(cpx_vtype, vartype_cont)
                cpx_lb = all_lbs[v]
                cpx_ub = all_ubs[v]
                lb = cpx_lb if cpx_lb != vartype.default_lb else None
                ub = cpx_ub if cpx_ub != vartype.default_ub else None

                if varname.startswith("Rg"):
                    # generated var for ranges
                    range_map[v] = self._RangeData(var_index=v, var_name=varname, ub=ub)
                else:
                    docplex_var = mdl._var(vartype, lb, ub, varname)
                    idx_to_var_map[v] = docplex_var

            # 2. upload linear constraints and ranges (mixed in cplex)
            cpx_linearcts = cpx.linear_constraints
            nb_linear_cts = cpx_linearcts.get_num()
            all_rows = cpx_linearcts.get_rows()
            all_rhs = cpx_linearcts.get_rhs()
            all_senses = cpx_linearcts.get_senses()
            try:
                all_names = cpx_linearcts.get_names()  # can be []
            except CplexError:
                all_names = None
            all_range_values = cpx_linearcts.get_range_values()

            has_range = range_map or any(s == "R" for s in all_senses)
            deferred_cts = []
            deferred_ctnames = []
            postpone = self._use_block_constraints

            for c in range(nb_linear_cts):
                row = all_rows[c]
                sense = all_senses[c]
                rhs = all_rhs[c]
                ctname = all_names[c] if all_names else None
                range_val = all_range_values[c]

                indices = row.ind
                coefs = row.val
                range_data = None

                # build an expr

                if not has_range:
                    expr = mdl.scal_prod((idx_to_var_map[idx] for idx in indices), coefs)
                    op = self._cplex_ctsense_to_python_op(sense)
                    ct = op(expr, rhs)
                    if postpone:
                        deferred_cts.append(ct)
                        deferred_ctnames.append(ctname)
                    else:
                        mdl.add_constraint(ct, ctname)
                else:
                    expr = mdl.linear_expr()
                    for idx, koef in zip(indices, coefs):
                        var = idx_to_var_map.get(idx, None)
                        if var:
                            expr._add_term(var, koef)
                        elif idx in range_map:
                            assert range_data is None  # cannot use two range vars
                            range_data = range_map[idx]
                        else:
                            print("ERROR: index not in var map or range map: {0}".format(idx))

                    if range_data:
                        assert sense == 'E'
                        rng_lb = rhs
                        rng_ub = rhs + range_data.ub
                        mdl.add_range(lb=rng_lb, expr=expr, ub=rng_ub, rng_name=ctname)
                    else:
                        if sense == 'R':
                            # range min is rangeval
                            range_lb = rhs
                            range_ub = rhs + range_val
                            mdl.add_range(lb=range_lb, ub=range_ub, expr=expr, rng_name=ctname)
                        else:
                            op = self._cplex_ctsense_to_python_op(sense)
                            ct = op(expr, rhs)
                            mdl.add_constraint(ct, ctname)
            if deferred_cts:
                # add constraint as a block
                mdl._add_constraints(cts=deferred_cts, names=deferred_ctnames, do_check=False)  # disable typechecks

            # 3. upload indicators
            cpx_indicators = cpx.indicator_constraints
            nb_indicators = cpx_indicators.get_num()
            try:
                all_ind_names = cpx_indicators.get_names()

            except CplexSolverError as cpxse:
                errcode = cpxse.args[2]
                # when all indicators have no names, cplex raises this error
                # CPLEX Error  1219: No names exist.
                if errcode == 1219:
                    # seems cplex raises this error when no indicator has name
                    all_ind_names = []
                else:
                    raise cpxse  # this is something else.

            except IndexError as e:
                # any other Pythonlayer error is abnormal
                print("Error when reading file: {0}, raised: {1!s}".format(filename, str(e)))
                raise e

            all_ind_bvars = cpx_indicators.get_indicator_variables()
            all_ind_rhs = cpx_indicators.get_rhs()
            all_ind_linearcts = cpx_indicators.get_linear_components()
            all_ind_senses = cpx_indicators.get_senses()
            all_ind_complemented = cpx_indicators.get_complemented()
            for i in range(nb_indicators):
                ind_bvar = all_ind_bvars[i]
                ind_name = all_ind_names[i] if all_ind_names else None
                ind_rhs = all_ind_rhs[i]
                ind_linear = all_ind_linearcts[i]  # SparsePair(ind, val)
                ind_sense = all_ind_senses[i]
                ind_complemented = all_ind_complemented[i]
                # 1 . check the bvar is ok
                ind_bvar = idx_to_var_map[ind_bvar]
                ind_linexpr = self._build_linear_expr_from_sparse_pair(mdl, idx_to_var_map, ind_linear)
                op = self._cplex_ctsense_to_python_op(ind_sense)
                ind_ct = op(ind_linexpr, ind_rhs)
                mdl.add_indicator(ind_bvar, ind_ct, active_value=ind_complemented, name=ind_name)

            # 4. upload objective
            cpx_obj = cpx.objective
            cpx_sense = cpx_obj.get_sense()
            obj_expr = mdl.linear_expr()
            # for v in range(nb_vars):
            #     if v in idx_to_var_map:
            #         obj_coef = cpx_obj.get_linear(v)
            #         obj_expr._add_term(idx_to_var_map[v], obj_coef)

            cpx_all_obj_coeffs = cpx_obj.get_linear()
            all_obj_vars  = []
            all_obj_coefs = []

            for v in range(nb_vars):
                if v in idx_to_var_map:
                    obj_coeff = cpx_all_obj_coeffs[v]
                    all_obj_coefs.append(obj_coeff)
                    all_obj_vars.append(idx_to_var_map[v])
                    #obj_expr._add_term(idx_to_var_map[v], cpx_all_obj_coeffs[v])
            obj_expr = mdl.dot(all_obj_vars, all_obj_coefs)
            is_maximize = cpx_sense == ObjSense.maximize

            if not obj_expr.is_constant():
                if is_maximize:
                    mdl.maximize(obj_expr)
                else:
                    mdl.minimize(obj_expr)

        except CplexError as cpx_e:
            print("* CPLEX error: {0} reading file {1}, code={2}".format(cpx_e.args[0], filename, cpx_e.args[2]))
            mdl = None

        except DOcplexException as do_e:
            print("! Internal DOcplex error raised: {0!s} while reading file {1}".format(do_e, filename))
            mdl = None

        except Exception as any_e:
            print("Internal exception raised: {0!s} while reading file {1}".format(any_e, filename))
            mdl = None

        finally:
            # clean up CPLEX instance...
            del cpx

        mdl.output_level = final_output_level
        return mdl


def read_model(filename, verbose=False):
    env = Environment()
    if not env.has_cplex:
        print("Model.read() requires a CPLEX DLL")
        return None
    elif not isinstance(filename, str):
        print("Model.read() expects a path, got: {0!s}".format(filename))
        return None

    docplex_reader = ModelReader()
    m = docplex_reader.read_model(filename, verbose=verbose)
    if m is None:
        print("* cannot read file: {0}".format(filename))
    return m


def read_all_in_dir(directory, verbose=True, use_block_cts=True):
    from collections import Counter

    mreader = ModelReader(use_block_cts=use_block_cts)
    if not os.path.isdir(directory):
        print("Not a directory: {0}".format(directory))
        return

    read_count = 0
    error_files = set({})
    all_models = []
    cc = Counter()
    for f in os.listdir(directory):
        full_path = os.path.join(directory, f)
        extension = os.path.splitext(full_path)[1]
        if extension in frozenset({".lp", ".sav", ".mps"}):
            read_count += 1
            cc.update({extension: 1})
            m = None
            try:
                #  print("{0} --> start reading file: {1}".format(read_count, full_path))
                m = mreader.read_model(full_path, verbose=verbose)
            except DOcplexException:
                m = None
            except Exception as e:
                print("Python exception while reading file {0}: {1!s}".format(full_path, e))
                m = None
            finally:
                status = "OK" if m is not None else "KO"
                print("{0}> read file: {1}: {2}".format(read_count, full_path, status))

            if m is None:
                print("! ERROR reading: {0}".format(full_path))
                error_files.add(full_path)
            else:
                all_models.append(m)
    print("* files by extension: {0!s}".format(cc))
    print("* read {0} files, #errors ={1}".format(read_count, len(error_files)))
    if error_files:
        for f in error_files:
            print("**** error reading file: {0}".format(f))
    return all_models


if __name__ == "__main__":
    tempmodels = read_all_in_dir("c:/temp")
