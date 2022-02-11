# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore

from cplex._internal._procedural import _safeDoubleArray, check_status
import cplex._internal._pycplex as CR

def safe_len(begin, end):
    return max(1, end - begin + 1)

def call_double_array_pycplex_fn(env, lp, fname, begin, end):
    lblen = safe_len(begin, end)
    dbl_array = _safeDoubleArray(lblen)
    status = fname(env, lp, dbl_array, begin, end)
    check_status(env, status)
    return dbl_array

def get_lb_array(cpx, begin, end):
    env = cpx._env._e
    lp = cpx._lp
    return call_double_array_pycplex_fn(env, lp, CR.CPXXgetlb, begin, end)

def get_ub_array(cpx, begin, end):
    env = cpx._env._e
    lp = cpx._lp
    return call_double_array_pycplex_fn(env, lp, CR.CPXXgetub, begin, end)

def get_obj_array(cpx, begin, end):
    env = cpx._env._e
    lp = cpx._lp
    return call_double_array_pycplex_fn(env, lp, CR.CPXXgetobj, begin, end)
