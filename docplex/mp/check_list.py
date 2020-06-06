import importlib
import platform
import sys
import warnings

from sys import version_info


def check_import(mname):
    try:
        importlib.import_module(mname)
        return True
    except ImportError:
        return False


def check_platform():
    platform_error_msg = "docplex is not compatible with this version of Python: only 64 bits on Windows, Linux, Darwin and AIX, with Python 2.7.9+, 3.4+ are supported."

    platform_system = platform.system()
    if platform_system in ('Darwin', 'Linux', 'Windows', 'Microsoft', 'AIX'):
        if version_info[0] == 3:
            if version_info < (3, 4, 0):
                warnings.warn(platform_error_msg)
        elif version_info[0] == 2:
            if version_info[1] != 7:
                warnings.warn(platform_error_msg)
        else:
            warnings.warn(platform_error_msg)
    else:
        print("docplex is not officially supported on this platform. Use it at your own risk.", RuntimeWarning)

    is_64bits = sys.maxsize > 2 ** 32
    if is_64bits is False:
        warnings.warn("docplex is not officially supported on 32 bits. Use it at your own risk.", RuntimeWarning)


def run_docplex_check_list():
    check_platform()
    cplex_latest_version_as_tuple = (12, 0)

    diagnostics = []

    # check requirements
    for rm in ["six", "enum", "cloudpickle"]:
        if not check_import(rm):
            diagnostics.append("Module {0} is missing, run: pip install {0}".format(rm))

    # check pandas
    try:
        import pandas as pd
        # noinspection PyUnresolvedReferences
        from pandas import DataFrame, Series
        dd = DataFrame({})
    except ImportError:
        print("-- pandas is not present, some features might be unavailable.")

    from docplex.mp.environment import Environment
    Environment().print_information()

    # check cplex
    try:
        # noinspection PyUnresolvedReferences
        from cplex import Cplex

        cpx = Cplex()
        cpxv = cpx.get_version()
        cpxvt = tuple(float(x) for x in cpx.get_version().split("."))
        if cpxvt < cplex_latest_version_as_tuple:
            lcpxv = ".".join(str(z for z in cplex_latest_version_as_tuple))
            print("* Your cplex version {0} is not the latest, {1} is available".format(cpxv, lcpxv))
        else:
            print("* you have the latest cplex version")

        del cpx
    except ImportError:
        print("Cplex DLL not found, if present, you must add it to PYTHONPATH")

    # check for
    try:
        # noinspection PyUnresolvedReferences
        from docplex.mp.model import Model
        m = Model()
        print("* DOcplex models solved with: {0}".format(m.get_engine().name))

        # promotional?
        if Model.is_cplex_ce():
            print("! Cplex promotional version , limited to 1000 variables, 1000 constraints")
            diagnostics.append("Your local CPLEX edition is limited. Consider purchasing a full license.")


    except ImportError:
        print("Docplex is not present: cannot import class docplex.mp.model")
    except Exception as e:
        print(" Exception raised when creating a model instance: {0!s}".format(e))

    if diagnostics:
        print("* diagnostics: {0}".format(len(diagnostics)))
        for s in diagnostics:
            print("  - {0}".format(s))
    else:
        print("No problem found: you're all set!")


if __name__ == "__main__":
    run_docplex_check_list()

