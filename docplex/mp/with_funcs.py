# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2020, 2021
# --------------------------------------------------------------------------

from contextlib import contextmanager

from docplex.mp.constants import ObjectiveSense


@contextmanager
def model_parameters(mdl, temp_parameters):
    """ This contextual function is used to override a model's parameters.
    As a contextual function, it is intended to be used with the `with` construct, for example:

    >>> with model_parameters(mdl, {"timelimit": 30, "empahsis.mip": 4}) as mdl2:
    >>>     mdl2.solve()


    The new model returned from the `with` has temporary parameters overriding those of the initial model.

    when exiting the with block, initial parameters are restored.

    :param mdl: an instance of `:class:Model`.
    :param temp_parameters: accepts either a dictionary of qualified names to values, for example
        {"mip.tolernaces.mipgap": 0.03, "emphasis.mip": 4}, or a dictionary from parameter objects to values.
    :return: the same model, with overridden parameters.

    See Also:
        - :func:`docplex.mp.params.Parameter.qualified_name'

    *New in version 2.21*
    """
    if not temp_parameters:
        try:
            yield mdl
        finally:
            pass
    else:
        ctx = mdl.context
        saved_context = ctx
        temp_ctx = ctx.copy()
        try:
            temp_ctx.update_cplex_parameters(temp_parameters)
            mdl.context = temp_ctx
            yield mdl
        finally:
            mdl.context = saved_context
            return mdl


@contextmanager
def model_objective(mdl, temp_obj, temp_sense=None):
    """ This contextual function is used to temporarily override the objective of a model.
    As a contextual function, it is intended to be used with the `with` construct, for example:

    >>> with model_objective(mdl, x+y) as mdl2:
    >>>     mdl2.solve()


    The new model returned from the `with` has a temporary objective overriding the initial objective.

    when exiting the with block, the initial objective and sense are restored.

    :param mdl: an instance of `:class:Model`.
    :param temp_obj: an expression.
    :param temp_sense: an optional objective sense to override thg model's. Default is None (keep same objective).
        Accepts either an instance of enumerated value `:class:docplex.mp.constants.ObjectiveSensea string 'min' or 'max'.
    :return: the same model, with overridden objective.

    *New in version 2.21*
    """
    saved_obj = mdl.objective_expr
    saved_sense = mdl.objective_sense
    new_sense_ = ObjectiveSense.parse(temp_sense, mdl) if temp_sense is not None else None

    try:
        mdl.set_objective_expr(temp_obj)
        if new_sense_:
            mdl.set_objective_sense(new_sense_)

        yield mdl
    finally:
        mdl.set_objective_expr(saved_obj)
        if new_sense_:
            mdl.set_objective_sense(saved_sense)
