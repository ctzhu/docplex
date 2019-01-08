# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore
import warnings

from docplex.mp.engine import NoSolveEngine, ZeroSolveEngine, FakeFailEngine, RaiseErrorEngine
from docplex.mp.docloud_engine import DOcloudEngine


class EngineFactory(object):
    """ A factory class that manages creation of solver instances.
    """
    _engine_types_by_key = {"local":   NoSolveEngine,
                            "nosolve": NoSolveEngine,
                            "zero":    ZeroSolveEngine,
                            "fail":    FakeFailEngine,
                            "raise":   RaiseErrorEngine,
                            "docloud": DOcloudEngine}

    cplex_engine_type = None

    @classmethod
    def _get_engine_by_code(cls, code, default_engine):
        if code is None:
            return default_engine
        else:
            return cls._engine_types_by_key.get(code.lower(), default_engine)

    @classmethod
    def _extend(cls, code, engine_type):
        # INTERNAL!!!
        cls._engine_types_by_key[code] = engine_type

    @classmethod
    def new_engine(cls, solver_agent, env, model, context=None):
        """ Returns a new engine instance from a key

        :param solver_agent:
        :param model:
        :param context:
        :return
        """
        has_cplex = env.has_cplex

        # store additional parameters here
        kwargs = {}

        if has_cplex:
            if not cls.cplex_engine_type:
                from docplex.mp.cplex_engine import CplexEngine

                cls.cplex_engine_type = CplexEngine
                cls._engine_types_by_key["cplex"] = CplexEngine
        elif context.solver.docloud.has_credentials():
            model.warning("CPLEX DLL not found, will solve on DOcloud")

        if context.solver.docloud.has_credentials():
            kwargs['docloud_context'] = context.solver.docloud

        # ---
        # what logic do we have:
        if has_cplex:
            # default is CPLEX if we have it
            engine_type = cls._get_engine_by_code(solver_agent, cls.cplex_engine_type)
        elif context.solver.docloud.has_credentials():
            # default is docloud
            engine_type = cls._get_engine_by_code(solver_agent, DOcloudEngine)

        else:
            # no CPLEX, no credentials
            # model.trace("CPLEX DLL not found and model has no DOcloud credentials. "
            #               "Credentials are required at solve time")
            engine_type = NoSolveEngine

        if not engine_type:
            model.fatal("Internal error, cannot build engine from spec {}", solver_agent)

        return engine_type(model, **kwargs)

