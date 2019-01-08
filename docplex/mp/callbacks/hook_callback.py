import cplex.callbacks as cpx_cb
# !/usr/bin/python
# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2017
# ---------------------------------------------------------------------------
#

import cplex._internal._constants as cpxcst

from docplex.mp.callbacks.cb_mixin import ModelCallbackMixin


class HookIncumbentCallback(ModelCallbackMixin, cpx_cb.IncumbentCallback):

    incumbent_sources = {
        cpxcst.CPX_CALLBACK_MIP_INCUMBENT_NODESOLN: 'node',
        cpxcst.CPX_CALLBACK_MIP_INCUMBENT_HEURSOLN: 'heuristic',
        cpxcst.CPX_CALLBACK_MIP_INCUMBENT_USERSOLN: 'user',
        cpxcst.CPX_CALLBACK_MIP_INCUMBENT_MIPSTART: 'mipstart'
    }

    def __init__(self, env):
        # non public...
        cpx_cb.IncumbentCallback.__init__(self, env)
        ModelCallbackMixin.__init__(self)
        self.nb_incumbents = 0
        self.hook_fn = None

    def reset(self):
        self.nb_incumbents = 0

    def get_source_name(self):
        src = self.get_solution_source()
        return self.incumbent_sources.get(src, ' unknown???')

    def __call__(self):
        self.nb_incumbents += 1
        incumbent_name = "{0}#{1}".format(self.model.name, self.nb_incumbents)
        hook_fn = self.hook_fn
        if hook_fn is not None:
            sol = self.make_solution_from_values(name=incumbent_name)  # taken from mixin
            hook_fn(sol)
