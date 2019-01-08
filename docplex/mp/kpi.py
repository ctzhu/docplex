# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


class KPI(object):
    """ Abstract class for key performance indicators (KPIs).

    Each KPI has a unique name. A KPI is attached to a model instance and can compute a numerical value,
    using the :func:`compute` method.

    Some KPIs require a valid solution of the model, while others do not. Use :func:`requires_solution` to
    check whether a given KPI requires a solution.
    """

    def __init__(self):
        pass

    def get_name(self):
        """
        Returns:
            string: The published name of the KPI, a non-empty, unique string.
        """
        raise NotImplementedError  # pragma: no cover

    def get_model(self):
        """
        Returns:
           The model instance on which the KPI is defined.
        :rtype: :class:`docplex.mp.model.Model`
        """
        raise NotImplementedError   # pragma: no cover

    def compute(self):
        raise NotImplementedError   # pragma: no cover

    def _check_name(self, name_arg):
        if not isinstance(name_arg, str):
            self.get_model().fatal("KPI_name, not a string: {0!s}", (name_arg,))
        elif not name_arg:
            self.get_model().fatal("KPI_name must be non-empty string, got: {0!s}", (name_arg,))
        else:
            pass

    def requires_solution(self):
        """ KPIs based on decision expressions or variables require a successful solution
        to be computed. KPIs based on functions may not require a solution.

        Returns:
           Boolean: True if the KPI requires a valid solution.
        """
        raise NotImplementedError   # pragma: no cover

    def copy(self, new_model, var_map):
        raise NotImplementedError   # pragma: no cover

    def clone(self):
        raise NotImplementedError  # pragma: no cover


class DecisionKPI(KPI):
    """ Specialized class of Key Performance Indicator, based in expressions.

    This subclass is built from a decision variable or a linear expression.
    The compute() method returns the solution value afetr a successful solve()

    """
    def __init__(self, decision_obj, name=None):
        KPI.__init__(self)
        self._dobj = decision_obj.to_linear_expr()
        self._name = name or decision_obj.name
        self._check_name(self._name)

    def get_name(self):
        return self._name

    name = property(get_name)

    def get_model(self):
        return self._dobj.model

    def compute(self):
        """ Recdfintion of the abstract compute() method

        Returns:
            the decision expression solution value.

        Raises:
            evaluating theis KPi raises an exception if the underlying model
            has not been solved successfully.
        """
        return self._dobj.solution_value

    def requires_solution(self):
        return True

    def as_expression(self):
        return self._dobj

    def copy(self, new_model, var_map):
        dobj_copy = self._dobj.copy(new_model, var_map)
        return DecisionKPI(decision_obj=dobj_copy, name=self.name)

    def clone(self):
        return DecisionKPI(self._dobj, self._name)


class FunctionalKPI(KPI):
    # Functional KPIs store a function that takes a model to compute a number
    # Functional KPIs do not require a successful solve.

    def __init__(self, fn, model, name):
        KPI.__init__(self)
        self._name = name
        self._function = fn
        self._model = model
        self._check_name(self._name)

    def get_name(self):
        return self._name

    name = property(get_name)

    def get_model(self):
        return self._model

    def compute(self):
        return self._function(self._model)

    def requires_solution(self):
        return False

    def copy(self, new_model, var_map):
        return FunctionalKPI(fn=self._function, model=self._model, name=self._name)

    def clone(self):
        return FunctionalKPI(fn=self._function, model=self._model, name=self._name)
