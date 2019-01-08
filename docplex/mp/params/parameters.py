# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

from __future__ import print_function

from six import iteritems

from docplex.mp.utils import is_int, StringIO, is_number
from docplex.mp.error_handler import docplex_fatal
from docplex.mp.utils import RedirectedOutputContext


class ParameterGroup(object):
    """ A group of parameters.

    Note:
        This class is not meant to be instantiated by users. Models come
        with a full hierarchy of parameters with groups as nodes.

    """

    def __init__(self, name, parent_group=None):
        self._name = name
        self._parent = parent_group
        self._params = []
        self._subgroups = []
        if parent_group:
            parent_group._add_subgroup(self)

    def to_string(self):
        return "group<%s>" % self.qualified_name()

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return "docplex.mp.params.ParameterGroup({0})".format(self.qualified_name())

    def clone(self):
        """
        Returns:
           A deep copy of the parameter group.
        """
        from copy import deepcopy
        return deepcopy(self)
    
    def copy(self):
        return self.clone()

    @property
    def name(self):
        """ This property returns the name of the group.

        Note:
            Parameter group names are always lowercase.
        """
        return self._name

    def iter_params(self):
        """ Iterates over the group's own parameters.

        Returns:
            An iterator over the group's parameters, non-recursively.
        """
        return iter(self._params)

    @property
    def number_of_params(self):
        """  This property returns the number of parameters in the group, not including subgroups.
        """
        return len(self._params)

    def total_number_of_params(self):
        """
        Includes all parameters of subgroups recursively.

        Returns:
           The total number of parameters inside the group.
        """
        subparams = sum(g.total_number_of_params() for g in self._subgroups)
        return subparams + self.number_of_params

    def total_number_of_groups(self):
        subgroups = sum(g.total_number_of_groups() for g in self._subgroups)
        return subgroups + 1

    @property
    def number_of_subgroups(self):
        """ This property returns the number of subgroups of the group, non-recursively.

        """
        return len(self._subgroups)

    def iter_subgroups(self):
        return iter(self._subgroups)

    @property
    def parent_group(self):
        """ This property returns the parent group (An instance of :class:`ParameterGroup`), or None for the root group.
        """
        return self._parent

    def _add_param(self, param):
        # internal
        self._params.append(param)

    def _add_subgroup(self, subgroup):
        # internal
        self._subgroups.append(subgroup)

    def is_root(self):
        """ Checks whether the group is the root group, in other words, has no parent group.

        Returns:
            True if the group is the root group.
        """
        return self._parent is None

    def root_group(self):
        group = self
        while not group.is_root():
            group = group.parent_group
        return group

    def qualified_name(self, sep=".", upto_root=True):
        """ Computes a string with all the parents of the parameters.

        Example:
            `parameter mip.mip_cuts.Cover` returns "mip.mip_cuts.Covers".

        Args:
            sep (str): The separator string. Default is ".".

        Returns:
            A string representation of the parameter hierarchy.
        """
        self_parent = self._parent
        if not self_parent:
            return self.name
        if not upto_root and self_parent.is_root():
            return self.name
        else:
            return "".join([self._parent.qualified_name(sep=sep, upto_root=upto_root), sep, self.name])

    def prettyprint(self, indent=0):
        tab = indent * 4 * " "
        print("{0}{1!s}={{".format(tab, self.qualified_name()))
        for p in self.iter_params():
            print("{0}    {1!s}".format(tab, p))
        for sg in self.iter_subgroups():
            assert isinstance(sg, ParameterGroup)
            sg.prettyprint(indent + 1)
        print("{0}}}".format(tab))

    def _update_self_dict(self, extra_dict, do_check=True):
        self_dict = self.__dict__
        if do_check:
            # new entries should not already be present in self.dict
            for k in extra_dict:
                if k in self_dict:
                    # should not happen
                    print("!! update_self_dict: name collision with: %s" % k)  # pragma : no cover
        self_dict.update(extra_dict)

    @staticmethod
    def make(name, param_dict_fn, subgroup_fn, parent=None):
        # INTERNAL
        # factory method to create one group from:
        # 1. a lambda function taking a group as argument and returning a dict of name: param instances
        # 2. a dict of subgroup name: subgroup_make functions
        # 3. a possibly-None parent group. If None, we are at root.
        group = ParameterGroup(name, parent) if parent else RootParameterGroup(name, cplex_version=None)
        group._initialize(param_dict_fn, subgroup_fn)
        return group

    def _initialize(self, param_dict_fn, subgroup_fn):
        param_dict = param_dict_fn(self)
        self._update_self_dict(param_dict)
        if subgroup_fn:
            subgroup_fn_dict = subgroup_fn()
            subgroup_dict = {group_name: group_fn(self)
                             for group_name, group_fn in iteritems(subgroup_fn_dict)}
            self._update_self_dict(subgroup_dict)

    def number_of_nondefaults(self):
        return sum(1 for _ in self.generate_nondefault_params())

    def has_nondefaults(self):
        for _ in self.generate_nondefault_params():
            return True
        else:
            return False

    def reset(self, recursive=False):
        """ Resets all parameters in the group.

        Args:
            recursive (bool): If True, also resets the subgroups.
        """
        for p in self.iter_params():
            p.reset()
        if recursive:
            for g in self.iter_subgroups():
                g.reset(recursive=True)

    def reset_all(self):
        self.reset(recursive=True)

    def generate_params(self):
        """  Returns a generator traversing all parameters.

        The generator yields all parameters from the group
        and also from its subgroups, recursively.
        Called from the root parameter group, returns all parameters.

        Returns:
            A generator object.

        """
        return self._generate_and_filter_params(predicate=None)

    def _generate_and_filter_params(self, predicate):
        """ A filtering generator function that traverses a group's parameters.

        This generator function traverses the group and its subgroup tree,
        yielding only  parameters that are accepted by th epredicate.

        Args:
           predicate: A function that takes one parameter as asrgument.
           The return value of this function will be interpeted as a boolean using
           Python conversionb rules.

        Returns:
            A generator object.
        """
        for p in self.iter_params():
            if predicate is None or predicate(p):
                yield p
        # now recurse
        for sg in self.iter_subgroups():
            for nd in sg._generate_and_filter_params(predicate):
                yield nd

    def generate_nondefault_params(self):
        """ A generator function that returns all non-default parameters.

        This generator function traverses the group and its subgroup tree,
        yielding those parameters with a non-default value, one at a time.
        A parameter is non-default as soon as its value differs from the default.

        Returns:
            A generator object.
        """
        return self._generate_and_filter_params(predicate=lambda p: p.is_nondefault())

    def generate_all_subgroups(self):
        # INTERNAL
        for sg in self.iter_subgroups():
            yield sg
            for ssg in sg.generate_all_subgroups():
                yield ssg

    def __setattr__(self, attr_name, value):
        if attr_name.startswith("_"):
            self.__dict__[attr_name] = value
        elif hasattr(self, attr_name):
            attr = getattr(self, attr_name)
            if isinstance(attr, Parameter):
                # attribute is set inside param, not necesarily in engine...
                attr.set(value)
            else:
                docplex_fatal("No parameter with name {0} in {1}", attr_name, self.qualified_name())
        else:
            docplex_fatal("No parameter with name {0} in {1}", attr_name, self.qualified_name())

    @property
    def cplex_version(self):
        return self.root_group().cplex_version


class Parameter(object):
    """ Base class for all parameters.

    This class is not meant to be instantiated, but subclassed.

    """

    # noinspection PyProtectedMember
    def __init__(self, group, short_name, cpx_name, param_key, description, default_value):
        assert isinstance(group, ParameterGroup)
        self._parent = group
        self._short_name = short_name
        self._cpx_name = cpx_name
        self._id = param_key
        self._description = description
        self._default_value = default_value
        # current = default at start...
        self._current_value = default_value
        # link to parent group
        group._add_param(self)

    @property
    def short_name(self):
        return self._short_name

    @property
    def qualified_name(self, sep='.'):
        return "%s.%s" % (self._parent.qualified_name(sep=sep), self.short_name)

    @property
    def cpx_name(self):
        return self._cpx_name

    @property
    def cpx_id(self):
        return self._id

    @property
    def description(self):
        return self._description

    @property
    def default_value(self):
        return self._default_value

    def reset_default_value(self, new_default):
        # INTERNAL: use with caution
        self._default_value = new_default  # pragma: no cover
        self._current_value = new_default  # pragma: no cover

    @property
    def current_value(self):
        return self._current_value

    def accept_value(self, new_value):
        """ Checks if `new_value` is an accepted value for the parameter.

        Args:
            new_value: The candidate value.

        Returns:
            True if acceptable, else False.

        """
        raise NotImplementedError()  # pragma: no cover

    def transform_value(self, raw_value):
        return raw_value

    def check_value(self, raw_value):
        if raw_value == self.default_value:
            return raw_value
        elif not self.accept_value(raw_value):
            docplex_fatal("Value {0!s} of type {2} is invalid for parameter {1}",
                          raw_value, self.qualified_name, type(raw_value))
        else:
            return self.transform_value(raw_value)

    def set(self, new_value):
        """ Changes the value of the parameter to `new_value`.

        This method checks that the new value is valid.
        If not valid, an exception is raised.

        Args:
            new_value: The new value for the parameter.
        """
        accepted_value = self.check_value(new_value)
        if accepted_value is not None:
            self._current_value = accepted_value
        return accepted_value

    def get(self):
        """
        Returns:
           The current value of the parameter.
        """
        return self._current_value

    def reset(self):
        """ Resets the parameter value to its default.
        """
        self._current_value = self.default_value

    def is_nondefault(self):
        """ Checks if the current value of the parameter does not equal its default.

        Returns:
           True if the current value of the parameter does not equal its default.

        """
        return self.get() != self._default_value

    def is_default(self):
        """  Checks if the current value of the parameter equals its default.

        Returns:
            True if the current value of the parameter equals its default.
        """
        return self.get() == self.default_value

    @staticmethod
    def _is_in_range(arg, range_min, range_max):
        if range_min is not None and arg < range_min:
            return False
        if range_max is not None and arg > range_max:
            return False
        return True

    def to_string(self):
        return "{0}:{1:s}(2!s)".format(self._short_name, self.type_name(), self._current_value)

    def __str__(self):
        return self.to_string()

    def is_numeric(self):
        return False  # pragma: no cover

    def type_name(self):
        raise NotImplementedError  # pragma: no cover

    def _root_group(self):
        return self._parent.root_group()

    def _repr_classname(self):
        return "docplex.mp.params.{0}".format(self.__class__.__name__)

    def __repr__(self):
        return "{0}({1},{2!s})".format(self._repr_classname(), self.qualified_name, self._current_value)


_BOOLEAN_STATES = {'1': True, 'yes': True, 'true': True, 'on': True,
                   '0': False, 'no': False, 'false': False, 'off': False}


class BoolParameter(Parameter):
    def __init__(self, group, short_name, cpx_name, param_key, description, default_value):
        Parameter.__init__(self, group, short_name, cpx_name, param_key, description, default_value)

    def transform_value(self, new_value):
        svalue = str(new_value).lower()
        if new_value in {True, False}:
            return new_value
        elif new_value in {0, 1}:
            return True if new_value else False
        elif svalue in _BOOLEAN_STATES:
            return _BOOLEAN_STATES[svalue]
        else:
            return None

    def accept_value(self, value):
        return value in {0, 1} or str(value).lower() in _BOOLEAN_STATES or value in {True, False}

    def type_name(self):
        return "bool"


class StrParameter(Parameter):
    def __init__(self, group, short_name, cpx_name, param_key, description, default_value):
        Parameter.__init__(self, group, short_name, cpx_name, param_key, description, default_value)
        assert isinstance(default_value, str)

    def accept_value(self, new_value):
        return isinstance(new_value, str)

    def type_name(self):
        return "string"


class IntParameter(Parameter):
    def accept_value(self, new_value):
        ivalue = int(new_value)
        return is_int(ivalue) and self._is_in_range(ivalue, self._min_value, self._max_value)

    def is_numeric(self):
        return True  # pragma: no cover

    def _get_min_value(self):
        return self._min_value  # pragma: no cover

    def _get_max_value(self):
        return self._max_value  # pragma: no cover

    def __init__(self, group, short_name, cpx_name, param_key, description, default_value, min_value=None,
                 max_value=None):
        Parameter.__init__(self, group, short_name, cpx_name, param_key, description, default_value)
        self._min_value = min_value
        self._max_value = max_value

    def type_name(self):
        return "int"

    def __repr__(self):
        return "{0}({1},{2!s})".format(self._repr_classname(), self.qualified_name, self._current_value)

    def transform_value(self, new_value):
        if is_int(new_value):
            return new_value
        else:
            return int(new_value)


class PositiveIntParameter(IntParameter):
    def __init__(self, group, short_name, cpx_name, param_key, description, default_value, max_value=None):
        IntParameter.__init__(self, group, short_name, cpx_name, param_key, description, default_value, min_value=0,
                              max_value=max_value)

    def type_name(self):
        return "count"


class NumParameter(Parameter):
    """ A numeric parameter can take any floating-point value, inside a range of `[min,max]` values.
    """

    def __init__(self, group, short_name, cpx_name, param_key, description, default_value, min_value=None,
                 max_value=None):
        Parameter.__init__(self, group, short_name, cpx_name, param_key, description, default_value)
        self._min_value = min_value
        self._max_value = max_value

    def is_numeric(self):
        return True  # pragma: no cover

    def _get_min_value(self):
        return self._min_value  # pragma: no cover

    def _get_max_value(self):
        return self._max_value  # pragma: no cover

    def accept_value(self, new_value):
        fvalue = float(new_value)
        return self._is_in_range(fvalue, self._min_value, self._max_value)

    def transform_value(self, new_value):
        if is_number(new_value):
            return new_value
        else:
            return float(new_value)

    def type_name(self):
        return "num"

# a dictionary of formats for each type.
_param_prm_formats = {NumParameter: "%.14f",
                      IntParameter: "%d",
                      PositiveIntParameter: "%d",
                      BoolParameter: "%d",
                      StrParameter: "\"%s\""  # need quotes
                      }


class RootParameterGroup(ParameterGroup):
    """ The root parameter group (there should be only one instance at the root of the tree).
    """

    def __init__(self, name, cplex_version):
        ParameterGroup.__init__(self, name)
        self._cplex_version = cplex_version

    def qualified_name(self, sep=".", upto_root=True):
        return self.name

    @property
    def cplex_version(self):
        return self._cplex_version

    def is_root(self):
        return True

    def export_prm(self, output, overload_params=None):
        """
        Exports parameters to an output stream in PRM format.

        This method writes non-default parameters in CPLEX PRM syntax.
        In addition to non-default parameters, some parameters can be forced to
        be printed with a specific value by passing a dictionary with
        Parameter objects as keys and values as arguments.
        These values are used in the print operation, but will not be kept,
        and the values of parameters will not be changed.
        Passing `None` as `overload_params` will disable this functionality, and
        only non-default parameters are printed.

        Args:
            output: The output stream, typically a filename.

            overload_params: A dictionary of overloaded values for
                certain parameters. This dictionary is of the form {param: value}
                the printed PRM file will use overloaded values
                for those parameters present in the dictionary.

        """
        with RedirectedOutputContext(output):
            cplex_version_string = self._cplex_version
            print("# -- This content is generated by DOcplex")
            print("CPLEX Parameter File Version %s" % cplex_version_string)

            param_generator = self.generate_params()
            for param in param_generator:
                if overload_params and param in overload_params:
                    param_value = overload_params[param]
                else:
                    param_value = param.get()

                if param_value != param.default_value:
                    print("{0:<33}".format(param.cpx_name), end="")
                    print(_param_prm_formats[type(param)] % param_value)

            print("# --- end of generated prm data ---")

    def print_information(self, indent_level=0, print_all=False):
        self.print_info_to_stream(output=None, overload_params=None, indent_level=indent_level, print_defaults=print_all)

    def print_info_to_stream(self, output, overload_params=None, print_defaults=False, indent_level=0):
        """ Writes parameters to an output stream.

        This method writes non-default parameters in a human readable syntax.
        In addition to non-default parameters, some parameters can be forced to
        be printed with a specific value by passing a dictionary with
        Parameter objects as keys and values as arguments.
        These values are used in the print operation but not be kept,
        and the values of parameters will not be changed.
        Passing `None` as `overload_params` will disable this functionality, and
        only non-default parameters are printed.

        Args:
            output: The output stream.
            overload_params: A dictionary of overloaded values for
                certain parameters. This dictionary is of the form {param: value}.
        """
        indent = " " * indent_level
        with RedirectedOutputContext(output):
            param_generator = self.generate_params()
            for param in param_generator:
                if overload_params and param in overload_params:
                    param_value = overload_params[param]
                elif print_defaults or param.is_nondefault():
                    param_value = param.get()
                else:
                    param_value = None
                if param_value is not None:
                    print("{0}{1} = {2!s}"
                          .format(indent,
                                  param.qualified_name,
                                  _param_prm_formats[type(param)] % param_value))

    def export_prm_to_string(self, overload_params=None):
        """  Exports non-default parameters in PRM format to a string.

        The logic of overload is the same as in :func:`export_prm`.
        A parameter is written if either it is a key on `overload_params`,
        or it has a non-default value.
        This allows merging non-default parameters with temporary parameter values.


        Args:
            overload_params: A dictionary of overloaded values, possibly None.

        Note:
            This method causes no side effects on the parameters.

        See Also:
            :func:`export_prm`

        Returns:
            A string, in CPLEX PRM format.
        """
        oss = StringIO()
        self.export_prm(oss, overload_params)
        return oss.getvalue()

    def print_info_to_string(self, overload_params=None, print_defaults=False):
        """  Writes parameters in readable format to a string.

        The logic of overload is the same as in :func:`export_prm`.
        A parameter is written if either it is a key on `overload_params`,
        or it has a non-default value.
        This allows merging non-default params with temporary parameter values.

        Args:
            overload_params: A dictionary of overloaded values, possibly None.

        Note:
            This method causes no side effects on the parameters.

        See Also:
            :func:`write_parameters`

        Returns:
            A string.
        """
        oss = StringIO()
        self.print_info_to_stream(oss, print_defaults=print_defaults, overload_params=overload_params)
        return oss.getvalue()

    @staticmethod
    def make(name, param_dict_fn, subgroup_fn, cplex_version):
        # INTERNAL
        # factory method to create one group from:
        # 1. a lambda function taking a group as argument and returning a dict of name: param instances
        # 2. a dict of subgroup name: subgroup_make functions
        # 3. a possibly-None parent group. If None, we are at root.
        root_group = RootParameterGroup(name, cplex_version)
        root_group._initialize(param_dict_fn, subgroup_fn)
        return root_group

    def prettyprint(self, indent=0):
        print("* CPLEX parameters version: {0}".format(self.cplex_version))
        ParameterGroup.prettyprint(self, indent)

    def __repr__(self):
        return "docplex.mp.params.RootParameterGroup(%s)" % self.cplex_version

        # def qualified_name(self, sep='.'):
        # return "parameters"

    def as_dict(self):
        # INTERNAL: returns a dictionary of qualified name -> parameter
        qdict = {p.qualified_name: p for p in self}
        return qdict

    def __iter__(self):
        for p in self.iter_params():
            yield p
        # now recurse
        for sg in self.iter_subgroups():
            for nd in sg._generate_and_filter_params(predicate=None):
                yield nd

    def update(self, other_params):
        if not isinstance(other_params, RootParameterGroup):
            docplex_fatal("Parameter.update expects  RootParameterGroup, got: {0!s}", other_params)
        elif self._cplex_version != other_params.cplex_version:
            docplex_fatal("Parameter.update expectes same cple version, self: {0}, other: {1}",
                          format(self.cplex_version, other_params.cplex_version))
        else:
            self_qdict = self.as_dict()
            nb_updates = 0
            for other_param in other_params:
                self_param = self_qdict[other_param.qualified_name]
                other_current = other_param.get()
                if self_param.get() != other_current:
                    self_param.set(other_current)
                    nb_updates += 1
            return nb_updates
