# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

from six import iteritems
from collections import defaultdict

from docplex.mp.basic import Priority
from docplex.mp.linear import AbstractConstraint
from docplex.mp.error_handler import docplex_fatal
from docplex.mp.solution import SolveSolution
from docplex.mp.sdetails import SolveDetails

# gendoc: ignore


def _match_priority_name(prio, s, sep='_'):
    if not s:
        return False  # pragma : no cover

    s_lower = s.lower()
    prio_name = prio.name.lower()
    return s_lower.find(prio_name) >= 0


class IConstraintPrioritizer(object):
    ''' Abstract Interface for the prioritizer.
        This class is a functor to be called on each model constraint.
    '''

    def get_priority(self, ct):
        raise NotImplementedError("base class")  # pragma: no cover

    def __call__(self, ct):
        return self.get_priority(ct)


class IRelaxationListener(object):
    ''' Base class for relaxation listeners.'''

    def notify_start_relaxation(self, priority, relaxables):
        ''' This method is called at each step of the relaxation loop.'''
        pass  # pragma: no cover

    def notify_failed_relaxation(self, priority, relaxables):
        ''' This method is called when a relaxation attempt fails.'''
        pass  # pragma: no cover

    def notify_successful_relaxation(self, priority, relaxables, relaxed_obj_value, violations):
        ''' This method is called when a relaxation succeeds.'''
        pass  # pragma: no cover


class DummyRelaxationListener(IRelaxationListener):
    ''' A do-nothing listener (better than None).'''

    def notify_start_relaxation(self, priority, relaxables):
        pass  # pragma: no cover

    def notify_failed_relaxation(self, priority, relaxables):
        pass  # pragma: no cover

    def notify_successful_relaxation(self, priority, relaxables, relaxed_obj_value, violations):
        pass  # pragma: no cover


class DebugRelaxationListener(IRelaxationListener):
    ''' A default implementation of the listener, which prints messages.'''

    def __init__(self):
        IRelaxationListener.__init__(self)
        self.relaxation_count = 0

    def notify_start_relaxation(self, priority, relaxables):
        self.relaxation_count += 1
        print("-> relaxation #{0} starts with priority: {1!s}, #relaxables={2:d}"
              .format(self.relaxation_count, priority, len(relaxables)))

    def notify_failed_relaxation(self, priority, relaxables):
        print("<- relaxation #{0} fails, priority: {1!s}, #relaxables={2:d}"
              .format(self.relaxation_count, priority, len(relaxables)))

    def notify_successful_relaxation(self, priority, relaxables, obj, violations):
        print("<- relaxation #{0} succeeds: priority: {1!s}, #relaxables={2:d}, obj={3}, #violations={4}".
              format(self.relaxation_count, priority, len(relaxables), obj, len(violations)))


class DefaultPrioritizer(IConstraintPrioritizer):
    """ Basic prioritizer relaxes any constraint with a name.

        More precisely, the prioritizer logic works as follows:

            - If the constraint has a "property" attribute, then it is assumed
              to hold a priority instance, use it.
            - If the constraint has a user-defined name, relax it with MEDIUM priority.
            - Or if the constraint has not been given a name by the user,
                relax it if the relax_unnamed flag is True, again with MEDIUM priority.
    """

    def __init__(self, relax_unnamed=True, used_priority=Priority.MEDIUM):
        self._relax_unnamed = relax_unnamed
        self._used_priority = used_priority

    def get_priority(self, ct):
        if hasattr(ct, "priority"):
            return ct.priority
        elif ct.has_user_name():
            if _match_priority_name(Priority.MANDATORY, ct.name):
                return Priority.MANDATORY
            else:
                return self._used_priority
        elif self._relax_unnamed:
            return self._used_priority
        else:
            return Priority.MANDATORY


class NamePrioritizer(IConstraintPrioritizer):
    """ Constraint prioritizer based on constraint names.

        This proiritizer analyzes constraint names for strings that match priority names.
        If a constraint contains a string which matches a priority name,
        then it is assigned this priority.

        For example: a constraint named "ct_salary_low will be considered as having priority "low"
    """

    def __init__(self,
                 priority_for_anonymous=Priority.MANDATORY,
                 priority_for_non_matches=Priority.MANDATORY):
        assert isinstance(priority_for_anonymous, Priority)
        assert isinstance(priority_for_non_matches, Priority)
        self.priorities = Priority.all_sorted()
        self.priority_for_unnamed_cts = priority_for_anonymous
        self.priority_for_non_matching_cts = priority_for_non_matches
        self.priorityBySymbol = {prio.name.lower(): prio for prio in self.priorities}

    def get_priority(self, ct):
        ''' Looks for known priority names inside constraint names.
        TODO: add a separator after or before to the matching, e.g. <name>[a-zA-Z]~.
        '''
        ctname = ct.name
        if not ctname or ct.has_automatic_name():
            return self.priority_for_unnamed_cts
        else:
            ctname_lower = ctname.lower()
            best_matched = 0
            best_prio = self.priority_for_non_matching_cts
            for (prio_symbol, prio) in iteritems(self.priorityBySymbol):
                if ctname_lower.find(prio_symbol) >= 0:
                    matched = len(prio_symbol)
                    # longer matches are preferred
                    # e.g. very_low and low both match in very_low_ctxxx
                    # but the prioritizer will return very_low as the match is longer.
                    if matched > best_matched:
                        best_matched = matched
                        best_prio = prio
            return best_prio


class CustomPrioritizer(IConstraintPrioritizer):
    """
    Constraint prioritizer based on a dictionary constraint -> priority.

    Inituialized from a dictionary and an optional default priority.

    Args:
        priority_mapping: a dict, with constraints as keys and priorities as values.

        default_priority: an optional priority, used when a constraint is not explicitly mentioned
            in the mapping. The default value is MANDATORY, meaning that any constraint not mentioned
            in the mapping will not be relaxed.
    """
    def __init__(self, priority_mapping, default_priority=Priority.MANDATORY):
        # --- typecheck that this dict is a a {ct: prio} mapping.
        if not isinstance(priority_mapping, dict):
            raise TypeError
        for k, v in iteritems(priority_mapping):
            if not isinstance(k, AbstractConstraint):
                raise TypeError
            if not isinstance(v, Priority):
                raise TypeError
        # ---

        self._mapping = priority_mapping
        self._default_priority = default_priority

    def get_priority(self, ct):
        return self._mapping.get(ct, self._default_priority)


class Relaxer(object):
    ''' This class is an asbtract algorithm, in the sense that it operates on interfaces.
    
        It takes:
          - a prioritizer, an implementation of IConstraintPrioritizer,
          - a threshold above which a relaxation is deemed valid, 
          - a flag to indicate whether or not relaxable constraints are
            accumulated. If yes, the set of relaxables keeps increasing while priorities 
            are examined; otherwise, the algorithm attemps to relax each level, 
            and only constraints of this level, at a time.
    '''
    default_min_relaxed = 1e-5

    def __init__(self, prioritizer='default',
                 min_relaxed=default_min_relaxed,
                 cumulative=True,
                 verbose=False,
                 **kwargs):
        if min_relaxed <= 0:
            print("Warning: min_relaxed should be > 0, got: {0:g}, using {1:g}"
                  .format(min_relaxed, self.default_min_relaxed))
            min_relaxed = self.default_min_relaxed
        self._min_relaxed = min_relaxed
        # ---
        if isinstance(prioritizer, IConstraintPrioritizer):
            self.__prioritizer = prioritizer
        elif prioritizer == 'name':
            self.__prioritizer = NamePrioritizer()
        elif isinstance(prioritizer, dict):
            self.__prioritizer = CustomPrioritizer(priority_mapping=prioritizer)
        elif prioritizer is None or prioritizer == 'default':
            relax_unnamed = kwargs.get("relax_unnamed", True)
            self.__prioritizer = DefaultPrioritizer(relax_unnamed=relax_unnamed)
        else:
            print("Cannot deduce a prioritizer from: {0!r} - expecting \"name\"|\"default\"| dict", prioritizer)
            raise TypeError

        self._ordered_priorities = Priority.all_sorted()
        self._cumulative = cumulative
        self._verbose = verbose
        self._listeners = []

        # result data
        self._last_relaxation_status = False
        self._last_relaxation_objective = -1e+75
        self._last_successful_relaxed_priority = Priority.MANDATORY
        self._last_relaxation_details = SolveDetails.make_dummy()
        self._relaxations = {}
        if self._verbose:
            self.add_listener(DebugRelaxationListener())

    def _check_successful_relaxation(self):
        if not self._last_relaxation_status:
            docplex_fatal("No relaxed solution is present")

    def reset(self):
        self._last_relaxation_status = False
        self._last_relaxation_objective = -1e+75
        self._last_successful_relaxed_priority = Priority.MANDATORY
        self._relaxations = {}

    def _accept_violation(self, violation):
        ''' The filter method which accepts or rejects a violation.'''
        return 0 == self._min_relaxed or abs(violation) >= self._min_relaxed

    def add_listener(self, listener):
        """ Adds a relaxation listener.

        Args:
            listener: The new listener to add. If ``listener`` is not an
               instance of ``IRelaxationListener``, it is ignored.

        See Also:
            IRelaxationListener
        """
        if isinstance(listener, IRelaxationListener):
            self._listeners.append(listener)

    def remove_listener(self, listener):
        """ Removes a relaxation listener.

        Args:
            listener: The listener to remove.
        """
        if isinstance(listener, IRelaxationListener) and listener in self._listeners:
            self._listeners.remove(listener)

    def clear_listeners(self):
        """ Removes all relaxation listeners.
        """
        self._listeners = []

    def parse_kwargs(self, **kwargs):
        # defaults: 2% gap, no sol limit, no time limit
        relax_gap = 0.01
        relax_nb_sols = -1
        pass_time_limit = -1

        for argname, argval in iteritems(kwargs):
            if argname == "gap":
                relax_gap = argval
            elif argname == "max_nb_sol":
                relax_nb_sols = argval
            elif argname == "pass_time_limit":
                pass_time_limit = argval
            else:
                print("relax does not recognize kwarg {0}={1!s} - ignored".format(argname, argval))
        return relax_gap, relax_nb_sols, pass_time_limit

    def relax(self, mdl, **kwargs):
        """ Runs the relaxation loop.

        Args:
            mdl: The model to be relaxed.
            kwargs: accepts named arguments similar to solve.

        Returns:
            If the relaxation succeeds, returns a solution object, an instance of SolveSolution; otherwise returns None.
        """

        relax_gap = kwargs.get("relax_gap", 0.01)  # default gap is 1% ???
        max_nb_sol = kwargs.get("max_nb_sol", -1)
        pass_time_limit = kwargs.get("pass_time_limit", -1)
        relaxation_limits = (relax_gap, max_nb_sol, pass_time_limit)
        # filter out those kwargs reserved for relax
        solve_kwargs = {k: v for k, v in iteritems(kwargs) if k not in ["relax_gap", "max_nb_sol", "pass_time_limit"]}
        self.reset()

        # 1. build a dir {priority : cts}
        priority_map = defaultdict(list)
        nb_prioritized_cts = 0
        for ct in mdl.iter_constraints():
            prio = self.__prioritizer.get_priority(ct)
            if not prio.is_mandatory():
                priority_map[prio].append(ct)
                nb_prioritized_cts += 1

        if 0 == nb_prioritized_cts:
            mdl.error("Relaxation algorithm found no relaxable constraints - exiting")
            return None

        # print("Total number of relaxable cts=%d, #levels=%d"%(nb_prioritized_cts, len(priorityMap)))

        # relaxation loop
        all_groups = []
        all_relaxable_cts = []
        is_cumulative = self._cumulative
        relax_ok = False
        engine = mdl.get_engine()
        is_model_optimized = mdl.is_optimized()

        # save this for restore later
        saved_context_log_output = mdl.context.solver.log_output
        saved_log_output_stream = mdl.get_log_output()
        saved_context = mdl.context

        # take into account local argument overrides
        context = mdl.prepare_actual_context(**solve_kwargs)

        try:
            # mdl.context has been saved in saved_context above
            mdl.context = context
            mdl.set_log_output(mdl.context.solver.log_output)

            # engine parameters, if needed to
            parameters = context.cplex_parameters
            # limit threads if needed
            if getattr(context.solver, 'max_threads', None) is not None:
                if parameters.threads.get() == 0:
                    max_threads = context.solver.max_threads
                else:
                    max_threads = min(context.solver.max_threads,
                                      parameters.threads.get())
                # we don't want to duplicate parameters unnecessary
                if max_threads != parameters.threads.get():
                    parameters = parameters.copy()
                    parameters.threads = max_threads
                    out_stream = context.solver.log_output_as_stream
                    if out_stream:
                        out_stream.write("WARNING: Number of workers has been reduced to %s to comply with platform limitations.\n" % max_threads)


            mdl._apply_parameters_to_engine(parameters)

            for prio in self._ordered_priorities:
                if prio in priority_map:
                    cts = priority_map[prio]
                    if not cts:
                        # this should not happen...
                        continue  # pragma : no cover

                    pref = prio.get_geometric_preference_factor()
                    # build a new group
                    relax_group = [pref, cts]

                    # relaxing new batch of cts:
                    if not is_cumulative:
                        # if not cumulative reset the groupset
                        all_groups = [relax_group]
                        all_relaxable_cts = cts
                    else:
                        all_groups.append(relax_group)
                        all_relaxable_cts += cts

                    # at this stage we have a sequence of groups
                    # a group is itself a sequence of two components
                    # - a preference factor
                    # - a sequence of constraints
                    for l in self._listeners:
                        l.notify_start_relaxation(prio, all_relaxable_cts)
                    try:
                        (relax_ok, relax_obj) = engine.solve_relaxed(mdl, all_groups, is_model_optimized, relaxation_limits)
                    finally:
                        self._last_relaxation_details = engine.get_solve_details()
                    if relax_ok:
                        self._last_successful_relaxed_priority = prio
                        self._last_relaxation_status = True
                        self._last_relaxation_objective = relax_obj
                        # relaxation ok, need to compute raw infeasibilities
                        raw_infeasibilities = engine.get_infeasibilities(all_relaxable_cts)
                        # now filter those real infeasibilities
                        for c in range(len(all_relaxable_cts)):
                            ct = all_relaxable_cts[c]
                            raw_infeas = raw_infeasibilities[c]
                            if self._accept_violation(raw_infeas):
                                self._relaxations[ct] = raw_infeas
    
                        for l in self._listeners:
                            l.notify_successful_relaxation(prio, all_relaxable_cts, relax_obj, self._relaxations)
                        break
                    else:
                        # relaxation has failed, notify the listeners
                        for l in self._listeners:
                            l.notify_failed_relaxation(prio, all_relaxable_cts)
    
            if relax_ok:
                all_indices = [dv.get_index() for dv in mdl.iter_variables()]
                if all_indices:
                    value_by_idx_map = engine.get_solutions(all_indices)
                    value_by_var_map = {dv: value_by_idx_map[dv.index] for dv in mdl.iter_variables()}
                else:
                    value_by_var_map = {}
                relaxed_sol = SolveSolution(mdl,
                                            obj=self._last_relaxation_objective,
                                            var_value_map=value_by_var_map, engine_name="relaxer")
            else:
                relaxed_sol = None
            mdl.notify_solve_relaxed(relaxed_sol, engine.get_solve_details())

        finally:
            # --- restore context, log_output if set.
            if saved_log_output_stream != mdl.get_log_output():
                mdl.set_log_output_as_stream(saved_log_output_stream)
            if saved_context_log_output != mdl.context.solver.log_output:
                mdl.context.solver.log_output = saved_context_log_output
            mdl.context = saved_context

        return relaxed_sol

    def iter_relaxations(self):
        """ Iterates on relaxations.

        Relaxations are built as a dictionary with constraints as keys and numeric violations as values,
        so this iterator returns couples of (ct, violation) pairs.
        """
        self._check_successful_relaxation()
        return iteritems(self._relaxations)

    def relaxations(self):
        """  Returns a dictionary with all relaxed constraints.

        Returns a dictionary where keys are the relaxed constraints,
        and values are the numerical slacks.

        """
        return self._relaxations.copy()

    def get_total_slack(self):
        self._check_successful_relaxation()
        return sum(abs(v) for v in self._relaxations.values())

    def print_information(self):
        self._check_successful_relaxation()
        print("* number of relaxations: {0}".format(len(self._relaxations)))
        for rct, slack in self.iter_relaxations():
            arg = rct.name if rct.has_user_name() else str(rct)
            print(" - relaxed: {0}, with slack: {1}".format(arg, abs(slack)))
        print("* total slack: {0}".format(self.get_total_slack()))

    @property
    def relaxed_objective_value(self):
        """  Returns the objective value of the relaxed solution.

        Raises:
            DOCplexException
                if the relaxation has not been successful.
        """
        self._check_successful_relaxation()
        return self._last_relaxation_objective

    @property
    def number_of_relaxations(self):
        """ Returns the number of relaxations found.
        """
        return len(self._relaxations)

    def get_relaxation(self, ct):
        """ Returns the relaxation found for this constraint.

        Args:
            ct: The constraint for which we want the violation.

        Returns:
            The amount by which the constraint has been violated by the relaxer.
            The method returns 0 if the constraint has not been relaxed.
        """
        self._check_successful_relaxation()
        return self._relaxations[ct] if ct in self._relaxations else 0
