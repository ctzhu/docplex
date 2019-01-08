# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore

from docplex.mp.solution import SolveSolution

class ProgressData(object):
    """ A container class to hold data retrived from progress callbacks.

    """

    def __init__(self):
        BIGNUM = 1e+75
        self.has_incumbent = False
        self.current_objective = BIGNUM
        self.best_bound = BIGNUM
        self.mip_gap = BIGNUM
        self.current_nb_nodes = 0
        self.remaining_nb_nodes = 0
        self.time = -1
        self.det_time = -1

    def get_tuple(self):
        return (self.has_incumbent,
                self.current_objective,
                self.best_bound,
                self.mip_gap,
                self.current_nb_nodes,
                self.remaining_nb_nodes,
                self.time,
                self.det_time)


class ProgressListener(object):
    def __init__(self):
        pass

    def requires_solution(self):
        """ Returns True if the listener wants solution information at each intermediate solution.
        The default is False, do not require solution information.
        """
        return False

    def notify_solution(self, s):
        """ Redefine this method to handle an intermediate solution from the callback.
        Args:
            s: solution
        :return:
        """
        pass

    def notify_start(self):
        """ The method called when a solve has been initiated on a model.

        Defaul behavior is to do nothing.
        Put here any code to reinitializae the state of the listener
        """
        pass

    def notify_jobid(self, jobid):
        """ The method called when a model is solved on the cloud and the job
        has been submitted.
        
        This method is not called when solve is using a local engine.
        """
        pass  # pragma: no cover

    def notify_end(self, status, objective):
        """The method called when solve is finished on a model. The status is the solve status from the
        solve() method
        """
        pass  # pragma: no cover

    def notify_progress(self, progress_data):
        """ This method is called from within the solve with a ProgressData instance.

        :param progress_data: an instance of ProgressData containing solver info,
            as called from the CPLEX solver.
        """
        pass  # pragma: no cover


class _IProgressFilter(object):
    def accept(self, pdata):
        raise NotImplementedError  # pragma : no cover

    def reset(self):
        pass


class _ProgressFilterAcceptAll(_IProgressFilter):
    def accept(self, pdata):
        return True


class _ProgressFilter(object):
    # INTERNAL: used to filter calls from CPLEX

    def __init__(self, wait_first_incumbent=True, node_diff=1e+20, relative_diff=1e-2, abs_diff=0.1):
        """ Builds a filter for progress listeners.

        A filter accepts calls from the callback according to the parameters below:

        :param wait_first_incumbent: A boolean indicating whether we skip or not any callback info until the first
            incumbent solution is found.
        :param node_diff: An integer. Accepts the call whenver the increment in the number of visited nodes
            exceeds this limit.
        :param relative_diff: A floating number, used to determine whether the objective or best bound have changed.
            if the relative difference between the last recorded value and the new value is greater than this value, the call is accepted.
        :param abs_diff: A floating number, used to determine whether the objective or best bound have changed.
            if the bsolite  difference between the last recorded value and the new value is greater than this value, the call is accepted.
        """
        self._wait_first_incumbent = wait_first_incumbent
        self._relative_change = relative_diff
        self._abs_change = abs_diff
        self._node_diff = node_diff
        # dynamic
        self._incumbent_count = 0
        self._last_incumbent_obj = None
        self._last_bound = None
        self._last_node = 0

    def reset(self):
        self._incumbent_count = 0
        self._last_incumbent_obj = None
        self._last_bound = None
        self._last_node = 0

    @staticmethod
    def make_from_kwargs(kwargs):
        wait_first_incumbent = kwargs.get("wait_first_incumbent", True)
        node_diff = kwargs.get("node_diff", 1e+20)
        relative_diff = kwargs.get("relative_diff", 1e-2)
        abs_diff = kwargs.get("abs_diff", 0.1)
        return _ProgressFilter(wait_first_incumbent=wait_first_incumbent,
                               node_diff=node_diff,
                               relative_diff=relative_diff,
                               abs_diff=abs_diff)

    def _is_significant_change(self, old_value, new_value):
        abs_diff = abs(new_value - old_value)
        rel_diff = abs_diff / (1.0 + abs(new_value))
        return rel_diff >= self._relative_change or abs_diff >= self._abs_change

    def accept(self, pdata):
        accept = False
        if self._wait_first_incumbent and self._incumbent_count == 0 and not pdata.has_incumbent:
            return False

        if pdata.has_incumbent:
            if (self._last_incumbent_obj is None) or self._is_significant_change(self._last_incumbent_obj,
                                                                                 pdata.current_objective):
                self._last_incumbent_obj = pdata.current_objective
                self._last_bound = pdata.best_bound
                self._last_node = pdata.current_nb_nodes
                accept = True

        if self._last_bound is None or self._is_significant_change(self._last_bound, pdata.best_bound):
            self._last_bound = pdata.best_bound
            self._last_node = pdata.current_nb_nodes
            if pdata.has_incumbent:
                self._last_incumbent_obj = pdata.current_objective
            accept = True

        # nodes
        if self._node_diff > 1:
            if pdata.current_nb_nodes - self._last_node > self._node_diff:
                self._last_node = pdata.current_nb_nodes
                self._last_bound = pdata.best_bound
                if pdata.has_incumbent:
                    self._last_incumbent_obj = pdata.current_objective
                accept = True

        return accept

class TextProgressListener(ProgressListener):
    """ A simple implementation of Progress Listener, which prints messages to stdout
    """

    def __init__(self, filtering=True, gap_fmt=None, obj_fmt=None, **kwargs):
        ProgressListener.__init__(self)
        self._gap_fmt = gap_fmt or "{:.2%}"
        self._obj_fmt = obj_fmt or "{:.4f}"
        self._count = 0
        if filtering:
            self._filter = _ProgressFilter.make_from_kwargs(kwargs)
        else:
            self._filter = _ProgressFilterAcceptAll()

    @property
    def message_count(self):
        return self._count

    def notify_start(self):
        ProgressListener.notify_start(self)
        self._count = 0
        self._filter.reset()

    def notify_progress(self, progress_data):
        if self._filter.accept(progress_data):
            self._count += 1
            pdata_has_incumbent = progress_data.has_incumbent
            incumbent_symbol = '+' if pdata_has_incumbent else ' '
            # if pdata_has_incumbent:
            #     self._incumbent_count += 1
            current_obj = progress_data.current_objective
            if pdata_has_incumbent:
                objs = self._obj_fmt.format(current_obj)
            else:
                objs = "N/A"
            best_bound = progress_data.best_bound
            nb_nodes = progress_data.current_nb_nodes
            remaining_nodes = progress_data.remaining_nb_nodes
            if pdata_has_incumbent:
                gap = self._gap_fmt.format(progress_data.mip_gap)
            else:
                gap = "N/A"
            raw_time = progress_data.time
            rounded_time = round(raw_time, 1)

            print("{0:>3}{7}: Best Integer={1}, Best Bound={2:.4f}, gap={3}, nodes={4}/{5} [{6}s]"
                  .format(self._count, objs, best_bound, gap, nb_nodes, remaining_nodes, rounded_time,
                          incumbent_symbol))


class RecordProgressListener(ProgressListener):

    def __init__(self, filtering=True, **kwargs):
        ProgressListener.__init__(self)
        self.__recorded = []
        self.__final_objective = 1e+75
        self.__final_status = False
        if filtering:
            self._filter = _ProgressFilter.make_from_kwargs(kwargs)
        else:
            self._filter = _ProgressFilterAcceptAll()

    def notify_start(self):
        # restart
        self.__recorded = []
        self.__final_objective = 1e+75
        self._filter.reset()

    def notify_progress(self, progress_data):
        if self._filter.accept(progress_data):
            self.__recorded.append(progress_data.get_tuple())

    def notify_end(self, status, objective):
        """ The method called when solve is finished on a model. The status is the solve status from the
        solve() method
        """
        self.__final_status = status
        if status:
            self.__final_objective = objective

    @property
    def final_objective(self):
        return self.__final_objective

    @property
    def final_status(self):
        return self.__final_status

    @property
    def number_of_records(self):
        return len(self.__recorded)

    def iter_progress_data(self):
        return iter(self.__recorded)


class SolutionListener(ProgressListener):

    def __init__(self, model):
        ProgressListener.__init__(self)
        self._model = model
        self._engine_name = model.get_engine().name
        self._current_solution = None
        self._current_objective = 1e+75 # bof

    def requires_solution(self):
        # this class of listeneres requires solution information
        return True

    def notify_progress(self, progress_data):
        if progress_data.has_incumbent:
            self._current_objective = progress_data.current_objective

    def notify_solution(self, incumbents):
        # create a new instance and replace current solution
        # check performance impact
        sol = SolveSolution(self._model, obj=self._current_objective, engine_name=self._engine_name)
        for v in self._model.iter_variables():
            # incumbent values are provided as a list with indices as positions.
            incumbent_value = incumbents[v._index]
            if 0 != incumbent_value:
                # silently round discrete values, just as with engine solutions.
                sol._set_var_value_internal(v, incumbent_value, rounding=True, do_warn_on_non_discrete=False)
        self._current_solution = sol

    @property
    def current_solution(self):
        return self._current_solution

