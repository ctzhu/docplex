# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module implements a simulator of model solver.
It is mainly used for testing.
"""

from docplex.cp.expression import *
from docplex.cp.solution import *
import docplex.cp.solver as solver
from docplex.cp.utils import *

import random

###############################################################################
##  Constants
###############################################################################

# Max value used by solution simulator
MAX_SIMULATED_VALUE = 10000

# Max number of solutions when using next() iterator
MAX_ITERATED_SOLUTIONS = 6


###############################################################################
##  Public classes
###############################################################################

class CpoSolverSimulatorFail(solver.CpoSolverAgent):
    """ CPO solver simulator that always fail (status unfeasible) """
    
    def __init__(self, model, params, context):
        """ Create a solver simulator

        Args:
            model:    Model to solve
            params:   Solving parameters
            context:  Solver context
        """
        super(CpoSolverSimulatorFail, self).__init__(model, params, context)

    def solve(self):
        """ Solve the model.

        Returns:
            Model solution expressed as CpoModelSolution
        """
        # Warn about simulator
        print("WARNING: Solver simulator always returns fail")

        # Build fake infeasible solution
        msol = CpoModelSolution()
        msol._set_solve_status(SOLVE_STATUS_INFEASIBLE)

        # Return
        return msol


class CpoSolverSimulatorRandom(solver.CpoSolverAgent):
    """ CPO solver simulator that generates a random solution """

    def __init__(self, model, params, context):
        """ Create a solver simulator

        Args:
            model:    Model to solve
            params:   Solving parameters
            context:  Solver context
        """
        super(CpoSolverSimulatorRandom, self).__init__(model, params, context)
        self.max_iterator_solutions = random.randint(0, MAX_ITERATED_SOLUTIONS)
        self.solution_count = 0

    def solve(self):
        """ Solve the model.

        Returns:
            Model solution expressed as CpoModelSolution
        """
        # Warn about simulator
        print("WARNING: Solver simulator returns a random solution")

        # Force generation of CPO format if required (for testing purpose only)
        if self.context.create_cpo:
            self._get_cpo_model_string()

        # Build fake feasible solution
        return self._generate_solution()


    def next(self):
        """ Get the next available solution.

        (This method starts search automatically.)

        Returns:
            Next model solution (type CpoModelSolution)
        """
        # Check first call to next
        if self.solution_count == 0:
            print("WARNING: Solver simulator returns a random list of solutions")
            # Force generation of CPO format if required (for testing purpose only)
            if self.context.create_cpo:
                self._get_cpo_model_string()

        # Check all solutions already generated
        if self.solution_count >= self.max_iterator_solutions:
            # Generate last solution
            msol = CpoModelSolution()
            msol._set_solve_status(SOLVE_STATUS_FEASIBLE)
            msol._set_fail_status(FAIL_STATUS_SEARCH_COMPLETED)
            return msol

        else:
           # Generate next solution
           self.solution_count += 1
           return self._generate_solution()


    def _generate_solution(self):
        """ Generate a random solution complying to the model

        Returns:
            Random model solution expressed as CpoModelSolution
        """
        # Build fake feasible solution
        msol = CpoModelSolution()
        msol._set_solve_status(SOLVE_STATUS_FEASIBLE)

        # Generate objective
        x = self.model.get_optimization_expression()
        if x:
            # Determine number of values
            x = x.get_operands()[0]
            if x.get_type().is_array():
                nbval = len(x.get_value())
            else:
                nbval = 1
            ovals = []
            for i in range(nbval):
                ovals.append(random.randint(0, MAX_SIMULATED_VALUE))
            msol._set_objective_values(ovals)

        # Generate a solution for each variable
        for (var, loc) in self.model.get_all_variables():
            if isinstance(var, CpoIntVar):
                vsol = CpoIntVarSolution(var.get_name(), _random_value_in_complex_domain(var.get_domain()))
                msol._add_var_solution(vsol)

            elif isinstance(var, CpoIntervalVar):
                # Generate presence
                if var.is_absent():
                    present = False
                elif var.is_present():
                    present = True
                else:
                    present = (random.random() > 0.3)
                # Generate start and end
                dom = _common_interval_domain(var.get_start(), var.get_end())
                start = end = 0
                while start >= end:
                    start = _random_value_in_interval_domain(dom)
                    end = _random_value_in_interval_domain(dom)
                # Generate size
                size = _random_value_in_interval_domain(var.get_size())
                # Add variable to solution
                vsol = CpoIntervalVarSolution(var.get_name(), present, start, end, size)
                msol._add_var_solution(vsol)

            elif isinstance(var, CpoStateFunction):
                # Build list of steps
                lsteps = []
                cstart = 0
                while cstart < MAX_SIMULATED_VALUE:
                    size = random.randint(1, MAX_SIMULATED_VALUE / 10)
                    lsteps.append((cstart, cstart + size, random.randint(0, 10)))
                    cstart += size
                vsol = CpoStateFunctionSolution(var.get_name(), lsteps)
                msol._add_var_solution(vsol)

        # Generate a solution for composite variable
        for (var, loc) in self.model.get_all_variables():
            if isinstance(var, CpoSequenceVar):
                # Build sequence or results
                lvres = []
                for v in var.get_interval_variables():
                    lvres.append(msol.get_var_solution(v.get_name()))
                random.shuffle(lvres)
                vsol = CpoSequenceVarSolution(var.get_name(), lvres)
                msol._add_var_solution(vsol)

        # Return
        return msol


###############################################################################
##  Private methods
###############################################################################

def _random_value_in_complex_domain(dom):
    """ Determine a random integer value in a domain

    Args:
        dom:  Value domain, list of integers or interval tuples
    Returns:
        Random value in this domain
    """
    # First select a domain element
    dlm = dom[random.randint(0, len(dom) - 1)]
    if is_int(dlm):
        return dlm
    return random.randint(dlm[0], min(dlm[1], MAX_SIMULATED_VALUE))

def _random_value_in_interval_domain(dom):
    """ Determine a random integer value in a domain expressed as a single interval

    Args:
        dom:  Interval domain (couple of values)
    Returns:
        Random value in this domain
    """
    return random.randint(dom[0], min(dom[1], MAX_SIMULATED_VALUE))

def _common_interval_domain(dom1, dom2):
    """ Determine the interval domain that is common to two interval domains

    Args:
        dom1:  First interval domain
        dom2:  Second interval domain
    Returns:
        Common domain
    """
    return (max(dom1[0], dom2[0]), min(dom1[1], dom2[1]))

