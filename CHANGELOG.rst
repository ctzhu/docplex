Changelog
---------

New in 2.1.28:
``````````````
* New methods ``Model.logical_or()`` and ``Model.logical_and()`` handle
  logical operations on binary variables.
* DOcplex now supports CPLEX 12.7.1 and Benders decomposition. Set annotations
  on constraints and variables using the ``benders_annotation`` property and use
  the proper CPLEX parameters governing Benders decomposition.
* CPLEX tutorials: in the documentation and as notebooks in the examples.
* Fixed a bug in ``docplex.mp.solution.SolveSolution.display()`` and in 
  ``docplex.mp.solution.Model.report_kpi()`` when using unicode variable names.
* There's now a simple command line interface for DOcplexcloud. It can be run
  in a terminal. ``python -m docplex.cli help`` for more info. That command
  line reads your DOcplexcloud credentials in your cplex_config.py file. It
  allows you to submit, list, delete jobs on DOcplexcloud. The cli is available
  in notebooks too, using the ``%docplex_cli`` magics. ``%docplex_cli help`` for
  some help. In a notebook, credentials can be passed using `%docplex_url` and
  `%docplex_key` magics.
* Removing constraints in 1 call
* Bug fixes when editing an existing model.
* Bug fix in the relaxation mechanism when using docplexcloud.


New in 2.0.15:
``````````````

* Piecewise linear (PWL) functions are now supported. An API is now available
  on ``docplex.mp.model`` to create PWL functions and to create constraints using these PWL functions.
  PWL functions may be defined with breakpoints (default API) or by using slopes. Some simple arithmetic is
  also available to build new PWL functions by adding, subtracting, or scaling existing PWL functions.
* DOcplex has undergone a significant overhaul effort that has resulted in an average of 30-50% improvement
  of modeling run-time performance. All parts of the API benefit from the performance improvements: creation of variables and constraints, removal of constraints, computation of sums of variables, and so on.
* Constraints are now fully editable: 
  the expressions of a constraint can be modified.
  Similarly, the objective expression can also be modified. This allows for complex workflows in which the model is modified after a solve and then solved again. 
* docplex is now available on Anaconda cloud and can be installed via the conda installation packager.
  See the `IBM Anaconda home <https://anaconda.org/IBMDecisionOptimization>`_
  CPLEX Community Edition for Python is also provided on Anaconda Cloud to get free local solving capabilities with limitations.
* Support of ``~/.docplexrc`` configuration files for ``docplex.mp.context.Context`` is now dropped.
  This feature has been deprecated since 1.0.0.
* Known incompatibility: class ``docplex.mp.model.AbstractModel`` moved to ``docplex.mp.absmodel.AbstractModel``. 
  Samples using this class have been updated.

New in 1.0.630:
```````````````

* Added support for CPLEX 12.7 and Python 3.5.
* Upgraded the DOcplexcloud client to version 1.0.202.
* Module ``docplex.mp.advmodel`` is now officially supported. This module
  provides support for efficient, specialized aggregator methods for large
  models.
* When solving on DOcplexcloud, proxies can now be specified with the
  ``context.solver.docloud.proxies`` property.
* When two constraints are defined with the same name, issue a warning instead of
  a fatal exception. The last constraint defined will take over the first one in the name directory.
* Fix ValueError when passing a pandas DataFrame as variable keys (using
  DataFrame indexes).
* Solution.get_values() returns a collection of variable values in one call.
* ``docplex.mp.model`` no longer imports ``docloud.status``. Any status
  previously initialized as ``JobSolveStatus.UNKNOWN`` is now initialized as
  ``None``.
* Minor improvements to notebooks and examples.
