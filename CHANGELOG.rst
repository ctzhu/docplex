Changelog
---------

New in 1.0.630:
---------------
* Added support for CPLEX 12.7 and Python 3.5.
* Upgraded the DOcplexcloud client to version 1.0.202.
* Module ``docplex.mp.advmodel`` is now officially supported. This module
   provides support for efficient, specialized aggregator methods for large
   models.
* When solving on DOcplexcloud, proxies can now be specified with the
   ``context.solver.docloud.proxies`` property.
* When two constraints are defined with the same name, issue a warning instead of
   a fatal exception. The last constraint defined will take over the first one.
* Fix ValueError when passing a pandas DataFrame as variable keys (using
   DataFrame indexes).
* Solution.get_values() returns a collection of variables' values in one call.
* ``docplex.mp.model`` no longer imports ``docloud.status``. Any status
   previously initialized as ``JobSolveStatus.UNKNOWN`` is now initialized as
   ``None``.
* Minor improvements to notebooks and examples.
