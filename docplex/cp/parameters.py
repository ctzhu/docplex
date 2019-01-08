# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Generated automatically

"""
This module handles the parameters that can be assigned to CP Optimizer to configure
the solving of a model.

The class `CpoParameters` contains the list of modifiable parameters
expressed as properties with getters and setters.
For the parameters that require special values, those values are given as constants.

Changing the value of a parameter can be done in multiple ways. For example, the `TimeLimit`
can be set to 60s :
   * `params.TimeLimit = 60`
   * `params.set_TimeLimit(60)`
   * `params["TimeLimit"] = 60`

Retrieving the value of a parameter can be done in the same way:
   * `v = params.TimeLimit`
   * `v = params.get_TimeLimit()`
   * `v = params["TimeLimit"]`

If a parameter is not set, the value returned by the first two access forms is None. The final access form
(element of a dictionary) raises an exception.

Getting the list of all parameters that have been changed can be done by calling the method
`keys()`.

Note that PEP8 naming convention is not applied here, to keep parameter names as they
are in the solver, so that they can be referenced in solver logs.
"""

from docplex.cp.utils import Context

#----------------------------------------------------------------------------
#  Symbolic parameters values
#----------------------------------------------------------------------------

VALUE_INT_MIN                     = 'IntMin'
VALUE_AUTO                        = 'Auto'
VALUE_OFF                         = 'Off'
VALUE_ON                          = 'On'
VALUE_DEFAULT                     = 'Default'
VALUE_LOW                         = 'Low'
VALUE_BASIC                       = 'Basic'
VALUE_MEDIUM                      = 'Medium'
VALUE_EXTENDED                    = 'Extended'
VALUE_STANDARD                    = 'Standard'
VALUE_INT_SCIENTIFIC              = 'IntScientific'
VALUE_INT_FIXED                   = 'IntFixed'
VALUE_BAS_SCIENTIFIC              = 'BasScientific'
VALUE_BAS_FIXED                   = 'BasFixed'
VALUE_SEARCH_HAS_NOT_FAILED       = 'SearchHasNotFailed'
VALUE_SEARCH_HAS_FAILED_NORMALLY  = 'SearchHasFailedNormally'
VALUE_SEARCH_STOPPED_BY_LIMIT     = 'SearchStoppedByLimit'
VALUE_SEARCH_STOPPED_BY_LABEL     = 'SearchStoppedByLabel'
VALUE_SEARCH_STOPPED_BY_EXIT      = 'SearchStoppedByExit'
VALUE_SEARCH_STOPPED_BY_ABORT     = 'SearchStoppedByAbort'
VALUE_SEARCH_STOPPED_BY_EXCEPTION = 'SearchStoppedByException'
VALUE_UNKNOWN_FAILURE_STATUS      = 'UnknownFailureStatus'
VALUE_QUIET                       = 'Quiet'
VALUE_TERSE                       = 'Terse'
VALUE_NORMAL                      = 'Normal'
VALUE_VERBOSE                     = 'Verbose'
VALUE_DEPTH_FIRST                 = 'DepthFirst'
VALUE_RESTART                     = 'Restart'
VALUE_MULTI_POINT                 = 'MultiPoint'
VALUE_DIVERSE                     = 'Diverse'
VALUE_FOCUSED                     = 'Focused'
VALUE_INTENSIVE                   = 'Intensive'
VALUE_SECONDS                     = 'Seconds'
VALUE_HOURS_MINUTES_SECONDS       = 'HoursMinutesSeconds'
VALUE_NO_TIME                     = 'NoTime'
VALUE_CPU_TIME                    = 'CPUTime'
VALUE_ELAPSED_TIME                = 'ElapsedTime'
VALUE_INFEASIBLE                  = 'Infeasible'
VALUE_HARD                        = 'Hard'
VALUE_COMPLEMENTARY_FEASIBLE      = 'ComplementaryFeasible'
VALUE_INT_MAX                     = 'IntMax'
VALUE_NUM_MAX                     = 'NumMax'
VALUE_INFINITY                    = 'Infinity'


#----------------------------------------------------------------------------
#  Parameters handler
#----------------------------------------------------------------------------


class CpoParameters(Context):
    """ Class for handling solving parameters
    """
    def __init__(self, **kwargs):
        """ Creates a new empty parameters repository
        """
        super(CpoParameters, self).__init__(**kwargs)


    # Properties definitions



    def set_AllDiffInferenceLevel(self, val):
        """
        This parameter specifies the inference level for every constraint AllDiff extracted to the invoking
        CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and Extended. The
        default value is Default, which allows the inference strength of all AllDiff constraints to be
        controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        self.set_attribute('AllDiffInferenceLevel', val)

    def get_AllDiffInferenceLevel(self):
        """ Value of parameter AllDiffInferenceLevel, None if not defined.

        This parameter specifies the inference level for every constraint AllDiff extracted to the invoking
        CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and Extended. The
        default value is Default, which allows the inference strength of all AllDiff constraints to be
        controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        return self.get_attribute('AllDiffInferenceLevel')

    AllDiffInferenceLevel = property(get_AllDiffInferenceLevel, set_AllDiffInferenceLevel)


    def set_AllMinDistanceInferenceLevel(self, val):
        """
        This parameter specifies the inference level for every constraint AllMinDistance extracted to the
        invoked CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and
        Extended. The default value is Default, which allows the inference strength of all AllMinDistance
        constraints to be controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        self.set_attribute('AllMinDistanceInferenceLevel', val)

    def get_AllMinDistanceInferenceLevel(self):
        """ Value of parameter AllMinDistanceInferenceLevel, None if not defined.

        This parameter specifies the inference level for every constraint AllMinDistance extracted to the
        invoked CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and
        Extended. The default value is Default, which allows the inference strength of all AllMinDistance
        constraints to be controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        return self.get_attribute('AllMinDistanceInferenceLevel')

    AllMinDistanceInferenceLevel = property(get_AllMinDistanceInferenceLevel, set_AllMinDistanceInferenceLevel)


    def set_AutomaticReplay(self, val):
        """
        This parameter is an advanced, low-level one for controlling the behavior of solve() and next().
        When the model being solved has an objective and solve is used, or when startNewSearch and next are
        used to produce multiple solutions, the solver may have a need to replay the last (or best) solution
        found. This can, in some cases, involve re-invoking the stategy which produced the solution.
        Normally this is only necessary if you use low level "Ilc" interfaces to specify problem elements
        not in the model (instance of Model). This parameter can take the values On or Off. The default
        value is On. A typical reason for setting this parameter to Off is, for instance, if you use your
        own custom goal (instance of IlcGoal), and this goal is not deterministic (does not do the same
        thing when executed twice). In this instance, the replay will not work correctly, and you can use
        this parameter to disable replay.
        The value must be a symbol in [Off, On].
        Default value is On.
        """
        self.set_attribute('AutomaticReplay', val)

    def get_AutomaticReplay(self):
        """ Value of parameter AutomaticReplay, None if not defined.

        This parameter is an advanced, low-level one for controlling the behavior of solve() and next().
        When the model being solved has an objective and solve is used, or when startNewSearch and next are
        used to produce multiple solutions, the solver may have a need to replay the last (or best) solution
        found. This can, in some cases, involve re-invoking the stategy which produced the solution.
        Normally this is only necessary if you use low level "Ilc" interfaces to specify problem elements
        not in the model (instance of Model). This parameter can take the values On or Off. The default
        value is On. A typical reason for setting this parameter to Off is, for instance, if you use your
        own custom goal (instance of IlcGoal), and this goal is not deterministic (does not do the same
        thing when executed twice). In this instance, the replay will not work correctly, and you can use
        this parameter to disable replay.
        The value must be a symbol in [Off, On].
        Default value is On.
        """
        return self.get_attribute('AutomaticReplay')

    AutomaticReplay = property(get_AutomaticReplay, set_AutomaticReplay)


    def set_BranchLimit(self, val):
        """
        This parameter limits the number of branches that are made before terminating a search. A branch is
        a decision made at a choice point in the search, a typical node being made up of two branches, for
        example: x == value and x != value. A branch is only counted at the moment a decision is executed,
        not when the two branches of the choice point are decided. A branch is counted even if the decision
        leads to an inconsistency (failure). The possible values of this parameter range from 0 to Infinity.
        A value of Infinity does not set any limit. The value Infinity is the default value.
        The value must be an integer in [0..Infinity].
        Default value is Infinity.
        """
        self.set_attribute('BranchLimit', val)

    def get_BranchLimit(self):
        """ Value of parameter BranchLimit, None if not defined.

        This parameter limits the number of branches that are made before terminating a search. A branch is
        a decision made at a choice point in the search, a typical node being made up of two branches, for
        example: x == value and x != value. A branch is only counted at the moment a decision is executed,
        not when the two branches of the choice point are decided. A branch is counted even if the decision
        leads to an inconsistency (failure). The possible values of this parameter range from 0 to Infinity.
        A value of Infinity does not set any limit. The value Infinity is the default value.
        The value must be an integer in [0..Infinity].
        Default value is Infinity.
        """
        return self.get_attribute('BranchLimit')

    BranchLimit = property(get_BranchLimit, set_BranchLimit)


    def set_ChoicePointLimit(self, val):
        """
        This parameter limits the number of choice points that are created before terminating a search. The
        possible values of this parameter range from 0 to Infinity. A value of Infinity does not set any
        limit. The value Infinity is the default value.
        The value must be an integer in [0..Infinity].
        Default value is Infinity.
        """
        self.set_attribute('ChoicePointLimit', val)

    def get_ChoicePointLimit(self):
        """ Value of parameter ChoicePointLimit, None if not defined.

        This parameter limits the number of choice points that are created before terminating a search. The
        possible values of this parameter range from 0 to Infinity. A value of Infinity does not set any
        limit. The value Infinity is the default value.
        The value must be an integer in [0..Infinity].
        Default value is Infinity.
        """
        return self.get_attribute('ChoicePointLimit')

    ChoicePointLimit = property(get_ChoicePointLimit, set_ChoicePointLimit)


    def set_ConflictRefinerBranchLimit(self, val):
        """
        This parameter limits the total number of branches that are made before terminating the conflict
        refiner. The possible values of this parameter range from 0 to Infinity. A value of Infinity does
        not set any limit. The value Infinity is the default value.
        The value must be an integer in [0..Infinity].
        Default value is Infinity.
        """
        self.set_attribute('ConflictRefinerBranchLimit', val)

    def get_ConflictRefinerBranchLimit(self):
        """ Value of parameter ConflictRefinerBranchLimit, None if not defined.

        This parameter limits the total number of branches that are made before terminating the conflict
        refiner. The possible values of this parameter range from 0 to Infinity. A value of Infinity does
        not set any limit. The value Infinity is the default value.
        The value must be an integer in [0..Infinity].
        Default value is Infinity.
        """
        return self.get_attribute('ConflictRefinerBranchLimit')

    ConflictRefinerBranchLimit = property(get_ConflictRefinerBranchLimit, set_ConflictRefinerBranchLimit)


    def set_ConflictRefinerFailLimit(self, val):
        """
        This parameter limits the total number of failures that can occur before terminating the conflict
        refiner. The possible values of this parameter range from 0 to Infinity. A value of Infinity does
        not set any limit. The value Infinity is the default value.
        The value must be an integer in [0..Infinity].
        Default value is Infinity.
        """
        self.set_attribute('ConflictRefinerFailLimit', val)

    def get_ConflictRefinerFailLimit(self):
        """ Value of parameter ConflictRefinerFailLimit, None if not defined.

        This parameter limits the total number of failures that can occur before terminating the conflict
        refiner. The possible values of this parameter range from 0 to Infinity. A value of Infinity does
        not set any limit. The value Infinity is the default value.
        The value must be an integer in [0..Infinity].
        Default value is Infinity.
        """
        return self.get_attribute('ConflictRefinerFailLimit')

    ConflictRefinerFailLimit = property(get_ConflictRefinerFailLimit, set_ConflictRefinerFailLimit)


    def set_ConflictRefinerIterationLimit(self, val):
        """
        This parameter limits the number of iterations that are made before terminating the conflict
        refiner. The possible values of this parameter range from 0 to Infinity. A value of Infinity does
        not set any limit. The value Infinity is the default value.
        The value must be an integer in [0..Infinity].
        Default value is Infinity.
        """
        self.set_attribute('ConflictRefinerIterationLimit', val)

    def get_ConflictRefinerIterationLimit(self):
        """ Value of parameter ConflictRefinerIterationLimit, None if not defined.

        This parameter limits the number of iterations that are made before terminating the conflict
        refiner. The possible values of this parameter range from 0 to Infinity. A value of Infinity does
        not set any limit. The value Infinity is the default value.
        The value must be an integer in [0..Infinity].
        Default value is Infinity.
        """
        return self.get_attribute('ConflictRefinerIterationLimit')

    ConflictRefinerIterationLimit = property(get_ConflictRefinerIterationLimit, set_ConflictRefinerIterationLimit)


    def set_ConflictRefinerOnVariables(self, val):
        """
        This parameter specifies whether the conflict refiner should refine variables domains. Possible
        values for this parameter are On (conflict refiner will refine both constraints and variables
        domains) and Off (conflict refiner will only refine constraints). The default value is Off.
        The value must be a symbol in [Off, On].
        Default value is Off.
        """
        self.set_attribute('ConflictRefinerOnVariables', val)

    def get_ConflictRefinerOnVariables(self):
        """ Value of parameter ConflictRefinerOnVariables, None if not defined.

        This parameter specifies whether the conflict refiner should refine variables domains. Possible
        values for this parameter are On (conflict refiner will refine both constraints and variables
        domains) and Off (conflict refiner will only refine constraints). The default value is Off.
        The value must be a symbol in [Off, On].
        Default value is Off.
        """
        return self.get_attribute('ConflictRefinerOnVariables')

    ConflictRefinerOnVariables = property(get_ConflictRefinerOnVariables, set_ConflictRefinerOnVariables)


    def set_ConflictRefinerTimeLimit(self, val):
        """
        This parameter limits the CPU time spent before terminating the conflict refiner. The possible
        values of this parameter range from 0 to Infinity. A value of Infinity does not set any limit. The
        value Infinity is the default value.
        The value must be a float in [0..Infinity].
        Default value is Infinity.
        """
        self.set_attribute('ConflictRefinerTimeLimit', val)

    def get_ConflictRefinerTimeLimit(self):
        """ Value of parameter ConflictRefinerTimeLimit, None if not defined.

        This parameter limits the CPU time spent before terminating the conflict refiner. The possible
        values of this parameter range from 0 to Infinity. A value of Infinity does not set any limit. The
        value Infinity is the default value.
        The value must be a float in [0..Infinity].
        Default value is Infinity.
        """
        return self.get_attribute('ConflictRefinerTimeLimit')

    ConflictRefinerTimeLimit = property(get_ConflictRefinerTimeLimit, set_ConflictRefinerTimeLimit)


    def set_CountDifferentInferenceLevel(self, val):
        """
        This parameter specifies the inference level for every constraint CountDifferent extracted to the
        invoking CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and
        Extended. The default value is Default, which allows the inference strength of all CountDifferent
        constraints to be controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        self.set_attribute('CountDifferentInferenceLevel', val)

    def get_CountDifferentInferenceLevel(self):
        """ Value of parameter CountDifferentInferenceLevel, None if not defined.

        This parameter specifies the inference level for every constraint CountDifferent extracted to the
        invoking CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and
        Extended. The default value is Default, which allows the inference strength of all CountDifferent
        constraints to be controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        return self.get_attribute('CountDifferentInferenceLevel')

    CountDifferentInferenceLevel = property(get_CountDifferentInferenceLevel, set_CountDifferentInferenceLevel)


    def set_CountInferenceLevel(self, val):
        """
        This parameter specifies the inference level for every constraint Count extracted to the invoked CP
        instance. Possible values for this parameter are Default, Low, Basic, Medium, and Extended. The
        default value is Default, which allows the inference strength of all Count constraints to be
        controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        self.set_attribute('CountInferenceLevel', val)

    def get_CountInferenceLevel(self):
        """ Value of parameter CountInferenceLevel, None if not defined.

        This parameter specifies the inference level for every constraint Count extracted to the invoked CP
        instance. Possible values for this parameter are Default, Low, Basic, Medium, and Extended. The
        default value is Default, which allows the inference strength of all Count constraints to be
        controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        return self.get_attribute('CountInferenceLevel')

    CountInferenceLevel = property(get_CountInferenceLevel, set_CountInferenceLevel)


    def set_CumulFunctionInferenceLevel(self, val):
        """
        This parameter specifies the inference level for constraints on expressions CumulFunctionExpr
        extracted to the invoked CP instance. Possible values for this parameter are Default, Low, Basic,
        Medium, and Extended. The default value is Default, which allows the inference strength of all
        constraints on CumulFunctionExpr to be controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        self.set_attribute('CumulFunctionInferenceLevel', val)

    def get_CumulFunctionInferenceLevel(self):
        """ Value of parameter CumulFunctionInferenceLevel, None if not defined.

        This parameter specifies the inference level for constraints on expressions CumulFunctionExpr
        extracted to the invoked CP instance. Possible values for this parameter are Default, Low, Basic,
        Medium, and Extended. The default value is Default, which allows the inference strength of all
        constraints on CumulFunctionExpr to be controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        return self.get_attribute('CumulFunctionInferenceLevel')

    CumulFunctionInferenceLevel = property(get_CumulFunctionInferenceLevel, set_CumulFunctionInferenceLevel)


    def set_DefaultInferenceLevel(self, val):
        """
        This parameter specifies the general inference level for constraints whose particular inference
        level is Default. Possible values for this parameter (in increasing order of inference strength) are
        Low, Basic, Medium, and Extended. The default value is Basic.
        The value must be a symbol in [Low, Basic, Medium, Extended].
        Default value is Basic.
        """
        self.set_attribute('DefaultInferenceLevel', val)

    def get_DefaultInferenceLevel(self):
        """ Value of parameter DefaultInferenceLevel, None if not defined.

        This parameter specifies the general inference level for constraints whose particular inference
        level is Default. Possible values for this parameter (in increasing order of inference strength) are
        Low, Basic, Medium, and Extended. The default value is Basic.
        The value must be a symbol in [Low, Basic, Medium, Extended].
        Default value is Basic.
        """
        return self.get_attribute('DefaultInferenceLevel')

    DefaultInferenceLevel = property(get_DefaultInferenceLevel, set_DefaultInferenceLevel)


    def set_DistributeInferenceLevel(self, val):
        """
        This parameter specifies the inference level for every constraint Distribute extracted to the
        invoked CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and
        Extended. The default value is Default, which allows the inference strength of all Distribute
        constraints to be controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        self.set_attribute('DistributeInferenceLevel', val)

    def get_DistributeInferenceLevel(self):
        """ Value of parameter DistributeInferenceLevel, None if not defined.

        This parameter specifies the inference level for every constraint Distribute extracted to the
        invoked CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and
        Extended. The default value is Default, which allows the inference strength of all Distribute
        constraints to be controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        return self.get_attribute('DistributeInferenceLevel')

    DistributeInferenceLevel = property(get_DistributeInferenceLevel, set_DistributeInferenceLevel)


    def set_DynamicProbing(self, val):
        """
        This parameter controls probing carried out during search. Probing can be useful on some problems as
        it can make stronger inferences on combinations of constraints. Possible values for this parameter
        are On (dynamic probing is activated with a constant strength), Auto (dynamic probing is activated
        and its strength is adjusted adaptively) and Off (dynamic probing is deactivated). The strength of
        probing can be defined by parameter DynamicProbingStrength. Dynamic probing only has an effect when
        using the "Restart" (Restart) search type, on problems without interval variables. The default value
        of this parameter is Auto.
        The value must be a symbol in [Off, On, Auto].
        Default value is Auto.
        """
        self.set_attribute('DynamicProbing', val)

    def get_DynamicProbing(self):
        """ Value of parameter DynamicProbing, None if not defined.

        This parameter controls probing carried out during search. Probing can be useful on some problems as
        it can make stronger inferences on combinations of constraints. Possible values for this parameter
        are On (dynamic probing is activated with a constant strength), Auto (dynamic probing is activated
        and its strength is adjusted adaptively) and Off (dynamic probing is deactivated). The strength of
        probing can be defined by parameter DynamicProbingStrength. Dynamic probing only has an effect when
        using the "Restart" (Restart) search type, on problems without interval variables. The default value
        of this parameter is Auto.
        The value must be a symbol in [Off, On, Auto].
        Default value is Auto.
        """
        return self.get_attribute('DynamicProbing')

    DynamicProbing = property(get_DynamicProbing, set_DynamicProbing)


    def set_DynamicProbingStrength(self, val):
        """
        This parameter controls the effort which is dedicated to dynamic probing. It is expressed as a
        factor of the total search effort: changing this parameter has no effect unless the DynamicProbing
        parameter is set to Auto or On. When DynamicProbing has value On, the probing strength is held
        constant througout the search process. When DynamicProbing has value Auto, the probing strength
        starts off at the specified value and is thereafter adjusted automatically. Possible values for this
        parameter range from 0.001 to 1000. A value of 1.0 indicates that dynamic probing will consume a
        roughly equal amount of effort as the rest of the search. The default value of this parameter is
        0.03, meaning that around 3% of total search time is dedicated to dynamic probing.
        The value must be a float in [0.001..1000].
        Default value is 0.03.
        """
        self.set_attribute('DynamicProbingStrength', val)

    def get_DynamicProbingStrength(self):
        """ Value of parameter DynamicProbingStrength, None if not defined.

        This parameter controls the effort which is dedicated to dynamic probing. It is expressed as a
        factor of the total search effort: changing this parameter has no effect unless the DynamicProbing
        parameter is set to Auto or On. When DynamicProbing has value On, the probing strength is held
        constant througout the search process. When DynamicProbing has value Auto, the probing strength
        starts off at the specified value and is thereafter adjusted automatically. Possible values for this
        parameter range from 0.001 to 1000. A value of 1.0 indicates that dynamic probing will consume a
        roughly equal amount of effort as the rest of the search. The default value of this parameter is
        0.03, meaning that around 3% of total search time is dedicated to dynamic probing.
        The value must be a float in [0.001..1000].
        Default value is 0.03.
        """
        return self.get_attribute('DynamicProbingStrength')

    DynamicProbingStrength = property(get_DynamicProbingStrength, set_DynamicProbingStrength)


    def set_ElementInferenceLevel(self, val):
        """
        This parameter specifies the inference level for every element constraint extracted to the invoked
        CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and Extended. The
        default value is Default, which allows the inference strength of all element constraints to be
        controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        self.set_attribute('ElementInferenceLevel', val)

    def get_ElementInferenceLevel(self):
        """ Value of parameter ElementInferenceLevel, None if not defined.

        This parameter specifies the inference level for every element constraint extracted to the invoked
        CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and Extended. The
        default value is Default, which allows the inference strength of all element constraints to be
        controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        return self.get_attribute('ElementInferenceLevel')

    ElementInferenceLevel = property(get_ElementInferenceLevel, set_ElementInferenceLevel)


    def set_FailLimit(self, val):
        """
        This parameter limits the number of failures that can occur before terminating the search. The
        possible values of this parameter range from 0 to Infinity. A value of Infinity does not set any
        limit. The value Infinity is the default value.
        The value must be an integer in [0..Infinity].
        Default value is Infinity.
        """
        self.set_attribute('FailLimit', val)

    def get_FailLimit(self):
        """ Value of parameter FailLimit, None if not defined.

        This parameter limits the number of failures that can occur before terminating the search. The
        possible values of this parameter range from 0 to Infinity. A value of Infinity does not set any
        limit. The value Infinity is the default value.
        The value must be an integer in [0..Infinity].
        Default value is Infinity.
        """
        return self.get_attribute('FailLimit')

    FailLimit = property(get_FailLimit, set_FailLimit)


    def set_FailureDirectedSearch(self, val):
        """
        This parameter controls usage of failure-directed search. Failure-directed search assumes that there
        is no (better) solution or that such a solution is very hard to find. Therefore it focuses on a
        systematic exploration of search space, first eliminating assignments that are most likely to fail.
        Failure-directed search is used only for scheduling problems (i.e. models containing interval
        variables) and only when the parameter SearchType is set to Restart or Auto. Legal values for the
        FailureDirectedSearch parameter are On (the default) and Off. When the value is On then CP Optimizer
        starts failure-directed search when other search strategies are (no longer) successful and when the
        memory necessary for the search does not exceed the value set by the FailureDirectedSearchMaxMemory
        parameter. Setting the parameter to Off disables the feature.
        The value must be a symbol in [Off, On].
        Default value is On.
        """
        self.set_attribute('FailureDirectedSearch', val)

    def get_FailureDirectedSearch(self):
        """ Value of parameter FailureDirectedSearch, None if not defined.

        This parameter controls usage of failure-directed search. Failure-directed search assumes that there
        is no (better) solution or that such a solution is very hard to find. Therefore it focuses on a
        systematic exploration of search space, first eliminating assignments that are most likely to fail.
        Failure-directed search is used only for scheduling problems (i.e. models containing interval
        variables) and only when the parameter SearchType is set to Restart or Auto. Legal values for the
        FailureDirectedSearch parameter are On (the default) and Off. When the value is On then CP Optimizer
        starts failure-directed search when other search strategies are (no longer) successful and when the
        memory necessary for the search does not exceed the value set by the FailureDirectedSearchMaxMemory
        parameter. Setting the parameter to Off disables the feature.
        The value must be a symbol in [Off, On].
        Default value is On.
        """
        return self.get_attribute('FailureDirectedSearch')

    FailureDirectedSearch = property(get_FailureDirectedSearch, set_FailureDirectedSearch)


    def set_FailureDirectedSearchEmphasis(self, val):
        """
        This parameter controls how much time CP Optimizer invests into failure-directed search once it is
        started. The default value Auto means that CP Optimizer observes the actual performance of
        failure-directed search and decides automaticaly how much time is invested. Any other value means
        that once failure-directed search has started, it is used by given number of workers. The value does
        not have to be integer. For example, value 1.5 means that first worker spends 100% of the time by
        failure-directed search, second worker 50% and remaining workers 0%. See also Workers For more
        information about failure-directed search see parameter FailureDirectedSearch.
        The value must be a float in [0..Infinity].
        Default value is Auto.
        """
        self.set_attribute('FailureDirectedSearchEmphasis', val)

    def get_FailureDirectedSearchEmphasis(self):
        """ Value of parameter FailureDirectedSearchEmphasis, None if not defined.

        This parameter controls how much time CP Optimizer invests into failure-directed search once it is
        started. The default value Auto means that CP Optimizer observes the actual performance of
        failure-directed search and decides automaticaly how much time is invested. Any other value means
        that once failure-directed search has started, it is used by given number of workers. The value does
        not have to be integer. For example, value 1.5 means that first worker spends 100% of the time by
        failure-directed search, second worker 50% and remaining workers 0%. See also Workers For more
        information about failure-directed search see parameter FailureDirectedSearch.
        The value must be a float in [0..Infinity].
        Default value is Auto.
        """
        return self.get_attribute('FailureDirectedSearchEmphasis')

    FailureDirectedSearchEmphasis = property(get_FailureDirectedSearchEmphasis, set_FailureDirectedSearchEmphasis)


    def set_FailureDirectedSearchMaxMemory(self, val):
        """
        This parameter controls the maximum amount of memory (in bytes) available to failure-directed search
        (see FailureDirectedSearchMaxMemory). The default value is 104,857,600 (100MB). Failure-directed
        search can sometimes consume a lot of memory, especially when end times of interval variables are
        not bounded. Therefore it is usually not started immediately, but only when the effective horizon
        (time period over which CP Optimizer must reason) becomes small enough for failure-directed search
        to operate inside the memory limit specified by this parameter. For many types of scheduling
        problems, the effective horizon tends to reduce when CP Optimizer finds a better solution (often
        most significantly when the initial solution is found). Therefore, when each new solution is found,
        CP Optimizer decides whether or not to turn on failure-directed search. Note that this parameter
        does not influence the effectiveness of failure-directed search, once started. Its purpose is only
        to control the point at which failure-directed search will begin to function.
        The value must be an integer in [0..Infinity].
        Default value is 104857600.
        """
        self.set_attribute('FailureDirectedSearchMaxMemory', val)

    def get_FailureDirectedSearchMaxMemory(self):
        """ Value of parameter FailureDirectedSearchMaxMemory, None if not defined.

        This parameter controls the maximum amount of memory (in bytes) available to failure-directed search
        (see FailureDirectedSearchMaxMemory). The default value is 104,857,600 (100MB). Failure-directed
        search can sometimes consume a lot of memory, especially when end times of interval variables are
        not bounded. Therefore it is usually not started immediately, but only when the effective horizon
        (time period over which CP Optimizer must reason) becomes small enough for failure-directed search
        to operate inside the memory limit specified by this parameter. For many types of scheduling
        problems, the effective horizon tends to reduce when CP Optimizer finds a better solution (often
        most significantly when the initial solution is found). Therefore, when each new solution is found,
        CP Optimizer decides whether or not to turn on failure-directed search. Note that this parameter
        does not influence the effectiveness of failure-directed search, once started. Its purpose is only
        to control the point at which failure-directed search will begin to function.
        The value must be an integer in [0..Infinity].
        Default value is 104857600.
        """
        return self.get_attribute('FailureDirectedSearchMaxMemory')

    FailureDirectedSearchMaxMemory = property(get_FailureDirectedSearchMaxMemory, set_FailureDirectedSearchMaxMemory)


    def set_IntervalSequenceInferenceLevel(self, val):
        """
        This parameter specifies the inference level for the maintenance of the domain of every interval
        sequence variable IntervalSequenceVar extracted to the invoking CP instance. Possible values for
        this parameter are Default, Low, Basic, Medium, and Extended. The default value is Default, which
        allows the inference strength of all IntervalSequenceVar to be controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        self.set_attribute('IntervalSequenceInferenceLevel', val)

    def get_IntervalSequenceInferenceLevel(self):
        """ Value of parameter IntervalSequenceInferenceLevel, None if not defined.

        This parameter specifies the inference level for the maintenance of the domain of every interval
        sequence variable IntervalSequenceVar extracted to the invoking CP instance. Possible values for
        this parameter are Default, Low, Basic, Medium, and Extended. The default value is Default, which
        allows the inference strength of all IntervalSequenceVar to be controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        return self.get_attribute('IntervalSequenceInferenceLevel')

    IntervalSequenceInferenceLevel = property(get_IntervalSequenceInferenceLevel, set_IntervalSequenceInferenceLevel)


    def set_LogPeriod(self, val):
        """
        The CP Optimizer search log includes information that is displayed periodically. This parameter
        controls how often that information is displayed. By setting this parameter to a value of k, the log
        is displayed every k branches (search decisions).
        The value must be an integer in [1..Infinity].
        Default value is 1000.
        """
        self.set_attribute('LogPeriod', val)

    def get_LogPeriod(self):
        """ Value of parameter LogPeriod, None if not defined.

        The CP Optimizer search log includes information that is displayed periodically. This parameter
        controls how often that information is displayed. By setting this parameter to a value of k, the log
        is displayed every k branches (search decisions).
        The value must be an integer in [1..Infinity].
        Default value is 1000.
        """
        return self.get_attribute('LogPeriod')

    LogPeriod = property(get_LogPeriod, set_LogPeriod)


    def set_LogSearchTags(self, val):
        """
        This parameter controls the log activation. When set to On, the engine will display failure tags
        (indices) in the engine log when solving the model. To specify the failures to explain, the member
        functions explainFailure(Int failureTag) or explainFailure(IntArray tagArray)	 needs to be called
        with the failure tags as the parameter. Several failures tags can be added. The member function
        clearExplanations() is used to clear the set of failure tags to be explained. To be able to see
        failure tags and explanations, the parameter SearchType must be set to DepthFirst and the parameter
        Workers to 1. The default value of this parameter is Off.
        The value must be a symbol in [Off, On].
        Default value is Off.
        """
        self.set_attribute('LogSearchTags', val)

    def get_LogSearchTags(self):
        """ Value of parameter LogSearchTags, None if not defined.

        This parameter controls the log activation. When set to On, the engine will display failure tags
        (indices) in the engine log when solving the model. To specify the failures to explain, the member
        functions explainFailure(Int failureTag) or explainFailure(IntArray tagArray)	 needs to be called
        with the failure tags as the parameter. Several failures tags can be added. The member function
        clearExplanations() is used to clear the set of failure tags to be explained. To be able to see
        failure tags and explanations, the parameter SearchType must be set to DepthFirst and the parameter
        Workers to 1. The default value of this parameter is Off.
        The value must be a symbol in [Off, On].
        Default value is Off.
        """
        return self.get_attribute('LogSearchTags')

    LogSearchTags = property(get_LogSearchTags, set_LogSearchTags)


    def set_LogVerbosity(self, val):
        """
        This parameter determines the verbosity of the search log. The possible values are Quiet, Terse,
        Normal, and Verbose. Mode Quiet does not display any information, the other modes display
        progressively more information. The default value is Normal. The CP Optimizer search log is meant
        for visual inspection only, not for mechanized parsing. In particular, the log may change from
        version to version of CP Optimizer in order to improve the quality of information displayed in the
        log. Any code based on the log output for correct functioning may have to be updated when a new
        version of CP Optimizer is released.
        The value must be a symbol in [Quiet, Terse, Normal, Verbose].
        Default value is Normal.
        """
        self.set_attribute('LogVerbosity', val)

    def get_LogVerbosity(self):
        """ Value of parameter LogVerbosity, None if not defined.

        This parameter determines the verbosity of the search log. The possible values are Quiet, Terse,
        Normal, and Verbose. Mode Quiet does not display any information, the other modes display
        progressively more information. The default value is Normal. The CP Optimizer search log is meant
        for visual inspection only, not for mechanized parsing. In particular, the log may change from
        version to version of CP Optimizer in order to improve the quality of information displayed in the
        log. Any code based on the log output for correct functioning may have to be updated when a new
        version of CP Optimizer is released.
        The value must be a symbol in [Quiet, Terse, Normal, Verbose].
        Default value is Normal.
        """
        return self.get_attribute('LogVerbosity')

    LogVerbosity = property(get_LogVerbosity, set_LogVerbosity)


    def set_ModelAnonymizer(self, val):
        """
        This parameter controls anonymization of a model dumped via dumpModel. The legal values of this
        parameter are Off and On. The default is Off. When the anonymizer is off, then names of variables
        and constraints in the model may be found in the output file. When the anonymizer is on, names given
        to variables or constraints in the model will not be reflected in the output file and standard
        anonymized names will be used.
        The value must be a symbol in [Off, On].
        Default value is Off.
        """
        self.set_attribute('ModelAnonymizer', val)

    def get_ModelAnonymizer(self):
        """ Value of parameter ModelAnonymizer, None if not defined.

        This parameter controls anonymization of a model dumped via dumpModel. The legal values of this
        parameter are Off and On. The default is Off. When the anonymizer is off, then names of variables
        and constraints in the model may be found in the output file. When the anonymizer is on, names given
        to variables or constraints in the model will not be reflected in the output file and standard
        anonymized names will be used.
        The value must be a symbol in [Off, On].
        Default value is Off.
        """
        return self.get_attribute('ModelAnonymizer')

    ModelAnonymizer = property(get_ModelAnonymizer, set_ModelAnonymizer)


    def set_MultiPointNumberOfSearchPoints(self, val):
        """
        This parameter controls the number of (possibly partial) solutions manipulated by the multi-point
        search algorithm. The default value is 30. A larger value will diversify the search, with possible
        improvement in solution quality at the expense of a longer run time. A smaller value will intensify
        the search, resulting in faster convergence at the expense of solution quality. Note that memory
        consumption increases proportionally to this parameter, for each search point must store each
        decision variable domain.
        The value must be an integer in [2..Infinity].
        Default value is 30.
        """
        self.set_attribute('MultiPointNumberOfSearchPoints', val)

    def get_MultiPointNumberOfSearchPoints(self):
        """ Value of parameter MultiPointNumberOfSearchPoints, None if not defined.

        This parameter controls the number of (possibly partial) solutions manipulated by the multi-point
        search algorithm. The default value is 30. A larger value will diversify the search, with possible
        improvement in solution quality at the expense of a longer run time. A smaller value will intensify
        the search, resulting in faster convergence at the expense of solution quality. Note that memory
        consumption increases proportionally to this parameter, for each search point must store each
        decision variable domain.
        The value must be an integer in [2..Infinity].
        Default value is 30.
        """
        return self.get_attribute('MultiPointNumberOfSearchPoints')

    MultiPointNumberOfSearchPoints = property(get_MultiPointNumberOfSearchPoints, set_MultiPointNumberOfSearchPoints)


    def set_NoOverlapInferenceLevel(self, val):
        """
        This parameter specifies the inference level for every constraint NoOverlap extracted to the invoked
        CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and Extended. The
        default value is Default, which allows the inference strength of all NoOverlap constraints to be
        controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        self.set_attribute('NoOverlapInferenceLevel', val)

    def get_NoOverlapInferenceLevel(self):
        """ Value of parameter NoOverlapInferenceLevel, None if not defined.

        This parameter specifies the inference level for every constraint NoOverlap extracted to the invoked
        CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and Extended. The
        default value is Default, which allows the inference strength of all NoOverlap constraints to be
        controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        return self.get_attribute('NoOverlapInferenceLevel')

    NoOverlapInferenceLevel = property(get_NoOverlapInferenceLevel, set_NoOverlapInferenceLevel)


    def set_OptimalityTolerance(self, val):
        """
        This parameter sets an absolute tolerance on the objective value for optimization models. This means
        that when CP Optimizer reports an optimal solution found, then there is no solution which improves
        the objective by more than the value of this parameter. The default value of this parameter is 0.
        This parameter is used in conjunction with RelativeOptimalityTolerance. The optimality of a solution
        is proven if either of the two parameters' criteria is fulfilled.
        The value must be a float in [0..Infinity].
        Default value is 1e-09.
        """
        self.set_attribute('OptimalityTolerance', val)

    def get_OptimalityTolerance(self):
        """ Value of parameter OptimalityTolerance, None if not defined.

        This parameter sets an absolute tolerance on the objective value for optimization models. This means
        that when CP Optimizer reports an optimal solution found, then there is no solution which improves
        the objective by more than the value of this parameter. The default value of this parameter is 0.
        This parameter is used in conjunction with RelativeOptimalityTolerance. The optimality of a solution
        is proven if either of the two parameters' criteria is fulfilled.
        The value must be a float in [0..Infinity].
        Default value is 1e-09.
        """
        return self.get_attribute('OptimalityTolerance')

    OptimalityTolerance = property(get_OptimalityTolerance, set_OptimalityTolerance)


    def set_PrecedenceInferenceLevel(self, val):
        """
        This parameter specifies the inference level for precedence constraints between interval variables
        extracted to the invoking CP instance. Possible values for this parameter are Default, Low, Basic,
        Medium, and Extended. The default value is Default, which allows the inference strength for
        precedence constraints between interval variables to be controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        self.set_attribute('PrecedenceInferenceLevel', val)

    def get_PrecedenceInferenceLevel(self):
        """ Value of parameter PrecedenceInferenceLevel, None if not defined.

        This parameter specifies the inference level for precedence constraints between interval variables
        extracted to the invoking CP instance. Possible values for this parameter are Default, Low, Basic,
        Medium, and Extended. The default value is Default, which allows the inference strength for
        precedence constraints between interval variables to be controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        return self.get_attribute('PrecedenceInferenceLevel')

    PrecedenceInferenceLevel = property(get_PrecedenceInferenceLevel, set_PrecedenceInferenceLevel)


    def set_Presolve(self, val):
        """
        This parameter controls the presolve of the model to produce more compact formulations and to
        achieve more domain reduction. Possible values for this parameter are On (presolve is activated) and
        Off (presolve is deactivated). The default value is On.
        The value must be a symbol in [Off, On].
        Default value is On.
        """
        self.set_attribute('Presolve', val)

    def get_Presolve(self):
        """ Value of parameter Presolve, None if not defined.

        This parameter controls the presolve of the model to produce more compact formulations and to
        achieve more domain reduction. Possible values for this parameter are On (presolve is activated) and
        Off (presolve is deactivated). The default value is On.
        The value must be a symbol in [Off, On].
        Default value is On.
        """
        return self.get_attribute('Presolve')

    Presolve = property(get_Presolve, set_Presolve)


    def set_PrintModelDetailsInMessages(self, val):
        """
        Whenever CP Optimizer prints an error or warning message, it can also print concerning part of the
        input model (in cpo file format). This parameter controls printing of this additional information.
        Possible values are On and Off, the default value is On. See also WarningLevel.
        The value must be a symbol in [Off, On].
        Default value is On.
        """
        self.set_attribute('PrintModelDetailsInMessages', val)

    def get_PrintModelDetailsInMessages(self):
        """ Value of parameter PrintModelDetailsInMessages, None if not defined.

        Whenever CP Optimizer prints an error or warning message, it can also print concerning part of the
        input model (in cpo file format). This parameter controls printing of this additional information.
        Possible values are On and Off, the default value is On. See also WarningLevel.
        The value must be a symbol in [Off, On].
        Default value is On.
        """
        return self.get_attribute('PrintModelDetailsInMessages')

    PrintModelDetailsInMessages = property(get_PrintModelDetailsInMessages, set_PrintModelDetailsInMessages)


    def set_RandomSeed(self, val):
        """
        The search uses some randomization in some strategies. This parameter sets the seed of the random
        generator used by these strategies Possible values range from 0 to Infinity. The default value is 0.
        The value must be an integer in [0..Infinity].
        Default value is 0.
        """
        self.set_attribute('RandomSeed', val)

    def get_RandomSeed(self):
        """ Value of parameter RandomSeed, None if not defined.

        The search uses some randomization in some strategies. This parameter sets the seed of the random
        generator used by these strategies Possible values range from 0 to Infinity. The default value is 0.
        The value must be an integer in [0..Infinity].
        Default value is 0.
        """
        return self.get_attribute('RandomSeed')

    RandomSeed = property(get_RandomSeed, set_RandomSeed)


    def set_RelativeOptimalityTolerance(self, val):
        """
        This parameter sets a relative tolerance on the objective value for optimization models. This means
        that when CP Optimizer reports an optimal solution found, then there is no solution which improves
        the objective by more than the absolute value of the objective times the value of this parameter.
        The default value of this parameter is 1e-4. This parameter is used in conjunction with
        OptimalityTolerance. The optimality of a solution is proven if either of the two parameters'
        criteria are fulfilled.
        The value must be a float in [0..Infinity].
        Default value is 0.0001.
        """
        self.set_attribute('RelativeOptimalityTolerance', val)

    def get_RelativeOptimalityTolerance(self):
        """ Value of parameter RelativeOptimalityTolerance, None if not defined.

        This parameter sets a relative tolerance on the objective value for optimization models. This means
        that when CP Optimizer reports an optimal solution found, then there is no solution which improves
        the objective by more than the absolute value of the objective times the value of this parameter.
        The default value of this parameter is 1e-4. This parameter is used in conjunction with
        OptimalityTolerance. The optimality of a solution is proven if either of the two parameters'
        criteria are fulfilled.
        The value must be a float in [0..Infinity].
        Default value is 0.0001.
        """
        return self.get_attribute('RelativeOptimalityTolerance')

    RelativeOptimalityTolerance = property(get_RelativeOptimalityTolerance, set_RelativeOptimalityTolerance)


    def set_RestartFailLimit(self, val):
        """
        When SearchType is set to Restart, a depth-first search is restarted after a certain number of
        failures. This parameter controls the number of failures that must occur before restarting search.
        Possible values range from 0 to Infinity. The default value is 100. This value can increase after
        each restart: see the parameter RestartGrowthFactor.
        The value must be an integer in [1..Infinity].
        Default value is 100.
        """
        self.set_attribute('RestartFailLimit', val)

    def get_RestartFailLimit(self):
        """ Value of parameter RestartFailLimit, None if not defined.

        When SearchType is set to Restart, a depth-first search is restarted after a certain number of
        failures. This parameter controls the number of failures that must occur before restarting search.
        Possible values range from 0 to Infinity. The default value is 100. This value can increase after
        each restart: see the parameter RestartGrowthFactor.
        The value must be an integer in [1..Infinity].
        Default value is 100.
        """
        return self.get_attribute('RestartFailLimit')

    RestartFailLimit = property(get_RestartFailLimit, set_RestartFailLimit)


    def set_RestartGrowthFactor(self, val):
        """
        When SearchType is set to Restart, a depth-first search is restarted after a certain number of
        failures. This parameter controls the increase of this number between restarts. If the last fail
        limit was f after a restart, for next run, the new fail limit will be f times the value of this
        parameter. Possible values of this parameter range from 1.0 to Infinity. The default value is 1.05.
        The initial fail limit can be controlled with the parameter RestartFailLimit.
        The value must be a float in [1..Infinity].
        Default value is 1.15.
        """
        self.set_attribute('RestartGrowthFactor', val)

    def get_RestartGrowthFactor(self):
        """ Value of parameter RestartGrowthFactor, None if not defined.

        When SearchType is set to Restart, a depth-first search is restarted after a certain number of
        failures. This parameter controls the increase of this number between restarts. If the last fail
        limit was f after a restart, for next run, the new fail limit will be f times the value of this
        parameter. Possible values of this parameter range from 1.0 to Infinity. The default value is 1.05.
        The initial fail limit can be controlled with the parameter RestartFailLimit.
        The value must be a float in [1..Infinity].
        Default value is 1.15.
        """
        return self.get_attribute('RestartGrowthFactor')

    RestartGrowthFactor = property(get_RestartGrowthFactor, set_RestartGrowthFactor)


    def set_SearchType(self, val):
        """
        This parameter determines the type of search that is applied when solving a problem. When set to
        DepthFirst, a regular depth-first search is applied. When set to Restart, a depth-first search that
        restarts from time to time is applied. When set to MultiPoint, a method that combines a set of -
        possibly partial - solutions is applied. When set to Auto in sequential mode, this value chooses the
        appropriate search method to be used. In general Auto will be the Restart search. The default value
        is Auto. In parallel mode (i.e, when the number of workers is greater than one - see the Workers
        parameter), the different searches described above are spread over the workers. When the value of
        SearchType is Auto, then the decision of choosing the search type for a worker is automatically
        made; otherwise, all workers execute the same type of search. Note that in the latter case, the
        workers will not do the same exploration due to some radomness introduced to break ties in decision
        making.
        The value must be a symbol in [DepthFirst, Restart, MultiPoint, Auto].
        Default value is Auto.
        """
        self.set_attribute('SearchType', val)

    def get_SearchType(self):
        """ Value of parameter SearchType, None if not defined.

        This parameter determines the type of search that is applied when solving a problem. When set to
        DepthFirst, a regular depth-first search is applied. When set to Restart, a depth-first search that
        restarts from time to time is applied. When set to MultiPoint, a method that combines a set of -
        possibly partial - solutions is applied. When set to Auto in sequential mode, this value chooses the
        appropriate search method to be used. In general Auto will be the Restart search. The default value
        is Auto. In parallel mode (i.e, when the number of workers is greater than one - see the Workers
        parameter), the different searches described above are spread over the workers. When the value of
        SearchType is Auto, then the decision of choosing the search type for a worker is automatically
        made; otherwise, all workers execute the same type of search. Note that in the latter case, the
        workers will not do the same exploration due to some radomness introduced to break ties in decision
        making.
        The value must be a symbol in [DepthFirst, Restart, MultiPoint, Auto].
        Default value is Auto.
        """
        return self.get_attribute('SearchType')

    SearchType = property(get_SearchType, set_SearchType)


    def set_SequenceInferenceLevel(self, val):
        """
        This parameter specifies the inference level for every constraint Sequence extracted to the invoked
        CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and Extended. The
        default value is Default, which allows the inference strength of all Sequence constraints to be
        controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        self.set_attribute('SequenceInferenceLevel', val)

    def get_SequenceInferenceLevel(self):
        """ Value of parameter SequenceInferenceLevel, None if not defined.

        This parameter specifies the inference level for every constraint Sequence extracted to the invoked
        CP instance. Possible values for this parameter are Default, Low, Basic, Medium, and Extended. The
        default value is Default, which allows the inference strength of all Sequence constraints to be
        controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        return self.get_attribute('SequenceInferenceLevel')

    SequenceInferenceLevel = property(get_SequenceInferenceLevel, set_SequenceInferenceLevel)


    def set_SolutionLimit(self, val):
        """
        This parameter limits the number of feasible solutions that are found before terminating a search.
        The possible values of this parameter range from 0 to Infinity. A value of Infinity does not set any
        limit. The value Infinity is the default value.
        The value must be an integer in [0..Infinity].
        Default value is Infinity.
        """
        self.set_attribute('SolutionLimit', val)

    def get_SolutionLimit(self):
        """ Value of parameter SolutionLimit, None if not defined.

        This parameter limits the number of feasible solutions that are found before terminating a search.
        The possible values of this parameter range from 0 to Infinity. A value of Infinity does not set any
        limit. The value Infinity is the default value.
        The value must be an integer in [0..Infinity].
        Default value is Infinity.
        """
        return self.get_attribute('SolutionLimit')

    SolutionLimit = property(get_SolutionLimit, set_SolutionLimit)


    def set_StateFunctionInferenceLevel(self, val):
        """
        This parameter specifies the inference level for constraints on state functions StateFunction
        extracted to the invoked CP instance. Possible values for this parameter are Default, Low, Basic,
        Medium, and Extended. The default value is Default, which allows the inference strength of all
        constraints on state functions StateFunction to be controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        self.set_attribute('StateFunctionInferenceLevel', val)

    def get_StateFunctionInferenceLevel(self):
        """ Value of parameter StateFunctionInferenceLevel, None if not defined.

        This parameter specifies the inference level for constraints on state functions StateFunction
        extracted to the invoked CP instance. Possible values for this parameter are Default, Low, Basic,
        Medium, and Extended. The default value is Default, which allows the inference strength of all
        constraints on state functions StateFunction to be controlled via DefaultInferenceLevel.
        The value must be a symbol in [Default, Low, Basic, Medium, Extended].
        Default value is Default.
        """
        return self.get_attribute('StateFunctionInferenceLevel')

    StateFunctionInferenceLevel = property(get_StateFunctionInferenceLevel, set_StateFunctionInferenceLevel)


    def set_TemporalRelaxation(self, val):
        """
        This advanced parameter can be used to control the usage of a temporal relaxation internal to the
        invoking CP engine. This parameter can take values On or Off, with On being the default, meaning the
        relaxation is used in the engine when needed. For some models, using the relaxation becomes
        inefficient, and you may deactivate the use of the temporal relaxation using value Off.
        The value must be a symbol in [Off, On].
        Default value is On.
        """
        self.set_attribute('TemporalRelaxation', val)

    def get_TemporalRelaxation(self):
        """ Value of parameter TemporalRelaxation, None if not defined.

        This advanced parameter can be used to control the usage of a temporal relaxation internal to the
        invoking CP engine. This parameter can take values On or Off, with On being the default, meaning the
        relaxation is used in the engine when needed. For some models, using the relaxation becomes
        inefficient, and you may deactivate the use of the temporal relaxation using value Off.
        The value must be a symbol in [Off, On].
        Default value is On.
        """
        return self.get_attribute('TemporalRelaxation')

    TemporalRelaxation = property(get_TemporalRelaxation, set_TemporalRelaxation)


    def set_TimeLimit(self, val):
        """
        This parameter limits the CPU time spent solving before terminating a search. The time is given in
        seconds. The possible values of this parameter range from 0 to Infinity. A value of Infinity does
        not set any limit. The value Infinity is the default value.
        The value must be a float in [0..Infinity].
        Default value is Infinity.
        """
        self.set_attribute('TimeLimit', val)

    def get_TimeLimit(self):
        """ Value of parameter TimeLimit, None if not defined.

        This parameter limits the CPU time spent solving before terminating a search. The time is given in
        seconds. The possible values of this parameter range from 0 to Infinity. A value of Infinity does
        not set any limit. The value Infinity is the default value.
        The value must be a float in [0..Infinity].
        Default value is Infinity.
        """
        return self.get_attribute('TimeLimit')

    TimeLimit = property(get_TimeLimit, set_TimeLimit)


    def set_TimeMode(self, val):
        """
        This parameter defines how time is measured in CP Optimizer, the two legal values being ElapsedTime
        and CPUTime. CP Optimizer uses time for both display purposes and for limiting the search via
        TimeLimit. Note that when multiple processors are available and the number of workers (Workers) is
        greater than one, then the CPU time can be greater than the elapsed time by a factor up to the
        number of workers. The default value is ElapsedTime.
        The value must be a symbol in [CPUTime, ElapsedTime].
        Default value is ElapsedTime.
        """
        self.set_attribute('TimeMode', val)

    def get_TimeMode(self):
        """ Value of parameter TimeMode, None if not defined.

        This parameter defines how time is measured in CP Optimizer, the two legal values being ElapsedTime
        and CPUTime. CP Optimizer uses time for both display purposes and for limiting the search via
        TimeLimit. Note that when multiple processors are available and the number of workers (Workers) is
        greater than one, then the CPU time can be greater than the elapsed time by a factor up to the
        number of workers. The default value is ElapsedTime.
        The value must be a symbol in [CPUTime, ElapsedTime].
        Default value is ElapsedTime.
        """
        return self.get_attribute('TimeMode')

    TimeMode = property(get_TimeMode, set_TimeMode)


    def set_UseFileLocations(self, val):
        """
        This parameter controls whether CP Optimizer processes file locations. With each constraint,
        variable or expression it is possible to associate a source file location (file name and line
        number). CP Optimizer can use locations later for reporting errors and conflicts. Locations are also
        included in exported/dumped models (#line directives). Legal values for this parameter are On (the
        default) and Off. When the value is Off then CP Optimizer ignores locations in the input model and
        also does not export them in CPO file format (functions dumpModel and exportModel).
        The value must be a symbol in [Off, On].
        Default value is On.
        """
        self.set_attribute('UseFileLocations', val)

    def get_UseFileLocations(self):
        """ Value of parameter UseFileLocations, None if not defined.

        This parameter controls whether CP Optimizer processes file locations. With each constraint,
        variable or expression it is possible to associate a source file location (file name and line
        number). CP Optimizer can use locations later for reporting errors and conflicts. Locations are also
        included in exported/dumped models (#line directives). Legal values for this parameter are On (the
        default) and Off. When the value is Off then CP Optimizer ignores locations in the input model and
        also does not export them in CPO file format (functions dumpModel and exportModel).
        The value must be a symbol in [Off, On].
        Default value is On.
        """
        return self.get_attribute('UseFileLocations')

    UseFileLocations = property(get_UseFileLocations, set_UseFileLocations)


    def set_WarningLevel(self, val):
        """
        This parameter controls the level of warnings issued by CP Optimizer when a solve is launched.
        Specifically, all warnings of level higher than this parameter are masked. Since CP Optimizer
        warning levels run from 1 to 3, setting this parameter to 0 turns off all warnings. Warnings issued
        may indicate potential errors or inefficiencies in your model. The default value of this parameter
        is 2. See also PrintModelDetailsInMessages.
        The value must be an integer in [0..3].
        Default value is 2.
        """
        self.set_attribute('WarningLevel', val)

    def get_WarningLevel(self):
        """ Value of parameter WarningLevel, None if not defined.

        This parameter controls the level of warnings issued by CP Optimizer when a solve is launched.
        Specifically, all warnings of level higher than this parameter are masked. Since CP Optimizer
        warning levels run from 1 to 3, setting this parameter to 0 turns off all warnings. Warnings issued
        may indicate potential errors or inefficiencies in your model. The default value of this parameter
        is 2. See also PrintModelDetailsInMessages.
        The value must be an integer in [0..3].
        Default value is 2.
        """
        return self.get_attribute('WarningLevel')

    WarningLevel = property(get_WarningLevel, set_WarningLevel)


    def set_Workers(self, val):
        """
        This parameter sets the number of workers to run in parallel to solve your model. If the number of
        workers is set to n (with n greater than one), the CP optimizer will create n workers, each in their
        own thread, that will work together to solve the problem. The emphasis of these workers is more to
        find better feasible solutions and then to speed up the proof of optimality. The default value is
        Auto. This amounts to using as many workers as there are CPU cores available on the machine. Note
        that the memory required by CP Optimizer grows roughly linearly as the number of workers is
        increased. If you are solving a very large model on a multi-core processor and memory usage is an
        issue, it is advisable to specify a reduced number of workers, or even one worker, rather than use
        the default value.
        The value must be an integer in [1..Infinity].
        Default value is Auto.
        """
        self.set_attribute('Workers', val)

    def get_Workers(self):
        """ Value of parameter Workers, None if not defined.

        This parameter sets the number of workers to run in parallel to solve your model. If the number of
        workers is set to n (with n greater than one), the CP optimizer will create n workers, each in their
        own thread, that will work together to solve the problem. The emphasis of these workers is more to
        find better feasible solutions and then to speed up the proof of optimality. The default value is
        Auto. This amounts to using as many workers as there are CPU cores available on the machine. Note
        that the memory required by CP Optimizer grows roughly linearly as the number of workers is
        increased. If you are solving a very large model on a multi-core processor and memory usage is an
        issue, it is advisable to specify a reduced number of workers, or even one worker, rather than use
        the default value.
        The value must be an integer in [1..Infinity].
        Default value is Auto.
        """
        return self.get_attribute('Workers')

    Workers = property(get_Workers, set_Workers)


# List of all available parameter names
ALL_PARAMETER_NAMES = ("AllDiffInferenceLevel", "AllMinDistanceInferenceLevel", "AutomaticReplay", "BranchLimit", "ChoicePointLimit", "ConflictRefinerBranchLimit", "ConflictRefinerFailLimit", "ConflictRefinerIterationLimit", "ConflictRefinerOnVariables", "ConflictRefinerTimeLimit", "CountDifferentInferenceLevel", "CountInferenceLevel", "CumulFunctionInferenceLevel", "DefaultInferenceLevel", "DistributeInferenceLevel", "DynamicProbing", "DynamicProbingStrength", "ElementInferenceLevel", "FailLimit", "FailureDirectedSearch", "FailureDirectedSearchEmphasis", "FailureDirectedSearchMaxMemory", "IntervalSequenceInferenceLevel", "LogPeriod", "LogSearchTags", "LogVerbosity", "ModelAnonymizer", "MultiPointNumberOfSearchPoints", "NoOverlapInferenceLevel", "OptimalityTolerance", "PrecedenceInferenceLevel", "Presolve", "PrintModelDetailsInMessages", "RandomSeed", "RelativeOptimalityTolerance", "RestartFailLimit", "RestartGrowthFactor", "SearchType", "SequenceInferenceLevel", "SolutionLimit", "StateFunctionInferenceLevel", "TemporalRelaxation", "TimeLimit", "TimeMode", "UseFileLocations", "WarningLevel", "Workers",)
