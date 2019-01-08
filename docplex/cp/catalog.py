# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Generated automatically

"""
This module contains the Python descriptors of the different CPO types and operations.
"""

from docplex.cp.catalog_elements import *

# ----------------------------------------------------------------------------
# Descriptors of CPO types
# ----------------------------------------------------------------------------

Type_Unknown               = CpoType("Unknown")
Type_FloatExpr             = CpoType("FloatExpr")
Type_Float                 = CpoType("Float", iscst=True, htyps=(Type_FloatExpr,))
Type_IntExpr               = CpoType("IntExpr", htyps=(Type_FloatExpr,))
Type_IntVar                = CpoType("IntVar", isvar=True, htyps=(Type_FloatExpr, Type_IntExpr,))
Type_Int                   = CpoType("Int", iscst=True, htyps=(Type_Float, Type_FloatExpr, Type_IntExpr, Type_IntVar,))
Type_FloatVar              = CpoType("FloatVar", isvar=True, htyps=(Type_FloatExpr,))
Type_CumulExpr             = CpoType("CumulExpr")
Type_CumulAtom             = CpoType("CumulAtom", htyps=(Type_CumulExpr,))
Type_CumulFunction         = CpoType("CumulFunction", htyps=(Type_CumulExpr,))
Type_FloatExprArray        = CpoType("FloatExprArray", eltyp=Type_FloatExpr)
Type_IntExprArray          = CpoType("IntExprArray", htyps=(Type_FloatExprArray,), eltyp=Type_IntExpr)
Type_Constraint            = CpoType("Constraint")
Type_BoolExpr              = CpoType("BoolExpr", htyps=(Type_Constraint, Type_FloatExpr, Type_IntExpr,))
Type_IntervalVar           = CpoType("IntervalVar", isvar=True)
Type_SequenceVar           = CpoType("SequenceVar", isvar=True)
Type_StateFunction         = CpoType("StateFunction", isvar=True)
Type_IntervalVarArray      = CpoType("IntervalVarArray", eltyp=Type_IntervalVar)
Type_IntVarArray           = CpoType("IntVarArray", htyps=(Type_FloatExprArray, Type_IntExprArray,), eltyp=Type_IntVar)
Type_SegmentedFunction     = CpoType("SegmentedFunction", iscst=True)
Type_StepFunction          = CpoType("StepFunction", iscst=True)
Type_TransitionMatrix      = CpoType("TransitionMatrix", iscst=True)
Type_IntervalArray         = CpoType("IntervalArray")
Type_FloatArray            = CpoType("FloatArray", htyps=(Type_FloatExprArray,), eltyp=Type_Float)
Type_IntArray              = CpoType("IntArray", htyps=(Type_FloatArray, Type_FloatExprArray, Type_IntExprArray,), eltyp=Type_Int)
Type_SequenceVarArray      = CpoType("SequenceVarArray", eltyp=Type_SequenceVar)
Type_CumulAtomArray        = CpoType("CumulAtomArray", eltyp=Type_CumulAtom)
Type_Objective             = CpoType("Objective")
Type_TupleSet              = CpoType("TupleSet", iscst=True)
Type_IntValueEval          = CpoType("IntValueEval")
Type_IntValueChooser       = CpoType("IntValueChooser")
Type_IntValueSelector      = CpoType("IntValueSelector", htyps=(Type_IntValueChooser,))
Type_IntValueSelectorArray = CpoType("IntValueSelectorArray", htyps=(Type_IntValueChooser,), eltyp=Type_IntValueSelector)
Type_IntVarEval            = CpoType("IntVarEval")
Type_IntVarChooser         = CpoType("IntVarChooser")
Type_IntVarSelector        = CpoType("IntVarSelector", htyps=(Type_IntVarChooser,))
Type_IntVarSelectorArray   = CpoType("IntVarSelectorArray", htyps=(Type_IntVarChooser,), eltyp=Type_IntVarSelector)
Type_SearchPhase           = CpoType("SearchPhase")
Type_TimeInt               = CpoType("TimeInt", htyps=(Type_Float, Type_FloatExpr, Type_Int, Type_IntExpr, Type_IntVar,), bastyp=Type_Int)
Type_PositiveInt           = CpoType("PositiveInt", htyps=(Type_Float, Type_FloatExpr, Type_Int, Type_IntExpr, Type_IntVar,), bastyp=Type_Int)
Type_BoolInt               = CpoType("BoolInt", htyps=(Type_Float, Type_FloatExpr, Type_Int, Type_IntExpr, Type_IntVar,), bastyp=Type_Int)
Type_Python                = CpoType("Python")
Type_Identifier            = CpoType("Identifier", isvar=True)
Type_BoolExprArray         = CpoType("BoolExprArray", htyps=(Type_FloatExprArray, Type_IntExprArray,), eltyp=Type_BoolExpr)
Type_Bool                  = CpoType("Bool", iscst=True, htyps=(Type_BoolExpr, Type_Constraint, Type_FloatExpr, Type_IntExpr,))

ALL_TYPES = (Type_Bool, Type_BoolExpr, Type_BoolExprArray, Type_BoolInt, Type_Constraint, Type_CumulAtom, Type_CumulAtomArray, Type_CumulExpr, Type_CumulFunction, Type_Float, Type_FloatArray, Type_FloatExpr, Type_FloatExprArray, Type_FloatVar, Type_Identifier, Type_Int, Type_IntArray, Type_IntExpr, Type_IntExprArray, Type_IntValueChooser, Type_IntValueEval, Type_IntValueSelector, Type_IntValueSelectorArray, Type_IntVar, Type_IntVarArray, Type_IntVarChooser, Type_IntVarEval, Type_IntVarSelector, Type_IntVarSelectorArray, Type_IntervalArray, Type_IntervalVar, Type_IntervalVarArray, Type_Objective, Type_PositiveInt, Type_Python, Type_SearchPhase, Type_SegmentedFunction, Type_SequenceVar, Type_SequenceVarArray, Type_StateFunction, Type_StepFunction, Type_TimeInt, Type_TransitionMatrix, Type_TupleSet, Type_Unknown)

# ----------------------------------------------------------------------------
# Descriptors of CPO operations
# ----------------------------------------------------------------------------

Oper_abs                         = CpoOperation("abs", "abs", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntExpr,)),
                                                                          CpoSignature(Type_FloatExpr, (Type_FloatExpr,))) )
Oper_abstraction                 = CpoOperation("abstraction", "abstraction", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray, Type_IntExprArray, Type_IntArray, Type_Int)),) )
Oper_all_diff                    = CpoOperation("alldiff", "all_diff", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray,)),) )
Oper_all_min_distance            = CpoOperation("allMinDistance", "all_min_distance", None, -1, ( CpoSignature(Type_BoolExpr, (Type_IntExprArray, Type_Int)),) )
Oper_allowed_assignments         = CpoOperation("allowedAssignments", "allowed_assignments", None, -1, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntArray)),
                                                                                                         CpoSignature(Type_BoolExpr, (Type_IntExprArray, Type_TupleSet))) )
Oper_alternative                 = CpoOperation("alternative", "alternative", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVarArray, CpoParam(Type_IntExpr, dval=0))),) )
Oper_always_constant             = CpoOperation("alwaysConstant", "always_constant", None, -1, ( CpoSignature(Type_Constraint, (Type_StateFunction, Type_IntervalVar, CpoParam(Type_BoolInt, dval=0), CpoParam(Type_BoolInt, dval=0))),
                                                                                                 CpoSignature(Type_Constraint, (Type_StateFunction, Type_TimeInt, Type_TimeInt, CpoParam(Type_BoolInt, dval=0), CpoParam(Type_BoolInt, dval=0)))) )
Oper_always_equal                = CpoOperation("alwaysEqual", "always_equal", None, -1, ( CpoSignature(Type_Constraint, (Type_StateFunction, Type_IntervalVar, Type_PositiveInt, CpoParam(Type_BoolInt, dval=0), CpoParam(Type_BoolInt, dval=0))),
                                                                                           CpoSignature(Type_Constraint, (Type_StateFunction, Type_TimeInt, Type_TimeInt, Type_PositiveInt, CpoParam(Type_BoolInt, dval=0), CpoParam(Type_BoolInt, dval=0)))) )
Oper_always_in                   = CpoOperation("alwaysIn", "always_in", None, -1, ( CpoSignature(Type_Constraint, (Type_CumulExpr, Type_IntervalVar, Type_PositiveInt, Type_PositiveInt)),
                                                                                     CpoSignature(Type_Constraint, (Type_CumulExpr, Type_TimeInt, Type_TimeInt, Type_PositiveInt, Type_PositiveInt)),
                                                                                     CpoSignature(Type_Constraint, (Type_StateFunction, Type_IntervalVar, Type_PositiveInt, Type_PositiveInt)),
                                                                                     CpoSignature(Type_Constraint, (Type_StateFunction, Type_TimeInt, Type_TimeInt, Type_PositiveInt, Type_PositiveInt))) )
Oper_always_no_state             = CpoOperation("alwaysNoState", "always_no_state", None, -1, ( CpoSignature(Type_Constraint, (Type_StateFunction, Type_IntervalVar)),
                                                                                                CpoSignature(Type_Constraint, (Type_StateFunction, Type_TimeInt, Type_TimeInt))) )
Oper_before                      = CpoOperation("before", "before", None, -1, ( CpoSignature(Type_Constraint, (Type_SequenceVar, Type_IntervalVar, Type_IntervalVar)),) )
Oper_bool_abstraction            = CpoOperation("boolAbstraction", "bool_abstraction", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray, Type_IntExprArray, Type_IntArray)),) )
Oper_constant                    = CpoOperation("constant", "constant", "", -1, ( CpoSignature(Type_Int, (Type_Int,)),
                                                                                  CpoSignature(Type_Float, (Type_Float,))) )
Oper_coordinate_piecewise_linear = CpoOperation("coordinatePiecewiseLinear", "coordinate_piecewise_linear", None, -1, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_Float, Type_FloatArray, Type_FloatArray, Type_Float)),) )
Oper_count                       = CpoOperation("count", "count", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntExprArray, Type_Int)),) )
Oper_count_different             = CpoOperation("countDifferent", "count_different", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntExprArray,)),) )
Oper_cumul_range                 = CpoOperation("cumulRange", "cumul_range", None, -1, ( CpoSignature(Type_Constraint, (Type_CumulExpr, Type_IntExpr, Type_IntExpr)),) )
Oper_diff                        = CpoOperation("diff", "diff", "!=", 6, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntExpr)),
                                                                           CpoSignature(Type_BoolExpr, (Type_FloatExpr, Type_FloatExpr))) )
Oper_distribute                  = CpoOperation("distribute", "distribute", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray, Type_IntArray, Type_IntExprArray)),
                                                                                        CpoSignature(Type_Constraint, (Type_IntExprArray, Type_IntExprArray))) )
Oper_domain_max                  = CpoOperation("domainMax", "domain_max", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper_domain_min                  = CpoOperation("domainMin", "domain_min", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper_domain_size                 = CpoOperation("domainSize", "domain_size", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper_element                     = CpoOperation("element", "element", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntArray)),
                                                                                  CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExprArray)),
                                                                                  CpoSignature(Type_FloatExpr, (Type_IntExpr, Type_FloatArray))) )
Oper_end_at_end                  = CpoOperation("endAtEnd", "end_at_end", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_IntExpr, dval=0))),) )
Oper_end_at_start                = CpoOperation("endAtStart", "end_at_start", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_IntExpr, dval=0))),) )
Oper_end_before_end              = CpoOperation("endBeforeEnd", "end_before_end", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_IntExpr, dval=0))),) )
Oper_end_before_start            = CpoOperation("endBeforeStart", "end_before_start", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_IntExpr, dval=0))),) )
Oper_end_eval                    = CpoOperation("endEval", "end_eval", None, -1, ( CpoSignature(Type_FloatExpr, (Type_IntervalVar, Type_SegmentedFunction, CpoParam(Type_Float, dval=0))),) )
Oper_end_of                      = CpoOperation("endOf", "end_of", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, CpoParam(Type_Int, dval=0))),) )
Oper_end_of_next                 = CpoOperation("endOfNext", "end_of_next", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_end_of_prev                 = CpoOperation("endOfPrev", "end_of_prev", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_equal                       = CpoOperation("equal", "equal", "==", 6, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntExpr)),
                                                                             CpoSignature(Type_BoolExpr, (Type_FloatExpr, Type_FloatExpr))) )
Oper_equal_or_escape             = CpoOperation("equalOrEscape", "equal_or_escape", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExpr, Type_IntExpr, Type_Int)),) )
Oper_exotic_object               = CpoOperation("exoticObject", "exotic_object", None, -1, ( CpoSignature(Type_Constraint, ()),) )
Oper_explicit_value_eval         = CpoOperation("explicitValueEval", "explicit_value_eval", None, -1, ( CpoSignature(Type_IntValueEval, (Type_IntArray, Type_FloatArray, CpoParam(Type_Float, dval=0))),) )
Oper_explicit_var_eval           = CpoOperation("explicitVarEval", "explicit_var_eval", None, -1, ( CpoSignature(Type_IntVarEval, (Type_IntExprArray, Type_FloatArray, CpoParam(Type_Float, dval=0))),) )
Oper_exponent                    = CpoOperation("exponent", "exponent", None, -1, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr,)),) )
Oper_false                       = CpoOperation("false", "false", None, -1, ( CpoSignature(Type_BoolExpr, ()),) )
Oper_first                       = CpoOperation("first", "first", None, -1, ( CpoSignature(Type_Constraint, (Type_SequenceVar, Type_IntervalVar)),) )
Oper_float_div                   = CpoOperation("floatDiv", "float_div", "/", 3, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_FloatExpr)),) )
Oper_float_var                   = CpoOperation("floatVar", "float_var", None, -1, ( CpoSignature(Type_FloatExpr, (Type_Float, Type_Float)),) )
Oper_forbid_end                  = CpoOperation("forbidEnd", "forbid_end", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_StepFunction)),) )
Oper_forbid_extent               = CpoOperation("forbidExtent", "forbid_extent", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_StepFunction)),) )
Oper_forbid_start                = CpoOperation("forbidStart", "forbid_start", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_StepFunction)),) )
Oper_forbidden_assignments       = CpoOperation("forbiddenAssignments", "forbidden_assignments", None, -1, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntArray)),
                                                                                                             CpoSignature(Type_BoolExpr, (Type_IntExprArray, Type_TupleSet))) )
Oper_greater                     = CpoOperation("greater", "greater", ">", 5, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntExpr)),
                                                                                CpoSignature(Type_BoolExpr, (Type_FloatExpr, Type_FloatExpr))) )
Oper_greater_or_equal            = CpoOperation("greaterOrEqual", "greater_or_equal", ">=", 5, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntExpr)),
                                                                                                 CpoSignature(Type_BoolExpr, (Type_FloatExpr, Type_FloatExpr)),
                                                                                                 CpoSignature(Type_Constraint, (Type_PositiveInt, Type_CumulExpr)),
                                                                                                 CpoSignature(Type_Constraint, (Type_CumulExpr, Type_PositiveInt)),
                                                                                                 CpoSignature(Type_Constraint, (Type_CumulExpr, Type_IntExpr)),
                                                                                                 CpoSignature(Type_Constraint, (Type_IntExpr, Type_CumulExpr))) )
Oper_height_at_end               = CpoOperation("heightAtEnd", "height_at_end", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, Type_CumulExpr, CpoParam(Type_Int, dval=0))),) )
Oper_height_at_start             = CpoOperation("heightAtStart", "height_at_start", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, Type_CumulExpr, CpoParam(Type_Int, dval=0))),) )
Oper_if_then                     = CpoOperation("ifThen", "if_then", "=>", 6, ( CpoSignature(Type_BoolExpr, (Type_BoolExpr, Type_BoolExpr)),) )
Oper_impact_of_last_branch       = CpoOperation("impactOfLastBranch", "impact_of_last_branch", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper_int_div                     = CpoOperation("intDiv", "int_div", "div", 3, ( CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExpr)),) )
Oper_inverse                     = CpoOperation("inverse", "inverse", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray, Type_IntExprArray)),) )
Oper_isomorphism                 = CpoOperation("isomorphism", "isomorphism", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVarArray, Type_IntervalVarArray, CpoParam(Type_IntExprArray, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_last                        = CpoOperation("last", "last", None, -1, ( CpoSignature(Type_Constraint, (Type_SequenceVar, Type_IntervalVar)),) )
Oper_length_eval                 = CpoOperation("lengthEval", "length_eval", None, -1, ( CpoSignature(Type_FloatExpr, (Type_IntervalVar, Type_SegmentedFunction, CpoParam(Type_Float, dval=0))),) )
Oper_length_of                   = CpoOperation("lengthOf", "length_of", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, CpoParam(Type_Int, dval=0))),) )
Oper_length_of_next              = CpoOperation("lengthOfNext", "length_of_next", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_length_of_prev              = CpoOperation("lengthOfPrev", "length_of_prev", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_less                        = CpoOperation("less", "less", "<", 5, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntExpr)),
                                                                          CpoSignature(Type_BoolExpr, (Type_FloatExpr, Type_FloatExpr))) )
Oper_less_or_equal               = CpoOperation("lessOrEqual", "less_or_equal", "<=", 5, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntExpr)),
                                                                                           CpoSignature(Type_BoolExpr, (Type_FloatExpr, Type_FloatExpr)),
                                                                                           CpoSignature(Type_Constraint, (Type_CumulExpr, Type_PositiveInt)),
                                                                                           CpoSignature(Type_Constraint, (Type_PositiveInt, Type_CumulExpr)),
                                                                                           CpoSignature(Type_Constraint, (Type_CumulExpr, Type_IntExpr)),
                                                                                           CpoSignature(Type_Constraint, (Type_IntExpr, Type_CumulExpr))) )
Oper_lexicographic               = CpoOperation("lexicographic", "lexicographic", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray, Type_IntExprArray)),) )
Oper_log                         = CpoOperation("log", "log", None, -1, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr,)),) )
Oper_logical_and                 = CpoOperation("and", "logical_and", "&&", 7, ( CpoSignature(Type_BoolExpr, (Type_BoolExpr, Type_BoolExpr)),) )
Oper_logical_not                 = CpoOperation("not", "logical_not", "!", 1, ( CpoSignature(Type_BoolExpr, (Type_BoolExpr,)),) )
Oper_logical_or                  = CpoOperation("or", "logical_or", "||", 8, ( CpoSignature(Type_BoolExpr, (Type_BoolExpr, Type_BoolExpr)),) )
Oper_max                         = CpoOperation("max", "max", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExpr)),
                                                                          CpoSignature(Type_IntExpr, (Type_IntExprArray,)),
                                                                          CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_FloatExpr)),
                                                                          CpoSignature(Type_FloatExpr, (Type_FloatExprArray,))) )
Oper_maximize                    = CpoOperation("maximize", "maximize", None, -1, ( CpoSignature(Type_Objective, (Type_FloatExpr,)),) )
Oper_maximize_static_lex         = CpoOperation("maximizeStaticLex", "maximize_static_lex", None, -1, ( CpoSignature(Type_Objective, (Type_FloatExprArray,)),) )
Oper_min                         = CpoOperation("min", "min", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExpr)),
                                                                          CpoSignature(Type_IntExpr, (Type_IntExprArray,)),
                                                                          CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_FloatExpr)),
                                                                          CpoSignature(Type_FloatExpr, (Type_FloatExprArray,))) )
Oper_minimize                    = CpoOperation("minimize", "minimize", None, -1, ( CpoSignature(Type_Objective, (Type_FloatExpr,)),) )
Oper_minimize_static_lex         = CpoOperation("minimizeStaticLex", "minimize_static_lex", None, -1, ( CpoSignature(Type_Objective, (Type_FloatExprArray,)),) )
Oper_minus                       = CpoOperation("minus", "minus", "-", 4, ( CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExpr)),
                                                                            CpoSignature(Type_IntExpr, (Type_IntExpr,)),
                                                                            CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_FloatExpr)),
                                                                            CpoSignature(Type_FloatExpr, (Type_FloatExpr,)),
                                                                            CpoSignature(Type_CumulExpr, (Type_CumulExpr, Type_CumulExpr)),
                                                                            CpoSignature(Type_CumulExpr, (Type_CumulExpr,))) )
Oper_mod                         = CpoOperation("mod", "mod", "%", 3, ( CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExpr)),) )
Oper_no_overlap                  = CpoOperation("noOverlap", "no_overlap", None, -1, ( CpoSignature(Type_Constraint, (Type_SequenceVar, CpoParam(Type_TransitionMatrix, dval=0), CpoParam(Type_BoolInt, dval=0))),
                                                                                       CpoSignature(Type_Constraint, (Type_IntervalVarArray,))) )
Oper_overlap_length              = CpoOperation("overlapLength", "overlap_length", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_Int, dval=0))),
                                                                                               CpoSignature(Type_IntExpr, (Type_IntervalVar, Type_TimeInt, Type_TimeInt, CpoParam(Type_Int, dval=0)))) )
Oper_pack                        = CpoOperation("pack", "pack", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray, Type_IntExprArray, Type_IntArray, CpoParam(Type_IntExpr, dval=0))),) )
Oper_plus                        = CpoOperation("plus", "plus", "+", 4, ( CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExpr)),
                                                                          CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_FloatExpr)),
                                                                          CpoSignature(Type_CumulExpr, (Type_CumulExpr, Type_CumulExpr))) )
Oper_power                       = CpoOperation("power", "power", "^", 2, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_FloatExpr)),) )
Oper_presence_of                 = CpoOperation("presenceOf", "presence_of", None, -1, ( CpoSignature(Type_BoolExpr, (Type_IntervalVar,)),) )
Oper_previous                    = CpoOperation("previous", "previous", None, -1, ( CpoSignature(Type_Constraint, (Type_SequenceVar, Type_IntervalVar, Type_IntervalVar)),) )
Oper_pulse                       = CpoOperation("pulse", "pulse", None, -1, ( CpoSignature(Type_CumulAtom, (Type_TimeInt, Type_TimeInt, Type_PositiveInt)),
                                                                              CpoSignature(Type_CumulAtom, (Type_IntervalVar, Type_PositiveInt)),
                                                                              CpoSignature(Type_CumulAtom, (Type_IntervalVar, Type_PositiveInt, Type_PositiveInt))) )
Oper_range                       = CpoOperation("range", "range", None, -1, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_Float, Type_Float)),
                                                                              CpoSignature(Type_BoolExpr, (Type_FloatExpr, Type_Float, Type_Float))) )
Oper_same_common_subsequence     = CpoOperation("sameCommonSubsequence", "same_common_subsequence", None, -1, ( CpoSignature(Type_Constraint, (Type_SequenceVar, Type_SequenceVar)),
                                                                                                                CpoSignature(Type_Constraint, (Type_SequenceVar, Type_SequenceVar, Type_IntervalVarArray, Type_IntervalVarArray))) )
Oper_same_sequence               = CpoOperation("sameSequence", "same_sequence", None, -1, ( CpoSignature(Type_Constraint, (Type_SequenceVar, Type_SequenceVar)),
                                                                                             CpoSignature(Type_Constraint, (Type_SequenceVar, Type_SequenceVar, Type_IntervalVarArray, Type_IntervalVarArray))) )
Oper_scal_prod                   = CpoOperation("scalProd", "scal_prod", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntArray, Type_IntExprArray)),
                                                                                     CpoSignature(Type_IntExpr, (Type_IntExprArray, Type_IntArray)),
                                                                                     CpoSignature(Type_IntExpr, (Type_IntExprArray, Type_IntExprArray)),
                                                                                     CpoSignature(Type_FloatExpr, (Type_FloatArray, Type_FloatExprArray)),
                                                                                     CpoSignature(Type_FloatExpr, (Type_FloatExprArray, Type_FloatArray)),
                                                                                     CpoSignature(Type_FloatExpr, (Type_FloatExprArray, Type_FloatExprArray))) )
Oper_search_phase                = CpoOperation("searchPhase", "search_phase", None, -1, ( CpoSignature(Type_SearchPhase, (Type_IntExprArray,)),
                                                                                           CpoSignature(Type_SearchPhase, (Type_IntVarChooser, Type_IntValueChooser)),
                                                                                           CpoSignature(Type_SearchPhase, (Type_IntExprArray, Type_IntVarChooser, Type_IntValueChooser)),
                                                                                           CpoSignature(Type_SearchPhase, (Type_IntervalVarArray,)),
                                                                                           CpoSignature(Type_SearchPhase, (Type_SequenceVarArray,))) )
Oper_select_largest              = CpoOperation("selectLargest", "select_largest", None, -1, ( CpoSignature(Type_IntVarSelector, (Type_Float, Type_IntVarEval)),
                                                                                               CpoSignature(Type_IntVarSelector, (Type_IntVarEval, CpoParam(Type_Float, dval=0))),
                                                                                               CpoSignature(Type_IntValueSelector, (Type_Float, Type_IntValueEval)),
                                                                                               CpoSignature(Type_IntValueSelector, (Type_IntValueEval, CpoParam(Type_Float, dval=0)))) )
Oper_select_random_value         = CpoOperation("selectRandomValue", "select_random_value", None, -1, ( CpoSignature(Type_IntValueSelector, ()),) )
Oper_select_random_var           = CpoOperation("selectRandomVar", "select_random_var", None, -1, ( CpoSignature(Type_IntVarSelector, ()),) )
Oper_select_smallest             = CpoOperation("selectSmallest", "select_smallest", None, -1, ( CpoSignature(Type_IntVarSelector, (Type_Float, Type_IntVarEval)),
                                                                                                 CpoSignature(Type_IntVarSelector, (Type_IntVarEval, CpoParam(Type_Float, dval=0))),
                                                                                                 CpoSignature(Type_IntValueSelector, (Type_Float, Type_IntValueEval)),
                                                                                                 CpoSignature(Type_IntValueSelector, (Type_IntValueEval, CpoParam(Type_Float, dval=0)))) )
Oper_sequence                    = CpoOperation("sequence", "sequence", None, -1, ( CpoSignature(Type_Constraint, (Type_Int, Type_Int, Type_Int, Type_IntExprArray, Type_IntArray, Type_IntExprArray)),) )
Oper_size_eval                   = CpoOperation("sizeEval", "size_eval", None, -1, ( CpoSignature(Type_FloatExpr, (Type_IntervalVar, Type_SegmentedFunction, CpoParam(Type_Float, dval=0))),) )
Oper_size_of                     = CpoOperation("sizeOf", "size_of", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, CpoParam(Type_Int, dval=0))),) )
Oper_size_of_next                = CpoOperation("sizeOfNext", "size_of_next", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_size_of_prev                = CpoOperation("sizeOfPrev", "size_of_prev", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_slope_piecewise_linear      = CpoOperation("slopePiecewiseLinear", "slope_piecewise_linear", None, -1, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_FloatArray, Type_FloatArray, Type_Float, Type_Float)),) )
Oper_span                        = CpoOperation("span", "span", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVarArray)),) )
Oper_spread                      = CpoOperation("spread", "spread", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray, Type_FloatExpr, Type_FloatExpr)),) )
Oper_square                      = CpoOperation("square", "square", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntExpr,)),
                                                                                CpoSignature(Type_FloatExpr, (Type_FloatExpr,))) )
Oper_standard_deviation          = CpoOperation("standardDeviation", "standard_deviation", None, -1, ( CpoSignature(Type_FloatExpr, (Type_IntExprArray, Type_Float, Type_Float)),) )
Oper_start_at_end                = CpoOperation("startAtEnd", "start_at_end", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_IntExpr, dval=0))),) )
Oper_start_at_start              = CpoOperation("startAtStart", "start_at_start", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_IntExpr, dval=0))),) )
Oper_start_before_end            = CpoOperation("startBeforeEnd", "start_before_end", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_IntExpr, dval=0))),) )
Oper_start_before_start          = CpoOperation("startBeforeStart", "start_before_start", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_IntExpr, dval=0))),) )
Oper_start_eval                  = CpoOperation("startEval", "start_eval", None, -1, ( CpoSignature(Type_FloatExpr, (Type_IntervalVar, Type_SegmentedFunction, CpoParam(Type_Float, dval=0))),) )
Oper_start_of                    = CpoOperation("startOf", "start_of", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, CpoParam(Type_Int, dval=0))),) )
Oper_start_of_next               = CpoOperation("startOfNext", "start_of_next", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_start_of_prev               = CpoOperation("startOfPrev", "start_of_prev", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_step_at                     = CpoOperation("stepAt", "step_at", None, -1, ( CpoSignature(Type_CumulAtom, (Type_TimeInt, Type_PositiveInt)),) )
Oper_step_at_end                 = CpoOperation("stepAtEnd", "step_at_end", None, -1, ( CpoSignature(Type_CumulAtom, (Type_IntervalVar, Type_PositiveInt)),
                                                                                        CpoSignature(Type_CumulAtom, (Type_IntervalVar, Type_PositiveInt, Type_PositiveInt))) )
Oper_step_at_start               = CpoOperation("stepAtStart", "step_at_start", None, -1, ( CpoSignature(Type_CumulAtom, (Type_IntervalVar, Type_PositiveInt)),
                                                                                            CpoSignature(Type_CumulAtom, (Type_IntervalVar, Type_PositiveInt, Type_PositiveInt))) )
Oper_strong                      = CpoOperation("strong", "strong", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray,)),) )
Oper_sum                         = CpoOperation("sum", "sum", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntExprArray,)),
                                                                          CpoSignature(Type_FloatExpr, (Type_FloatExprArray,))) )
Oper_synchronize                 = CpoOperation("synchronize", "synchronize", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVarArray)),) )
Oper_times                       = CpoOperation("times", "times", "*", 3, ( CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExpr)),
                                                                            CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_FloatExpr))) )
Oper_true                        = CpoOperation("true", "true", None, -1, ( CpoSignature(Type_BoolExpr, ()),) )
Oper_type_of_next                = CpoOperation("typeOfNext", "type_of_next", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_type_of_prev                = CpoOperation("typeOfPrev", "type_of_prev", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_value                       = CpoOperation("value", "value", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper_value_impact                = CpoOperation("valueImpact", "value_impact", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper_value_index                 = CpoOperation("valueIndex", "value_index", None, -1, ( CpoSignature(Type_IntValueEval, (Type_IntArray, CpoParam(Type_Float, dval=-1))),) )
Oper_value_success_rate          = CpoOperation("valueSuccessRate", "value_success_rate", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper_var_impact                  = CpoOperation("varImpact", "var_impact", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper_var_index                   = CpoOperation("varIndex", "var_index", None, -1, ( CpoSignature(Type_IntVarEval, (Type_IntExprArray, CpoParam(Type_Float, dval=-1))),) )
Oper_var_local_impact            = CpoOperation("varLocalImpact", "var_local_impact", None, -1, ( CpoSignature(Type_IntVarEval, (CpoParam(Type_Int, dval=-1),)),) )
Oper_var_success_rate            = CpoOperation("varSuccessRate", "var_success_rate", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )

ALL_OPERATIONS = (Oper_abs, Oper_abstraction, Oper_all_diff, Oper_all_min_distance, Oper_allowed_assignments, Oper_alternative, Oper_always_constant, Oper_always_equal, Oper_always_in, Oper_always_no_state, Oper_before, Oper_bool_abstraction, Oper_constant, Oper_coordinate_piecewise_linear, Oper_count, Oper_count_different, Oper_cumul_range, Oper_diff, Oper_distribute, Oper_domain_max, Oper_domain_min, Oper_domain_size, Oper_element, Oper_end_at_end, Oper_end_at_start, Oper_end_before_end, Oper_end_before_start, Oper_end_eval, Oper_end_of, Oper_end_of_next, Oper_end_of_prev, Oper_equal, Oper_equal_or_escape, Oper_exotic_object, Oper_explicit_value_eval, Oper_explicit_var_eval, Oper_exponent, Oper_false, Oper_first, Oper_float_div, Oper_float_var, Oper_forbid_end, Oper_forbid_extent, Oper_forbid_start, Oper_forbidden_assignments, Oper_greater, Oper_greater_or_equal, Oper_height_at_end, Oper_height_at_start, Oper_if_then, Oper_impact_of_last_branch, Oper_int_div, Oper_inverse, Oper_isomorphism, Oper_last, Oper_length_eval, Oper_length_of, Oper_length_of_next, Oper_length_of_prev, Oper_less, Oper_less_or_equal, Oper_lexicographic, Oper_log, Oper_logical_and, Oper_logical_not, Oper_logical_or, Oper_max, Oper_maximize, Oper_maximize_static_lex, Oper_min, Oper_minimize, Oper_minimize_static_lex, Oper_minus, Oper_mod, Oper_no_overlap, Oper_overlap_length, Oper_pack, Oper_plus, Oper_power, Oper_presence_of, Oper_previous, Oper_pulse, Oper_range, Oper_same_common_subsequence, Oper_same_sequence, Oper_scal_prod, Oper_search_phase, Oper_select_largest, Oper_select_random_value, Oper_select_random_var, Oper_select_smallest, Oper_sequence, Oper_size_eval, Oper_size_of, Oper_size_of_next, Oper_size_of_prev, Oper_slope_piecewise_linear, Oper_span, Oper_spread, Oper_square, Oper_standard_deviation, Oper_start_at_end, Oper_start_at_start, Oper_start_before_end, Oper_start_before_start, Oper_start_eval, Oper_start_of, Oper_start_of_next, Oper_start_of_prev, Oper_step_at, Oper_step_at_end, Oper_step_at_start, Oper_strong, Oper_sum, Oper_synchronize, Oper_times, Oper_true, Oper_type_of_next, Oper_type_of_prev, Oper_value, Oper_value_impact, Oper_value_index, Oper_value_success_rate, Oper_var_impact, Oper_var_index, Oper_var_local_impact, Oper_var_success_rate)
SEARCH_PHASE_OPERATIONS = (Oper_search_phase)
