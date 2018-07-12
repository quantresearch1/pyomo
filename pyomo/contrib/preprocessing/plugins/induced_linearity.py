"""Transformation to reformulate nonlinear models with linearity induced from
discrete variables.

Ref: Grossmann, IE; Voudouris, VT; Ghattas, O. Mixed integer linear
reformulations for some nonlinear discrete design optimization problems.

"""

from __future__ import division

import textwrap
from math import fabs

from pyomo.core.base import Block, Constraint, VarList, Objective
from pyomo.core.expr.current import ExpressionReplacementVisitor, identify_variables
from pyomo.core.expr.numvalue import value
from pyomo.core.kernel import ComponentMap, ComponentSet
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn import generate_standard_repn
from pyomo.common.plugin import alias
from pyomo.gdp import Disjunct
import logging

logger = logging.getLogger('pyomo.contrib.preprocessing')


class InducedLinearity(IsomorphicTransformation):
    """Reformulate nonlinear constraints with induced linearity.

    Finds continuous variables v where v = d1 + d2 + d3, where d's are discrete
    variables. These continuous variables may participate nonlinearly in other
    expressions, which may then be induced to be linear.

    The overall algorithm flow can be summarized as:
    1. Detect effectively discrete variables and the constraints that
    imply discreteness.
    2. Determine the set of valid values for each effectively discrete variable
        - NOTE: 1, 2 must incorporate scoping considerations (Disjuncts)
    3. Find nonlinear expressions in which effectively discrete variables
    participate.
    4. Reformulate nonlinear expressions appropriately.

    """

    alias('contrib.induced_linearity',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def _apply_to(self, model):
        """Apply the transformation to the given model."""
        equality_tolerance = 1E-6
        eff_discr_vars = detect_effectively_discrete_vars(
            model, equality_tolerance)
        # TODO will need to go through this for each disjunct, since it does
        # not (should not) descend into Disjuncts.

        # Determine the valid values for the effectively discrete variables
        determine_valid_values(eff_discr_vars)

        # Collect find bilinear expressions that can be reformulated using
        # knowledge of effectively discrete variables
        bilinear_map = _bilinear_expressions(model)

        # Relevant constraints are those with bilinear terms that involve
        # effectively_discrete_vars
        processed_pairs = ComponentSet()
        for v1, discrete_constr in effectively_discrete_vars:
            v1_pairs = bilinear_map.get(v1, ())
            for v2, bilinear_constrs in v1_pairs.items():
                if (v1, v2) in processed_pairs:
                    continue
                _process_bilinear_constraints(
                    v1, v2, discrete_constr, bilinear_constrs)
                processed_pairs.add((v2, v1))
                processed_pairs.add((v1, v2))  # TODO is this necessary?

        # Reformulate the bilinear terms
        pass


def determine_valid_values(block, discr_var_to_constrs_map):
    """Calculate valid values for each effectively discrete variable.

    We need the set of possible values for the effectively discrete variable in
    order to do the reformulations.

    Right now, we select a naive approach where we look for variables in the
    discreteness-inducing constraints. We then adjust their values and see if
    things are stil feasible. Based on their coefficient values, we can infer a
    set of allowable values for the effectively discrete variable.

    We try to make this more efficient by first constructing a mapping of
    variables to the constraints that they participate in.

    Args:
        block: The model or a disjunct on the model.

    """
    var_to_constraints_map = ComponentMap()
    if block.type() == Disjunct:
        # Get constraints from the disjunct's parent model
        add_constraints_to_map(var_to_constraints_map, block.model())
    # Get constraints from the disjunct (or model)
    add_constraints_to_map(var_to_constraints_map, block)

    # Go through
    for eff_discr_var, constrs in discr_var_to_constrs_map.items():
        pass


def add_constraints_to_map(var_to_constraints_map, block):
    for constr in block.model().component_data_objects(
            Constraint, active=True):
        for var in identify_variables(constr.body, include_fixed=False):
            constr_list = var_to_constraints_map.get(var, [])
            constr_list.append(constr)
            var_to_constraints_map[var] = constr_list


def _process_bilinear_constraints(v1, v2, discrete_constr, bilinear_constrs):
    # Categorize as case 1 or case 2
    for bilinear_constr in bilinear_constrs:
        # repn = generate_standard_repn(bilinear_constr.body)

        # Case 1: no other variables besides bilinear term in constraint. v1
        # (effectively discrete variable) is positive.
        # if (len(repn.quadratic_vars) == 1 and len(repn.linear_vars) == 0
        #         and repn.nonlinear_expr is None):
        #     _reformulate_case_1(v1, v2, discrete_constr, bilinear_constr)

        # NOTE: Case 1 is left unimplemented for now, because it involves some
        # messier logic with respect to how the transformation needs to happen.

        # Case 2: this is everything else, but do we want to have a special
        # case if there are nonlinear expressions involved with the constraint?
        pass
        if True:
            _reformulate_case_2(v1, v2, discrete_constr, bilinear_constr)
    pass


def _reformulate_case_1(v1, v2, discrete_constr, bilinear_constr):
    raise NotImplementedError()


def _reformulate_case_2(v1, v2, discrete_constr, bilinear_constr):
    pass


def _bilinear_expressions(model):
    # TODO for now, we look for only expressions where the bilinearities are
    # exposed on the root level SumExpression, and thus accessible via
    # generate_standard_repn. This will not detect exp(x*y). We require a
    # factorization transformation to be applied beforehand in order to pick
    # these constraints up.
    pass
    # Bilinear map will be stored in the format:
    # x --> (y --> [constr1, constr2, ...], z --> [constr2, constr3])
    bilinear_map = ComponentMap()
    for constr in model.component_data_objects(
            Constraint, active=True, descend_into=(Block, Disjunct)):
        if constr.body.polynomial_degree() in (1, 0):
            continue  # Skip trivial and linear constraints
        repn = generate_standard_repn(constr.body)
        for pair in repn.quadratic_vars:
            v1, v2 = pair
            v1_pairs = bilinear_map.get(v1, ComponentMap())
            if v2 in v1_pairs:
                # bilinear term has been found before. Simply add constraint to
                # the set associated with the bilinear term.
                v1_pairs[v2].add(constr)
            else:
                # We encounter the bilinear term for the first time.
                bilinear_map[v1] = v1_pairs
                bilinear_map[v2] = bilinear_map.get(v2, ComponentMap())
                constraints_with_bilinear_pair = ComponentSet([constr])
                bilinear_map[v1][v2] = constraints_with_bilinear_pair
                bilinear_map[v2][v1] = constraints_with_bilinear_pair
    return bilinear_map


def detect_effectively_discrete_vars(block, equality_tolerance):
    """Detect effectively discrete variables.

    These continuous variables are the sum of discrete variables.

    """
    # Map of effectively_discrete var --> inducing constraints
    effectively_discrete = ComponentMap()

    for constr in block.component_data_objects(Constraint, active=True):
        if constr.lower is None or constr.upper is None:
            continue  # skip inequality constraints
        if fabs(value(constr.lower) - value(constr.upper)
                ) > equality_tolerance:
            continue  # not equality constriant. Skip.
        if constr.body.polynomial_degree() not in (1, 0):
            continue  # skip nonlinear expressions
        repn = generate_standard_repn(constr.body)
        if len(repn.linear_vars) < 2:
            # TODO should this be < 2 or < 1?
            # TODO we should make sure that trivial equality relations are
            # preprocessed before this, or we will end up reformulating
            # expressions that we do not need to here.
            continue
        non_discrete_vars = list(v for v in repn.linear_vars
                                 if v.is_continuous())
        if len(non_discrete_vars) == 1:
            # We know that this is an effectively discrete continuous
            # variable. Add it to our identified variable list.
            var = non_discrete_vars[0]
            inducing_constraints = effectively_discrete.get(var, [])
            inducing_constraints.append(constr)
            effectively_discrete[var] = inducing_constraints
        # TODO we should eventually also look at cases where all other
        # non_discrete_vars are effectively_discrete_vars

    return effectively_discrete
