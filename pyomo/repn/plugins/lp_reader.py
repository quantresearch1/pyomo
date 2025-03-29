import re
import numpy as np
from scipy.sparse import csr_matrix
from pyomo.environ import *
from pyomo.core.base.matrix_constraint import MatrixConstraint


class LPParser:
    """
    A parser for LP format files to extract the constraint matrix and other
    components for use with Pyomo's MatrixConstraint.
    """

    def __init__(self, filename):
        self.filename = filename
        self.variables = []
        self.objective = {}
        self.constraints = []
        self.bounds = {}
        self.var_type = {}
        self.sense = None
        self.rhs = []
        self.constraint_names = []

    def parse(self):
        """Parse the LP file and extract all components."""
        with open(self.filename, 'r') as f:
            content = f.read()

        # Split the file into sections
        sections = self._split_sections(content)

        # Parse each section
        self._parse_objective(sections.get('objective', ''))
        self._parse_constraints(sections.get('constraints', ''))
        self._parse_bounds(sections.get('bounds', ''))
        self._parse_variable_types(sections.get('integer', ''), 'Integer')
        self._parse_variable_types(sections.get('binary', ''), 'Binary')

        return self

    def _split_sections(self, content):
        """Split the LP file into its main sections."""
        sections = {}

        # Find objective function section
        obj_match = re.search(
            r'(Minimize|Maximize)\s*:(.*?)(?:Subject To|s\.t\.)',
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if obj_match:
            sections['objective'] = obj_match.group(2).strip()
            self.sense = (
                'minimize' if obj_match.group(1).lower() == 'minimize' else 'maximize'
            )

        # Find constraints section
        const_match = re.search(
            r'(?:Subject To|s\.t\.)\s*:(.*?)(?:Bounds|Binary|Integer|End)',
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if const_match:
            sections['constraints'] = const_match.group(1).strip()

        # Find bounds section
        bounds_match = re.search(
            r'Bounds\s*:(.*?)(?:Binary|Integer|End)', content, re.DOTALL | re.IGNORECASE
        )
        if bounds_match:
            sections['bounds'] = bounds_match.group(1).strip()

        # Find integer variables section
        int_match = re.search(
            r'Integer(?:\s+Variables)?\s*:(.*?)(?:Binary|Bounds|End)',
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if int_match:
            sections['integer'] = int_match.group(1).strip()

        # Find binary variables section
        bin_match = re.search(
            r'Binary(?:\s+Variables)?\s*:(.*?)(?:Integer|Bounds|End)',
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if bin_match:
            sections['binary'] = bin_match.group(1).strip()

        return sections

    def _parse_objective(self, obj_section):
        """Parse the objective function section."""
        # Remove line breaks and multiple spaces
        obj_section = ' '.join(obj_section.split())

        # Parse each term
        terms = re.findall(r'([+-]?\s*\d*\.?\d*)\s*\*?\s*(\w+)', obj_section)
        for coef, var in terms:
            # Clean and convert coefficient
            coef = coef.strip()
            if coef == '+':
                coef = '1'
            elif coef == '-':
                coef = '-1'
            elif coef == '':
                coef = '1'

            self.objective[var] = float(coef)

            # Add to variables list if not already there
            if var not in self.variables:
                self.variables.append(var)

    def _parse_constraints(self, const_section):
        """Parse the constraints section."""
        # Split into individual constraints
        constraint_lines = const_section.strip().split('\n')

        for line in constraint_lines:
            line = line.strip()
            if not line:
                continue

            # Check if constraint has a name
            name_match = re.match(r'(\w+)\s*:', line)
            constraint_name = None
            if name_match:
                constraint_name = name_match.group(1)
                line = line[name_match.end() :].strip()

            # Clean line
            line = ' '.join(line.split())

            # Find the inequality/equality sign
            sign_match = re.search(r'(<=|>=|=)', line)
            if not sign_match:
                continue

            sign = sign_match.group(1)
            lhs = line[: sign_match.start()].strip()
            rhs = line[sign_match.end() :].strip()

            # Parse the LHS terms
            terms = {}
            lhs_terms = re.findall(r'([+-]?\s*\d*\.?\d*)\s*\*?\s*(\w+)', lhs)
            for coef, var in lhs_terms:
                # Clean and convert coefficient
                coef = coef.strip()
                if coef == '+':
                    coef = '1'
                elif coef == '-':
                    coef = '-1'
                elif coef == '':
                    coef = '1'

                terms[var] = float(coef)

                # Add to variables list if not already there
                if var not in self.variables:
                    self.variables.append(var)

            # Parse the RHS value
            try:
                rhs_val = float(rhs)
            except ValueError:
                rhs_val = 0.0

            # Add constraint
            self.constraints.append(
                {'name': constraint_name, 'terms': terms, 'sign': sign, 'rhs': rhs_val}
            )

            self.constraint_names.append(
                constraint_name
                if constraint_name
                else f"constraint_{len(self.constraint_names)}"
            )
            self.rhs.append(rhs_val)

    def _parse_bounds(self, bounds_section):
        """Parse the bounds section."""
        if not bounds_section:
            return

        bounds_lines = bounds_section.strip().split('\n')

        for line in bounds_lines:
            line = line.strip()
            if not line:
                continue

            # Handle different bound formats
            # Format: lower <= var <= upper
            full_bound_match = re.match(
                r'(\d*\.?\d*)\s*<=\s*(\w+)\s*<=\s*(\d*\.?\d*)', line
            )
            if full_bound_match:
                lb, var, ub = full_bound_match.groups()
                self.bounds[var] = (float(lb), float(ub))
                if var not in self.variables:
                    self.variables.append(var)
                continue

            # Format: var <= upper
            upper_bound_match = re.match(r'(\w+)\s*<=\s*(\d*\.?\d*)', line)
            if upper_bound_match:
                var, ub = upper_bound_match.groups()
                lb = self.bounds.get(var, (0.0, None))[0]
                self.bounds[var] = (lb, float(ub))
                if var not in self.variables:
                    self.variables.append(var)
                continue

            # Format: lower <= var
            lower_bound_match = re.match(r'(\d*\.?\d*)\s*<=\s*(\w+)', line)
            if lower_bound_match:
                lb, var = lower_bound_match.groups()
                ub = self.bounds.get(var, (None, None))[1]
                self.bounds[var] = (float(lb), ub)
                if var not in self.variables:
                    self.variables.append(var)
                continue

            # Format: var free
            free_match = re.match(r'(\w+)\s+free', line)
            if free_match:
                var = free_match.group(1)
                self.bounds[var] = (None, None)
                if var not in self.variables:
                    self.variables.append(var)

    def _parse_variable_types(self, section, var_type):
        """Parse variable type sections (Integer or Binary)."""
        if not section:
            return

        # Clean and split the section
        clean_section = ' '.join(section.split())
        var_list = clean_section.split()

        for var in var_list:
            if var:
                self.var_type[var] = var_type
                if var not in self.variables:
                    self.variables.append(var)

    def to_csr_matrix(self):
        """
        Convert the constraint data to CSR matrix format for use with Pyomo's
        MatrixConstraint.

        Returns:
            data, indices, indptr, lb, ub - components for MatrixConstraint
        """
        # Map variable names to indices
        var_indices = {var: i for i, var in enumerate(self.variables)}

        # Initialize lists for CSR matrix
        data = []
        indices = []
        indptr = [0]
        lb = []
        ub = []

        # Process each constraint
        for constraint in self.constraints:
            row_data = []
            row_indices = []

            for var, coef in constraint['terms'].items():
                row_data.append(coef)
                row_indices.append(var_indices[var])

            # Sort by column index (variable index)
            sorted_pairs = sorted(zip(row_indices, row_data), key=lambda x: x[0])

            if sorted_pairs:
                row_indices, row_data = zip(*sorted_pairs)
                data.extend(row_data)
                indices.extend(row_indices)

            indptr.append(len(data))

            # Set bounds based on the constraint sign
            if constraint['sign'] == '<=':
                lb.append(None)
                ub.append(constraint['rhs'])
            elif constraint['sign'] == '>=':
                lb.append(constraint['rhs'])
                ub.append(None)
            else:  # '='
                lb.append(constraint['rhs'])
                ub.append(constraint['rhs'])

        return data, indices, indptr, lb, ub

    def build_pyomo_model(self):
        """
        Build a Pyomo model using the parsed LP file.

        Returns:
            Pyomo ConcreteModel
        """
        model = ConcreteModel()

        # Create variables
        model.x = Var(range(len(self.variables)))

        # Set variable bounds and types
        for i, var_name in enumerate(self.variables):
            bounds = self.bounds.get(var_name, (0.0, None))
            model.x[i].setlb(bounds[0])
            if bounds[1] is not None:
                model.x[i].setub(bounds[1])

            var_type = self.var_type.get(var_name)
            if var_type == 'Binary':
                model.x[i] = Var(domain=Binary)
            elif var_type == 'Integer':
                model.x[i] = Var(domain=Integers)

        # Create objective
        obj_expr = 0
        for var_name, coef in self.objective.items():
            idx = self.variables.index(var_name)
            obj_expr += coef * model.x[idx]

        if self.sense == 'maximize':
            model.obj = Objective(expr=obj_expr, sense=maximize)
        else:
            model.obj = Objective(expr=obj_expr, sense=minimize)

        # Create constraints using MatrixConstraint
        data, indices, indptr, lb, ub = self.to_csr_matrix()
        x_vars = [model.x[i] for i in range(len(self.variables))]
        model.constraints = MatrixConstraint(data, indices, indptr, lb, ub, x_vars)

        return model

    def get_variable_map(self):
        """
        Returns a dictionary mapping Pyomo variable indices to original variable names.
        """
        return {i: name for i, name in enumerate(self.variables)}


def lp_to_pyomo_matrix(lp_file):
    """
    Parse an LP file and return components for MatrixConstraint and a Pyomo model.

    Args:
        lp_file: Path to the LP format file

    Returns:
        data, indices, indptr, lb, ub, x_vars, model - components for MatrixConstraint
        and a Pyomo model
    """
    parser = LPParser(lp_file)
    parser.parse()

    data, indices, indptr, lb, ub = parser.to_csr_matrix()
    model = parser.build_pyomo_model()
    x_vars = [model.x[i] for i in range(len(parser.variables))]
    var_map = parser.get_variable_map()

    return data, indices, indptr, lb, ub, x_vars, model, var_map


# Example usage:
if __name__ == "__main__":
    # Example LP file
    lp_file = "D:\Dev\pyomo\model_v2.lp"

    # Parse the LP file and create a Pyomo model
    data, indices, indptr, lb, ub, x_vars, model, var_map = lp_to_pyomo_matrix(lp_file)

    # Print the components
    print("Data:", data)
    print("Indices:", indices)
    print("Indptr:", indptr)
    print("Lower bounds:", lb)
    print("Upper bounds:", ub)
    print("Variable mapping:", var_map)

    # Solve the model
    solver = SolverFactory('gurobi_direct')
    results = solver.solve(model)

    # Print results
    print("\nResults:")
    print("Solver Status:", results.solver.status)
    print("Termination Condition:", results.solver.termination_condition)

    # Print variable values with original names
    print("\nVariable Values:")
    for i, name in var_map.items():
        print(f"{name} = {value(model.x[i])}")
