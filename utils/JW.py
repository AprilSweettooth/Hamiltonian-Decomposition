#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Jordan-Wigner transform on fermionic operators."""
import itertools
import numpy
from openfermion.ops.operators import QubitOperator
from openfermion.utils.operator_utils import count_qubits

def JW_transformation(iop, cancel=True):
    """Output InteractionOperator as QubitOperator class under JW transform.

    One could accomplish this very easily by first mapping to fermions and
    then mapping to qubits. We skip the middle step for the sake of speed.

    This only works for real InteractionOperators (no complex numbers).

    Returns:
        qubit_operator: An instance of the QubitOperator class.
    """
    # Initialize qubit operator as constant.
    number, coulomb, no_excitation, hopping, double_excitation = QubitOperator((), iop.constant), QubitOperator((), iop.constant), QubitOperator((), iop.constant), QubitOperator((), iop.constant), QubitOperator((), iop.constant) 
    # qubit_operator = QubitOperator((), iop.constant)
    double_excitations = []
    n_qubits=count_qubits(iop)
    # Transform diagonal one-body terms
    for p in range(n_qubits):
        coefficient = iop[(p, 1), (p, 0)]
        # qubit_operator += jordan_wigner_one_body(p, p, coefficient)
        number += jordan_wigner_one_body(p, p, coefficient) 

    # Transform other one-body terms and "diagonal" two-body terms
    for p, q in itertools.combinations(range(n_qubits), 2):
        # One-body
        coefficient = .5 * (iop[(p, 1), (q, 0)] + iop[(q, 1),
                                                      (p, 0)].conjugate())
        # qubit_operator += jordan_wigner_one_body(p, q, coefficient)
        hopping += jordan_wigner_one_body(p, q, coefficient)

        # Two-body
        coefficient = (iop[(p, 1), (q, 1), (p, 0),
                           (q, 0)] - iop[(p, 1), (q, 1), (q, 0),
                                         (p, 0)] - iop[(q, 1), (p, 1), (p, 0),
                                                       (q, 0)] + iop[(q, 1),
                                                                     (p, 1),
                                                                     (q, 0),
                                                                     (p, 0)])
        coulomb += jordan_wigner_two_body(p, q, p, q, coefficient)

    # Transform the rest of the two-body terms
    for (p, q), (r, s) in itertools.combinations(
            itertools.combinations(range(n_qubits), 2), 2):
        coefficient = 0.5 * (iop[(p, 1), (q, 1), (r, 0),
                                 (s, 0)] + iop[(s, 1), (r, 1), (q, 0),
                                               (p, 0)].conjugate() -
                             iop[(p, 1), (q, 1), (s, 0),
                                 (r, 0)] - iop[(r, 1), (s, 1), (q, 0),
                                               (p, 0)].conjugate() -
                             iop[(q, 1), (p, 1), (r, 0),
                                 (s, 0)] - iop[(s, 1), (r, 1), (p, 0),
                                               (q, 0)].conjugate() +
                             iop[(q, 1), (p, 1), (s, 0),
                                 (r, 0)] + iop[(r, 1), (s, 1), (p, 0),
                                               (q, 0)].conjugate())
        if len(set([p, q, r, s])) == 3:
            no_excitation += jordan_wigner_two_body(p, q, r, s, coefficient)
        elif len(set([p,q,r,s])) == 4:
            # print(p,q,r,s)
            # print(jordan_wigner_two_body(p, q, r, s, coefficient))
            if cancel:
                double_excitation += jordan_wigner_two_body(p, q, r, s, coefficient)
            else:
                scatter = jordan_wigner_two_body(p, q, r, s, coefficient)
                if list(scatter):
                    double_excitations.append(scatter)
    if cancel:
        return number, coulomb, hopping, no_excitation, double_excitation
    else:
        return number, coulomb, hopping, no_excitation, double_excitations


def jordan_wigner_one_body(p, q, coefficient=1.):
    r"""Map the term a^\dagger_p a_q + h.c. to QubitOperator.

    Note that the diagonal terms are divided by a factor of 2
    because they are equal to their own Hermitian conjugate.
    """
    # Handle off-diagonal terms.
    qubit_operator = QubitOperator()
    if p != q:
        if p > q:
            p, q = q, p
            coefficient = coefficient.conjugate()
        parity_string = tuple((z, 'Z') for z in range(p + 1, q))
        for c, (op_a, op_b) in [(coefficient.real, 'XX'),
                                (coefficient.real, 'YY'),
                                (coefficient.imag, 'YX'),
                                (-coefficient.imag, 'XY')]:
            operators = ((p, op_a),) + parity_string + ((q, op_b),)
            qubit_operator += QubitOperator(operators, .5 * c)

    # Handle diagonal terms.
    else:
        qubit_operator += QubitOperator((), .5 * coefficient)
        qubit_operator += QubitOperator(((p, 'Z'),), -.5 * coefficient)

    return qubit_operator


def jordan_wigner_two_body(p, q, r, s, coefficient=1.):
    r"""Map the term a^\dagger_p a^\dagger_q a_r a_s + h.c. to QubitOperator.

    Note that the diagonal terms are divided by a factor of two
    because they are equal to their own Hermitian conjugate.
    """
    # Initialize qubit operator.
    qubit_operator = QubitOperator()
    # coulomb, no_excitation, double_excitation = QubitOperator(), QubitOperator(), QubitOperator() 

    # Return zero terms.
    if (p == q) or (r == s):
        return qubit_operator 

    # Handle case of four unique indices.
    if len(set([p, q, r, s])) == 4:
        if (p > q) ^ (r > s):
            coefficient *= -1
        # Loop through different operators which act on each tensor factor.
        for ops in itertools.product('XY', repeat=4):
            # Get coefficients.
            if ops.count('X') % 2:
                coeff = .125 * coefficient.imag
                if ''.join(ops) in ['XYXX', 'YXXX', 'YYXY', 'YYYX']:
                    coeff *= -1
            else:
                coeff = .125 * coefficient.real
                if ''.join(ops) not in ['XXYY', 'YYXX']:
                    coeff *= -1
            if not coeff:
                continue
            
            # Sort operators.
            [(a, operator_a), (b, operator_b), (c, operator_c),
             (d, operator_d)] = sorted(zip([p, q, r, s], ops))
            # print((a, operator_a))
            # Compute operator strings.
            operators = ((a, operator_a),)
            operators += tuple((z, 'Z') for z in range(a + 1, b))
            
            operators += ((b, operator_b),)
            operators += ((c, operator_c),)
            operators += tuple((z, 'Z') for z in range(c + 1, d))
            operators += ((d, operator_d),)
            # print(operators)
            # Add term.
            # print(coeff)
            qubit_operator  += QubitOperator(operators, coeff)

    # Handle case of three unique indices.
    elif len(set([p, q, r, s])) == 3:

        # Identify equal tensor factors.
        if p == r:
            if q > s:
                a, b = s, q
                coefficient = -coefficient.conjugate()
            else:
                a, b = q, s
                coefficient = -coefficient
            c = p
        elif p == s:
            if q > r:
                a, b = r, q
                coefficient = coefficient.conjugate()
            else:
                a, b = q, r
            c = p
        elif q == r:
            if p > s:
                a, b = s, p
                coefficient = coefficient.conjugate()
            else:
                a, b = p, s
            c = q
        elif q == s:
            if p > r:
                a, b = r, p
                coefficient = -coefficient.conjugate()
            else:
                a, b = p, r
                coefficient = -coefficient
            c = q

        # Get operators.
        parity_string = tuple((z, 'Z') for z in range(a + 1, b))
        pauli_z = QubitOperator(((c, 'Z'),))
        for c, (op_a, op_b) in [(coefficient.real, 'XX'),
                                (coefficient.real, 'YY'),
                                (coefficient.imag, 'YX'),
                                (-coefficient.imag, 'XY')]:
            operators = ((a, op_a),) + parity_string + ((b, op_b),)
            if not c:
                continue

            # Add term.
            hopping_term = QubitOperator(operators, c / 4)
            qubit_operator  -= pauli_z * hopping_term
            qubit_operator  += hopping_term
            # qubit_operator -= pauli_z * hopping_term
            # qubit_operator += hopping_term

    # Handle case of two unique indices.
    elif len(set([p, q, r, s])) == 2:

        # Get coefficient.
        if p == s:
            coeff = -.25 * coefficient
        else:
            coeff = .25 * coefficient

        # Add terms.
        qubit_operator -= QubitOperator((), coeff)
        qubit_operator += QubitOperator(((p, 'Z'),), coeff)
        qubit_operator += QubitOperator(((q, 'Z'),), coeff)
        qubit_operator -= QubitOperator(((min(q, p), 'Z'), (max(q, p), 'Z')),
                                        coeff)
        # qubitop -= QubitOperator((), coeff)
        # qubitop += QubitOperator(((p, 'Z'),), coeff)
        # qubitop += QubitOperator(((q, 'Z'),), coeff)
        # qubitop -= QubitOperator(((min(q, p), 'Z'), (max(q, p), 'Z')),
        #                                 coeff) 

    # return coulomb, no_excitation, double_excitation
    return qubit_operator 