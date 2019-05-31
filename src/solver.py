"""
Generic solver methods for Markov Decision Processes (MDPs) and methods for
policy computations for GridWorld.
"""

import numpy as np


def value_iteration(p, reward, discount, eps=1e-3):
    """
    Basic value-iteration algorithm to solve the given MDP.

    Args:
        p: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        reward: The reward signal per state as table
            `[state: Integer] -> reward: Float`.
        discount: The discount (gamma) applied during value-iteration.
        eps: The threshold to be used as convergence criterion. Convergence
            is assumed if the value-function changes less than the threshold
            on all states in a single iteration.

    Returns:
        The value function as table `[state: Integer] -> value: Float`.
    """
    n_states, _, n_actions = p.shape
    v = np.zeros(n_states)

    # Setup transition probability matrices for easy use with numpy.
    #
    # This is an array of matrices, one matrix per action. Multiplying
    # state-values v(s) with one of these matrices P_a for action a represents
    # the equation
    #     P_a * [ v(s_i) ]_i^T = [ sum_k p(s_k | s_j, a) * v(s_K) ]_j^T
    p = [np.matrix(p[:, :, a]) for a in range(n_actions)]

    delta = np.inf
    while delta > eps:      # iterate until convergence
        v_old = v

        # compute state-action values (note: we actually have Q[a, s] here)
        q = discount * np.array([p[a] @ v for a in range(n_actions)])

        # compute state values
        v = reward + np.max(q, axis=0)[0]

        # compute maximum delta
        delta = np.max(np.abs(v_old - v))

    return v


def stochastic_value_iteration(p, reward, discount, eps=1e-3):
    """
    A modified version of the value-iteration algorithm to solve the given MDP.

    During iteration, this modified version computes the average over all
    state-action values instead of choosing the maximum. The modification is
    intended to give a better expectation of the value for an agent that
    chooses sub-optimal actions. It is intended as an alternative to the
    standard value-iteration for automated trajectory generation.

    Args:
        p: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        reward: The reward signal per state as table
            `[state: Integer] -> reward: Float`.
        discount: The discount (gamma) applied during value-iteration.
        eps: The threshold to be used as convergence criterion. Convergence
            is assumed if the value-function changes less than the threshold
            on all states in a single iteration.

    Returns:
        The value function as table `[state: Integer] -> value: Float`.
    """
    n_states, _, n_actions = p.shape
    v = np.zeros(n_states)

    # Setup transition probability matrices for easy use with numpy.
    #
    # This is an array of matrices, one matrix per action. Multiplying
    # state-values v(s) with one of these matrices P_a for action a represents
    # the equation
    #     P_a * [ v(s_i) ]_i^T = [ sum_k p(s_k | s_j, a) * v(s_K) ]_j^T
    p = [np.matrix(p[:, :, a]) for a in range(n_actions)]

    delta = np.inf
    while delta > eps:      # iterate until convergence
        v_old = v

        # compute state-action values (note: we actually have Q[a, s] here)
        q = discount * np.array([p[a] @ v for a in range(n_actions)])

        # compute state values
        v = reward + np.average(q, axis=0)[0]

        # compute maximum delta
        delta = np.max(np.abs(v_old - v))

    return v


def optimal_policy_from_value(world, value):
    """
    Compute the optimal policy from the given value function.

    Args:
        world: The `GridWorld` instance for which the the policy should be
            computed.
        value: The value-function dictating the policy as table
            `[state: Integer] -> value: Float`

    Returns:
        The optimal (deterministic) policy given the provided arguments as
        table `[state: Integer] -> action: Integer`.
    """
    policy = np.array([
        np.argmax([value[world.state_index_transition(s, a)] for a in range(world.n_actions)])
        for s in range(world.n_states)
    ])

    return policy


def optimal_policy(world, reward, discount, eps=1e-3):
    """
    Compute the optimal policy using value-iteration

    Args:
        world: The `GridWorld` instance for which the the policy should be
            computed.
        reward: The reward signal per state as table
            `[state: Integer] -> reward: Float`
        discount: The discount (gamma) applied during value-iteration.
        eps: The threshold to be used as convergence criterion. Convergence
            is assumed if the value-function changes less than the threshold
            on all states in a single iteration.

    Returns:
        The optimal (deterministic) policy given the provided arguments as
        table `[state: Integer] -> action: Integer`.

    See also:
        - `value_iteration`
        - `optimal_policy_from_value`
    """
    value = value_iteration(world.p_transition, reward, discount, eps)
    return optimal_policy_from_value(world, value)


def stochastic_policy_from_value(world, value, w=lambda x: x):
    """
    Compute a stochastic policy from the given value function.

    Args:
        world: The `GridWorld` instance for which the the policy should be
            computed.
        value: The value-function dictating the policy as table
            `[state: Integer] -> value: Float`
        w: A weighting function `(value: Float) -> value: Float` applied to
            all state-action values before normalizing the results, which
            are then used as probabilities. I.e. choosing `x -> x**2` here
            will cause the preference of suboptimal actions to decrease
            quadratically compared to the preference of the optimal action.

    Returns:
        The stochastic policy given the provided arguments as table
        `[state: Integer, action: Integer] -> probability: Float`
        describing a probability distribution p(action | state) of selecting
        an action given a state.
    """
    policy = np.array([
        np.array([w(value[world.state_index_transition(s, a)]) for a in range(world.n_actions)])
        for s in range(world.n_states)
    ])

    return policy / np.sum(policy, axis=1)[:, None]
