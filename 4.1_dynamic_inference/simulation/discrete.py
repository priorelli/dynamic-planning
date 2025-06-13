import numpy as np
import itertools
import utils
import config as c


# Define discrete class
class Discrete:
    def __init__(self):
        # Likelihood matrix
        pos_states = ['BALL', 'SQUARE']

        self.states = [pos_states]
        self.n_states = len(pos_states)
        self.state_to_idx, self.idx_to_state = self.get_idx()
        self.A = self.get_A()

        # Transition matrix
        self.actions = ['STAY']
        self.n_actions = len(self.actions)
        self.B = self.get_B()

        # Preference matrix
        self.C = np.ones(self.n_states)
        self.C /= self.n_states

        # Prior matrix
        self.D = np.ones(self.n_states)
        self.D /= self.n_states
        self.prior = self.D.copy()

        # Initialize policies and habit matrix
        self.policies = self.construct_policies()
        self.E = np.zeros(len(self.policies))

        # Compute entropy
        self.H_A = self.entropy()

        # Initialize observations and log evidences
        self.o_ext = self.get_expected_obs(self.prior)  # eq. 32
        self.L_ext = np.zeros(len(self.A))

        # Get state-index mappings
    def get_idx(self):
        state_to_idx = {}
        idx_to_state = {}
        c = 0
        for i in self.states[0]:
            state_to_idx[i] = c
            idx_to_state[c] = i
            c += 1

        return state_to_idx, idx_to_state

    # Get likelihood matrix
    def get_A(self):
        A = np.zeros((2, self.n_states))

        for state, idx in self.state_to_idx.items():
            if state == 'BALL':
                A[0, idx] = 1.0
            elif state == 'SQUARE':
                A[1, idx] = 1.0

        return A

    # Get transition matrix
    def get_B(self):
        B = np.zeros((self.n_states, self.n_states, self.n_actions))

        for state, idx in self.state_to_idx.items():
            for action_id, action_label in enumerate(self.actions):
                next_label = state

                next_idx = self.state_to_idx[next_label]
                B[next_idx, idx, action_id] = 1.0

        return B

    # Get all policies
    def construct_policies(self):
        x = [self.n_actions] * c.n_policy

        policies = list(itertools.product(*[list(range(i)) for i in x]))
        for pol_i in range(len(policies)):
            policies[pol_i] = np.array(policies[pol_i]).reshape(c.n_policy, 1)

        return policies

    # Compute likelihood entropy
    def entropy(self):
        H_A = - (self.A * utils.log_stable(self.A)).sum(axis=0)

        return H_A

    # Infer current states
    def infer_states(self, r_t):
        # Get expected state from observation
        qs_t = self.A.T.dot(r_t)

        log_prior = utils.log_stable(self.prior)
        log_post = utils.log_stable(qs_t)

        qs = utils.softmax(log_post + log_prior)

        return qs

    # Compute expected states
    def get_expected_states(self, qs_current, action):
        qs_u = self.B[:, :, action].dot(qs_current)

        return qs_u

    # Compute expected observations
    def get_expected_obs(self, qs_u):
        qo_u = self.A.dot(qs_u)

        return qo_u

    # Compute KL divergence
    def kl_divergence(self, qs_u):
        return (utils.log_stable(qs_u) - utils.log_stable(self.C)).dot(qs_u)

    # Compute expected free energy
    def compute_G(self, qs_current):
        G = np.zeros(len(self.policies))

        for policy_id, policy in enumerate(self.policies):
            qs_pi_t = 0

            for t in range(policy.shape[0]):
                action = policy[t, 0]
                qs_prev = qs_current if t == 0 else qs_pi_t

                qs_pi_t = self.get_expected_states(qs_prev, action)

                kld = self.kl_divergence(qs_pi_t)

                G[policy_id] += kld

        return G

    # Compute action posterior
    def compute_prob_actions(self, Q_pi):
        P_u = np.zeros(self.n_actions)

        for policy_id, policy in enumerate(self.policies):
            P_u[int(policy[0, 0])] += Q_pi[policy_id]

        P_u = utils.norm_dist(P_u)

        return P_u

    # Get next states
    def get_qs_next(self, P_u, qs_t):
        qs_next = np.zeros(self.n_states)

        for action_idx, prob in enumerate(P_u):
            qs_next += prob * self.B[:, :, action_idx].dot(qs_t)

        return qs_next

    # Run discrete step
    def step(self):
        # Perform BMC
        # q_r = utils.bmc(self.o_ext, self.L_ext, c.w_bmc,
        #                 c.gain_prior, c.gain_evidence)

        # Infer current state
        # qs_current = self.infer_states(q_r)

        # Compute expected free energy
        # G = self.compute_G(qs_current)

        # Marginalize P(u|pi)
        # Q_pi = utils.softmax(self.E - G)

        # Compute action posterior
        # P_u = self.compute_prob_actions(Q_pi)

        # Compute next observations
        # qs_next = self.get_qs_next(P_u, qs_current)
        # self.o_ext[:] = self.get_expected_obs(qs_next)

        # Clear evidence
        self.L_ext[:] = 0
