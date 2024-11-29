import numpy as np
import itertools
import utils
import config as c


# Define discrete class
class Discrete:
    def __init__(self):
        # Likelihood matrix
        pos_states = ['FREE', 'TOOL', 'BALL']
        tool_states = ['FREE', 'PICKED']

        self.states = [pos_states, tool_states]
        self.n_states = len(pos_states) * len(tool_states)
        self.state_to_idx, self.idx_to_state = self.get_idx()
        self.A_ext = self.get_A_ext()
        self.A_tact = self.get_A_tact()

        # Transition matrix
        self.actions = ['STAY', 'TO_TOOL', 'TO_BALL']
        self.n_actions = len(self.actions)
        self.P_u = np.zeros(self.n_actions)
        self.B = self.get_B()

        # Preference matrix
        self.reward_state = ('BALL', 'PICKED')
        reward_idx = self.state_to_idx[self.reward_state]

        self.C = np.zeros(self.n_states)
        self.C[reward_idx] = 1.0

        # Prior matrix
        self.start_state = ('FREE', 'FREE')
        start_idx = self.state_to_idx[self.start_state]

        self.D = np.zeros(self.n_states)
        self.D[start_idx] = 1.0
        self.prior = self.D.copy()

        # Initialize policies and habit matrix
        self.policies = self.construct_policies()
        self.E = np.zeros(len(self.policies))

        # Compute entropy
        self.H_A = self.entropy()

        # Initialize observations and log evidences
        self.o_int = np.zeros(1)
        self.o_ext = self.get_expected_obs(self.A_ext, self.prior)

        self.L_int = np.zeros(1)
        self.L_ext = np.zeros((c.n_joints + 1, len(self.A_ext)))

    # Get state-index mappings
    def get_idx(self):
        state_to_idx = {}
        idx_to_state = {}
        c = 0
        for i in self.states[0]:
            for j in self.states[1]:
                state_to_idx[(i, j)] = c
                idx_to_state[c] = (i, j)
                c += 1

        return state_to_idx, idx_to_state

    # Get likelihood extrinsic matrix
    def get_A_ext(self):
        A = np.zeros((3, self.n_states))

        for state, idx in self.state_to_idx.items():
            if state[0] == 'FREE':
                A[0, idx] = 1.0
            elif state[0] == 'TOOL':
                A[1, idx] = 1.0
            else:
                A[2, idx] = 1.0

        return A

    # Get likelihood tactile matrix
    def get_A_tact(self):
        A = np.zeros((2, self.n_states))

        for state, idx in self.state_to_idx.items():
            if state[1] == 'FREE':
                A[0, idx] = 1.0
            else:
                A[1, idx] = 1.0

        return A

    # Get transition matrix
    def get_B(self):
        B = np.zeros((self.n_states, self.n_states, self.n_actions))

        for state, idx in self.state_to_idx.items():
            for action_id, action_label in enumerate(self.actions):
                next_label = list(state)

                if action_label == 'TO_TOOL':
                    next_label[0] = 'TOOL'

                elif action_label == 'TO_BALL':
                    if state == ('TOOL', 'PICKED'):
                        next_label[0] = 'BALL'

                next_idx = self.state_to_idx[tuple(next_label)]
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
        H_A_ext = - (self.A_ext * utils.log_stable(self.A_ext)).sum(axis=0)
        H_A_tact = - (self.A_tact * utils.log_stable(self.A_tact)).sum(axis=0)

        return H_A_ext + H_A_tact

    # Infer current states
    def infer_states(self, r_t_ext, o_tact):
        # Get expected state from observations
        qs_t_ext = self.A_ext.T.dot(r_t_ext)
        qs_t_tact = self.A_tact.T.dot(o_tact)

        log_prior = utils.log_stable(self.prior) * c.k_d
        log_post_ext = utils.log_stable(qs_t_ext)
        log_post_tact = utils.log_stable(qs_t_tact)

        qs = utils.softmax(log_post_ext + log_post_tact + log_prior)

        return qs

    # Compute expected states
    def get_expected_states(self, qs_current, action):
        qs_u = self.B[:, :, action].dot(qs_current)

        return qs_u

    # Compute expected observations
    def get_expected_obs(self, A, qs_u):
        qo_u = A.dot(qs_u)

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
    def step(self, o_tact):
        # Perform BMC
        q_r_ext = utils.bmc(self.o_ext, np.sum(self.L_ext, axis=0), c.w_bmc,
                            c.gain_prior, c.gain_evidence)

        # Infer current state
        qs_current = self.infer_states(q_r_ext, o_tact)

        # Compute expected free energy
        G = self.compute_G(qs_current)

        # Marginalize P(u|pi)
        Q_pi = utils.softmax(self.E - G)

        # Compute action posterior
        self.P_u = self.compute_prob_actions(Q_pi)

        # Compute next observations
        self.prior = self.get_qs_next(self.P_u, qs_current)
        self.o_int[:] = [1.0]
        self.o_ext[:] = self.get_expected_obs(self.A_ext, self.prior)

        if c.debug:
            np.set_printoptions(precision=2, suppress=True)
            # print('actions:', self.P_u)
            print('L_ext:', self.L_ext * c.gain_evidence)
            print('qr_ext:', q_r_ext)
            # print('qs:', qs_current)
            # print('qs_n:', self.prior)
            # print('v_ext:', self.o_ext)
            input()

        # Clear evidences
        self.L_int[:] = 0
        self.L_ext[:] = 0
