import torch
import numpy as np
import utils


# Define unit class
class Unit:
    def __init__(self, dim, inputs, v, L, pi_eta_x, p_x,
                 pi_x, lr, F_m, lmbda, g=None):
        # Point to inputs
        self.inputs = inputs
        self.v = v
        self.L = L

        # Initialize hidden states
        self.x = torch.zeros(dim)

        # Initialize precisions
        self.pi_eta_x = torch.tensor(pi_eta_x)
        self.p_x = torch.tensor(p_x)
        self.pi_x = torch.tensor(pi_x)

        # Initialize likelihood and dynamics
        self.g = g
        self.F_m = F_m
        self.Preds_x = torch.zeros((len(self.v), *dim[1:]))

        # Initialize learning rate and gain
        self.lr = torch.tensor(lr)
        self.lmbda = lmbda

        # Initialize prediction errors and gradients
        self.eps_eta_x = torch.zeros_like(self.x)
        self.eps_x = torch.zeros_like(self.x[1])

        self.grad_o = []
        self.grad_x = torch.zeros_like(self.x[0])

    # Perform likelihood step
    def step_likelihood(self):
        if self.g is None:
            self.eps_eta_x[0] = (self.x[0] - torch.tensor(self.inputs)) \
                * self.pi_eta_x
        else:
            # Get priors from parent units
            Eta_x = [inpt.x[0].clone().detach().requires_grad_(True)
                     for inpt in self.inputs]

            # Predict hidden states
            pred_eta_x = self.g(*Eta_x)

            # Compute prior prediction error
            x0 = self.x[0].clone().detach()
            self.eps_eta_x[0] = (x0 - pred_eta_x) * self.pi_eta_x

            # Backpropagate gradient to parent units
            self.eps_eta_x.backward(self.eps_eta_x)
            for eta_x, inpt in zip(Eta_x, self.inputs):
                inpt.grad_o.append(eta_x.grad)

    # Perform dynamics step
    def step_dynamics(self):
        # Get parents
        x = self.x[0].clone().detach().requires_grad_(True)

        # Predict dynamics
        self.Preds_x = torch.stack([f(x, self.lmbda) for f in self.F_m])
        v = torch.from_numpy(self.v).float()
        pred_x = torch.tensordot(v, self.Preds_x, dims=([0], [0]))

        # Compute dynamics prediction error
        x1 = self.x[1].clone().detach()
        self.eps_x = (x1 - pred_x) * self.pi_x

        # Backpropagate gradients to parents
        self.eps_x.backward(self.eps_x)
        self.grad_x = x.grad

        # Accumulate log evidence
        self.L += utils.acc_log_evidence(pred_x, self.Preds_x, x1, self.pi_x,
                                         self.pi_x, self.p_x)

    # Perform a step
    def step(self):
        self.step_likelihood()
        self.step_dynamics()

    # Integrate with gradient descent
    def update(self, dt):
        # Compute free energy derivative
        dF_dx = torch.zeros_like(self.x)
        dF_dx += self.eps_eta_x
        for grad_o in self.grad_o:
            dF_dx += grad_o
        dF_dx[0] += self.grad_x
        dF_dx[1] += self.eps_x

        # Compute belief derivative
        x_dot = utils.shift(self.x) - dF_dx.clone().detach()

        # Update belief
        self.x += dt * x_dot * self.lr

        # Clear variables
        self.eps_eta_x = torch.zeros_like(self.x)
        self.grad_o = []


# Define observation class
class Obs:
    def __init__(self, dim, inputs, pi_o, g, lr=None):
        # Point to inputs
        self.inputs = inputs

        # Initialize observation and actions
        self.o = torch.zeros(dim)
        self.actions = np.zeros_like(self.o) if lr else None

        # Initialize precision
        self.pi_o = torch.tensor(pi_o)

        # Initialize likelihood
        self.g = g

        # Initialize learning rate
        self.lr = lr

        # Initialize prediction error
        self.eps_o = torch.zeros_like(self.o)

    # Compute prediction
    def predict(self, get_x=False):
        # Get priors from parent units
        X = [inpt.x.clone().detach().requires_grad_(get_x)
             for inpt in self.inputs]

        # Compute prediction
        if not get_x:
            return self.g(*X)
        return X, self.g(*X)

    # Perform a step
    def step(self):
        # Predict hidden state
        X, p_o = self.predict(get_x=True)

        # Compute prior prediction error
        self.eps_o = (self.o - p_o) * self.pi_o

        # Backpropagate gradient to parent units
        self.eps_o.backward(self.eps_o)
        for eta_o, inpt in zip(X, self.inputs):
            inpt.grad_o.append(eta_o.grad)

    # Integrate with gradient descent
    def update(self, dt):
        if self.actions is not None:
            # Compute free energy derivative
            a_dot = -dt * self.eps_o.detach().numpy()

            # Update actions
            self.actions += dt * a_dot * self.lr
