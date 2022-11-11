import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        print(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.double = True
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.input_dims = input_dims

        # Initialize the target and policy networks
        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        self.Q_target = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        # Allocate memory for the replay buffer
        self.state_memory = np.zeros((self.mem_size, *self.input_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, *self.input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype=np.bool)
    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation, test=False):
        # check if the model is being trained or tested
        if test:
            state = T.tensor([observation], dtype=T.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
            return action
        # If the model is training act according to an epsilon greedy policy
        if np.random.random() > self.epsilon:
            observation = np.expand_dims(observation, axis=0)
            state = T.tensor(observation, dtype=T.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        max_mem = min(self.mem_cntr, self.mem_size)
        # get a batch of transitions from the memory
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        action_batch = T.tensor(self.action_memory[batch], dtype=T.long).to(self.Q_eval.device)
        next_state_batch = T.tensor(self.next_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        done_batch = T.tensor(self.done_memory[batch]).to(self.Q_eval.device)
        # Compute the estimated Q values from the policy network
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]

        # Compute the target Q values for the next state
        if self.double:
            # Select the next actions with the policy network Q_eval
            q_next_policy = self.Q_eval.forward(next_state_batch)
            next_actions = q_next_policy.argmax(dim=1)

            # get the Q values of the next actions using the target network
            q_next_target = self.Q_target.forward(next_state_batch)[batch_index, next_actions]
            q_target = reward_batch + self.gamma * q_next_target

        else:
            q_next = self.Q_target.forward(next_state_batch)
            q_next[done_batch] = 0.0
            q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        # calculate the loss function
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        self.Q_eval.optimizer.zero_grad()
        # Backpropagate the loss through the network
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
