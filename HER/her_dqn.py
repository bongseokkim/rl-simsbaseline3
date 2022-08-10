import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from her import Hindsight_replay_bufer
import torch.nn.functional as F

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.iter_cntr = 0
        self.replace_target = 100
        self.replay_buffer= Hindsight_replay_bufer(self.mem_size,input_dims,n_actions=n_actions)
        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions,
                                   input_dims=2*input_dims[0],
                                   fc1_dims=256, fc2_dims=256)
        print(self.replay_buffer.action_memory)

    def choose_action(self, observation, goal):
        if np.random.random() > self.epsilon:
            concat_state_goal = np.concatenate([observation, goal])
            concat_state_goal = T.tensor([concat_state_goal],dtype=T.float).to(self.Q_eval.device)

            actions = self.Q_eval.forward(concat_state_goal)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.replay_buffer.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.replay_buffer.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.replay_buffer.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(
                self.replay_buffer.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.replay_buffer.action_memory[batch]
        reward_batch = T.tensor(
                self.replay_buffer.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(
                self.replay_buffer.terminal_memory[batch]).to(self.Q_eval.device)
        goal_batch = T.tensor(self.replay_buffer.goal_memory[batch]).to(self.Q_eval.device)
        
        concat_state_goal =T.tensor(np.concatenate([state_batch, goal_batch], 1), dtype = T.float)
        concat_next_state_goal =T.tensor(np.concatenate([new_state_batch, goal_batch], 1), dtype = T.float)
   
        q_eval = self.Q_eval.forward(concat_state_goal)[batch_index, action_batch]
        q_next = self.Q_eval.forward(concat_next_state_goal)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]

        loss = F.smooth_l1_loss(q_eval, q_target)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min