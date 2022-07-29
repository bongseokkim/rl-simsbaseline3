import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.distributions import Normal
import numpy as np
from env import env
import matplotlib.pyplot as plt
import datetime

# Hyperparameters

gae_lmbda = 0.95
eps_clip = 0.2

rollout_len = 3
buffer_size = 30000
minibatch_size = 32

experiment_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

wandb.init(project='RL_PPO',name=f'experiment_{experiment_id}')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Actornetwork(nn.Module):
    def __init__(self, num_inputs , hidden_size=128, num_outputs=1):
        super(Actornetwork, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, num_outputs)  # for mu
        self.fc_std = nn.Linear(hidden_size, num_outputs)  # for sigma
    def forward(self,x, softmax_dim=0 ):
        x = F.relu(self.fc1(x))
        mu = (torch.tanh(self.fc_mu(x))+1)/2 ## env 에 맞춰 수정
        sd = F.softmax(self.fc_std(x))
        return mu, sd

class Criticnetwork(nn.Module):
    def __init__(self, num_inputs , hidden_size = 128, num_outputs = 1):
        super(Criticnetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fcv = nn.Linear(hidden_size, num_outputs) # for mu
    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.fcv(x)
        return value


class PPO(nn.Module):
    def __init__(self,epoches,input_dims, gamma, actor_lr, critic_lr, hidden_size):
        super(PPO, self).__init__()
        self.data = []
        self.gamma = gamma
        self.epoches = epoches
        self.optimization_step = 0


        ## define actor & critic network
        self.actor = Actornetwork(num_inputs=input_dims, hidden_size=hidden_size, num_outputs=1)
        self.critic = Criticnetwork(num_inputs=input_dims, hidden_size=hidden_size, num_outputs=1)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(buffer_size):
            for i in range(minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition

                    s_lst.append(s)
                    a_lst.append([a])
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)

            mini_batch = torch.tensor(s_batch, dtype=torch.float), torch.tensor(a_batch, dtype=torch.float), \
                         torch.tensor(r_batch, dtype=torch.float), torch.tensor(s_prime_batch, dtype=torch.float), \
                         torch.tensor(done_batch, dtype=torch.float), torch.tensor(prob_a_batch, dtype=torch.float)
            data.append(mini_batch)

        return data
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        mu, sd = self.actor.forward(state)
        normal = Normal(mu, sd)
        action = normal.sample()
        log_prob = normal.log_prob(action)
        action = torch.clamp(min=0, max=1, input=action)

        return action.item(), log_prob.item()


    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + self.gamma * self.critic.forward(s_prime) * done_mask
                delta = td_target - self.critic.forward(s)
            delta = delta.numpy()
            ## GAE advantage calculation \hat A_t = \delta_t + (r \lambda) \delta_{t+1} + ... + (r \lambda)^{T-t+1} \delta_{T-1}
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]: #뒤에서부터 거꾸로 calculation
                advantage = self.gamma * gae_lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))
        return data_with_adv

    def train_net(self):
        if len(self.data) == minibatch_size * buffer_size:
            data = self.make_batch()

            data = self.calc_advantage(data)

            for i in range(self.epoches):
                for mini_batch in data:
                    state, action, reward, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch
                    try:
                        mu, sd = self.actor.forward(state, softmax_dim=1)
                    except (AssertionError, ValueError) as e:
                        mu, sd = self.actor.forward(state, softmax_dim=1)

                    # mu, sd = self.actor.forward(state, softmax_dim=1)
                    normal = Normal(mu, sd)
                    log_prob = normal.log_prob(action)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
                    # Clipped Surrogate Objective
                    # loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.critic.forward(state), td_target)

                    # actor loss + critic loss
                    actor_loss = -torch.min(surr1, surr2)
                    critic_loss = F.smooth_l1_loss(self.critic.forward(state), td_target)
                    # smooth_l1_loss 는 nn.MSELoss 보다 이상치에 덜 민감해 기울기 폭발을 방지

                    ### CHANGE

                    self.critic_optimizer.zero_grad()
                    critic_loss.mean().backward()
                    self.critic_optimizer.step()
                    self.actor_optimizer.zero_grad()
                    actor_loss.mean().backward()
                    self.actor_optimizer.step()
                    self.optimization_step += 1
                  
def main(sta=None):
    gamma = 0.98
    actor_lr = 0.00003;
    critic_lr = 0.0001;
    epoches = 10
    # actor_lr = 0.0002; critic_lr = 0.002;z
    model = PPO(epoches=epoches, input_dims=3, gamma=gamma, actor_lr=actor_lr, critic_lr=critic_lr, hidden_size=128);
    # model = PPO()
    score = 0.0
    print_interval = 1
    rollout = []
    epi_state = []
    epi_reward = []
    epi_action = []



    for n_epi in range(10000):
        state = env.reset()
        done = False
        for i in range(1002):
            for t in range(rollout_len):
                state_buffer = []
                reward_buffer = []
                action_buffer = []

                action, log_prob = model.select_action(state)

                s_prime, reward, done = env.step(action)

                rollout.append((state, action, reward, s_prime, log_prob, done))

                if len(rollout) == rollout_len:
                    model.put_data(rollout)
                    rollout = []
                state_buffer.append(state)
                action_buffer.append(action)
                reward_buffer.append(reward)
                state = s_prime
                score += reward
                if done:
                    epi_state.append(state_buffer)
                    epi_action.append(action_buffer)
                    break

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg reward : {:.1f}, opt step: {},  final state : {}, final action : {}".format(n_epi, score / print_interval,
                                                                    model.optimization_step,state,action))
            epi_reward.append(score / print_interval)
            wandb.log({'mean_reward': score / print_interval, 'action': action})
            score = 0.0


if __name__ == '__main__':
    main()
