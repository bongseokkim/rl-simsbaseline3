import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from env import env
import matplotlib.pyplot as plt
import datetime
import wandb
import os

# Hyperparameters





experiment_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
os.mkdir(f"logs/{experiment_id}")
# wandb.init(project='RL_PPO',name=f'experiment_{experiment_id}')
# os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Actornetwork(nn.Module):
    def __init__(self, num_inputs , hidden_size=32, num_outputs=1):
        super(Actornetwork, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, num_outputs)  # for mu
        self.fc_std = nn.Linear(hidden_size, num_outputs)  # for sigma
    def forward(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        mu = torch.tanh(self.fc_mu(x)+1)/2
        # mu = self.fc_mu(x)
        sd = F.softmax(self.fc_std(x))
        # sd = torch.clamp(sd,min=0.001, max=0.4)
        return mu, sd

class Criticnetwork(nn.Module):
    def __init__(self, num_inputs , hidden_size = 32, num_outputs = 1):
        super(Criticnetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fcv = nn.Linear(hidden_size, num_outputs) # for mu
    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.fcv(x)
        return value


class PPO(nn.Module):
    def __init__(self,epoches,input_dims, gamma, actor_lr, critic_lr, hidden_size, buffer_size):
        super(PPO, self).__init__()
        self.data = []
        self.gamma = gamma
        self.epoches = epoches
        self.optimization_step = 0
        self.buffer_size = buffer_size
        self.max_norm = 1 # 0.5
        self.eps_clip = 0.1
        self.gae_lmbda = 0.95 #0.9 ~ 1.0     0.95 로 바꾸기
        self.vf_coeff = 1 # 0.5, 1
        self.entropy_coeff = 0 # 0 ~ 0.01
        self.minibatch_size = 64  # 4 ~ 4096
        self.input_dims = input_dims
        ## define actor & critic network
        self.actor = Actornetwork(num_inputs=self.input_dims, hidden_size=hidden_size, num_outputs=1)
        self.critic = Criticnetwork(num_inputs=self.input_dims, hidden_size=hidden_size, num_outputs=1)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def append_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        # data = []
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            if done:
                mask=0
            else:
                mask=1
            done_lst.append([mask])

        s, a, r, s_prime, mask, old_log_prob= torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float),\
                                                   torch.tensor(r_lst,dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float),\
                                                   torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst,dtype=torch.float)
        self.data = []
        return s, a, r, s_prime, mask, old_log_prob

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        mu, sd = self.actor.forward(state)
        normal = Normal(mu, sd)
        action = normal.sample()
        log_prob = normal.log_prob(action)
        action = torch.clamp(min=0, max=1, input=action)

        return action.item(), log_prob.item()
    def cal_advantage(self, adv, delta):
        advantage = adv
        advantage_lst = []
        for delta_t in delta[::-1]:  # 뒤에서부터 거꾸로 calculation
            advantage = self.gamma * self.gae_lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        return advantage_lst

    def train_net(self):
        state, action, reward, s_prime, done_mask, old_log_prob = self.make_batch()

        for i in range(self.epoches):
            # for rollout_data in self.rollout_buffer.get(self.batch_size):

            with torch.no_grad():
                td_target = reward + self.gamma * self.critic.forward(s_prime) * done_mask
                delta = td_target - self.critic.forward(state)
            delta = delta.numpy()
            ## GAE advantage calculation \hat A_t = \delta_t + (r \lambda) \delta_{t+1} + ... + (r \lambda)^{T-t+1} \delta_{T-1}
            adv = 0.0
            advantage_lst = self.cal_advantage(adv, delta)
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            try:
                mu, sd = self.actor.forward(state, softmax_dim=1)
            except (AssertionError, ValueError) as e:
                mu, sd = self.actor.forward(state, softmax_dim=1)

            # mu, sd = self.actor.forward(state, softmax_dim=1)
            normal = Normal(mu, sd)
            log_prob = normal.log_prob(action)
            ratio = torch.exp(log_prob - old_log_prob)

            # self.critic_optimizer.zero_grad()
            # critic_loss = F.smooth_l1_loss(self.critic.forward(state), td_target)  # smooth_l1_loss
            #
            # critic_loss.backward()
            # nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_norm)
            # self.critic_optimizer.step()
            #
            # # Clipped Surrogate ObjectivePPO,
            #
            # self.actor_optimizer.zero_grad()
            # policy_adv = torch.min(ratio * advantage, torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage) # policy reward
            # policy_adv.backward()
            # self.actor_optimizer.step()
            # self.optimization_step += 1

            policy_adv = torch.min(ratio * advantage, torch.clamp(ratio, 1 - self.eps_clip,
                                                                  1 + self.eps_clip) * advantage)  # policy reward
            critic_loss = F.smooth_l1_loss(self.critic.forward(state), td_target)  # smooth_l1_loss

            loss = -policy_adv + self.vf_coeff * critic_loss

            ## CHANGE
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            self.optimization_step += 1


env = env()
def main(sta=None):
    gamma = 0.99
    # actor_lr = 0.0002;
    actor_lr = 0.0001 #0.00001
    critic_lr = 0.001
    epoches = 10 # Num of epoch when optimizing the surrogate loss
    buffer_size = 30
    T = 32 # 32


    model = PPO(epoches=epoches, input_dims=3, gamma=gamma, actor_lr=actor_lr, critic_lr=critic_lr, hidden_size=32,
                buffer_size = buffer_size);

    # model = PPO()
    score = 0.0
    print_interval = 1
    epi_reward = []
    epi_state = []
    epi_action = []



    for n_epi in range(10000):
        state_buffer = []
        reward_buffer = []
        action_buffer = []
        state = env.reset()
        done = False
        # for k in range(max_len * 4):
        #     state_representation.append((0, 0, 0))
        # state_representation.append(state)

        while not done:
            for t in range(T):
                # print(state_representation)
                # state_concate = np.concatenate(state_representation)
                action , log_prob = model.select_action(state)
                # action, log_prob = model.select_action(state)
                next_state, reward, done = env.step(action)
                # state_representation.append(next_state)
                # new_state_concate = np.concatenate(state_representation)

                transition = (state, action, reward, next_state, log_prob, done)
                # model.data.append(transition)
                model.append_data(transition)
                state = next_state
                score += reward
                state_buffer.append(state)
                action_buffer.append(action)
                reward_buffer.append(reward)
                if done:
                    epi_state.append(state_buffer)
                    epi_reward.append(np.sum(reward_buffer) / t)
                    epi_action.append(action_buffer)
                    break

            critic_loss = model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f},final state : {}, final action : {}, opt step: {}".format(n_epi, score / print_interval,state, action, model.optimization_step))
            # wandb.log({'mean_reward': score / print_interval, 'action': a})
            score = 0.0
 


if __name__ == '__main__':
    main()
