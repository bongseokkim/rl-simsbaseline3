import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('common'))))
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F 
from torch.distributions import Categorical 

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
            weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class Actor_Critic(nn.Module):
    def __init__(self, num_input, n_action, gamma):
        super(Actor_Critic,self).__init__()
        self.gamma = gamma 
        self.rewards = []
        self.actions = []
        self.states = []

        self.pi_1 = nn.Linear(num_input, 128)
        self.pi =  nn.Linear(128, n_action)

        self.v_1 = nn.Linear(num_input, 128)
        self.v =  nn.Linear(128, 1)


    def forward(self, state):
        pi_1 = F.relu(self.pi_1(state))
        v_1 = F.relu(self.v_1(state))

        pi = self.pi(pi_1)
        v =  self.v(v_1)

        return pi, v
    
    def choose_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float)
        pi, v = self.forward(state)
        probs = torch.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]

        return action 
    
    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def clear_mem(self):
        self.rewards = []
        self.actions = []
        self.states = []
        
    def calcualte_return(self, done):
        state = torch.tensor(np.array(self.states), dtype = torch.float)
        _, v = self.forward(state)

        R = v[-1] * (1-int(done)) # 0 for terminal state, else V 

        batch_return = []  
        for reward in self.rewards[::-1]:# do backward
            R = reward + self.gamma * R # r + gamma*R 
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype= torch.float)

        return batch_return

    def calculate_loss(self, done):
        states = torch.tensor(np.array(self.states), dtype= torch.float)
        actions = torch.tensor(self.actions, dtype= torch.float)
        returns = self.calcualte_return(done)

        pi, values = self.forward(states)
        
        # for critic loss
        values = values.squeeze()
        critic_loss = (returns-values)**2

        # for actor loss 
        probs = torch.softmax(pi,dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss+actor_loss).mean()
        
        return total_loss

    

class A3C_worker(mp.Process):
    def __init__(self, num_input, n_action, lr, gamma,
                 global_agent, optimizer, global_ep_idx,name,
                 chkpt_dir=None, env=None, saving_name='A3C'):
        super(A3C_worker,self).__init__()
        self.num_input = num_input
        self.n_action = n_action
        self.lr = lr 
        self.gamma = gamma 
        
        self.global_agent = global_agent
        self.local_agent = Actor_Critic(num_input, n_action, gamma)
        self.optimizer = optimizer
        self.episode_idx = global_ep_idx

        self.name = 'w%02i' % name
        self.env = env 
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, saving_name)
        

    
    def run(self):

        thered_step = 1 
        while self.episode_idx.value < 5000 :
            done = False 
            state = self.env.reset()
            score = 0 
            self.local_agent.clear_mem()

            while not done :
                action = self.local_agent.choose_action(state) # pefrom a according to policy pi
                state_, reward, done, info = self.env.step(action) # receive reward r_t, and new sate s_{t+1}
                score += reward 
                self.local_agent.remember(state, action, reward/100)

                if thered_step % 5 == 0 or done :
                    loss = self.local_agent.calculate_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()

                    for local_param, global_param in zip(self.local_agent.parameters(),self.global_agent.parameters()):
                        global_param._grad = local_param.grad 
                    self.optimizer.step()

                    self.local_agent.load_state_dict(self.global_agent.state_dict())
                    self.local_agent.clear_mem()
                thered_step += 1
                state = state_ 
            
            with self.episode_idx.get_lock():
                self.episode_idx.value +=1 
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)

if __name__ == '__main__':
    print(f'start, cpu count :{mp.cpu_count()}')
    lr = 0.0002
    env = gym.make('CartPole-v1')
    n_actions = 2 
    num_input = 4
    gamma = 0.98

    global_agent = Actor_Critic(num_input, n_actions, gamma)
    global_agent.share_memory()
    optimizer = SharedAdam(global_agent.parameters(), lr=lr, betas=(0.92, 0.999))
    
    global_ep = mp.Value('i', 0)

    workers =[A3C_worker(global_agent= global_agent,num_input=num_input,name=i, env= env, optimizer= optimizer,
                         n_action=n_actions, gamma=gamma, lr=lr, global_ep_idx=global_ep, chkpt_dir='model')
                         for i in range(mp.cpu_count())]
    [worker.start() for worker in workers]
    [worker.join() for worker in workers]










