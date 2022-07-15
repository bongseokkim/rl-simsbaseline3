import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('common'))))
import numpy as np 
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from common.mlp import MLP 



class Reinforce(nn.Module):
    def __init__(self,input_dim, output_dim, num_neurons,lr,gamma, chkpt_dir,env,saving_name='reinforce'):
        super(Reinforce, self).__init__()

        self.actor = MLP(input_dim, output_dim, num_neurons,
                 hidden_activation ='ReLU', out_activation='Identity')
        self.optimizer = optim.Adam(self.actor.parameters())
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.lr =lr 
        self.gamma = gamma
        self.action_memory = []
        self.reward_memory = []
        self.saving_name = saving_name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, saving_name)
        self.env = env 

    def choose_action(self,obs):
        obs = torch.tensor(obs)
        probs = F.softmax(self.actor.forward(obs))
        action_probs = Categorical(probs)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def train(self):
        self.optimizer.zero_grad()
        return_G = np.zeros_like(self.reward_memory, dtype=np.float64)

        # calculate return, G 
        for t in range(len(self.reward_memory)):
            G_sum =0 
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k]*discount
                discount *=self.gamma
            return_G[t] = G_sum
        mean = np.mean(return_G)
        std = np.std(return_G) if np.std(return_G) > 0 else 1 
        return_G = (return_G-mean)/std 

        return_G = torch.tensor(return_G, dtype=torch.float).to(self.device)

        loss = 0 
        for g,log_prob in zip(return_G, self.action_memory):
            loss += -g*log_prob # gradient acesnt -log^pi *G 
        loss.backward()
        self.optimizer.step()

        self.action_memory = [] 
        self.reward_memory = [] 
    
    def learn(self, num_episodes=10000):
            score_history = [] 
            score = 0 
            for i in range(num_episodes):
                print('episode: ', i,'score: ', score)
                done = False
                score = 0
                observation = self.env.reset()
                while not done:
                    action = self.choose_action(observation)
                    observation_, reward, done, info = env.step(action)
                    self.store_rewards(reward)
                    observation = observation_
                    score += reward
                score_history.append(score)
                self.train()

    def save_checkpoint(self):
        print('\033[31m'+'... saving agent brain checkpoint ...'+'\033[0m')
        torch.save(self.actor.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        print('\033[31m'+'... load agent brain checkpoint ...'+'\033[0m')
        self.actor.load_state_dict(torch.load(self.checkpoint_file))


if __name__=='__main__':
    env = gym.make('CartPole-v1')
    agent = Reinforce(lr = 0.001, input_dim=4, gamma=0.99, output_dim=2,env=env,
                    num_neurons=[128,128],chkpt_dir='model',saving_name='test')
    score_history = [] 
    score = 0 
    num_episodes = 2500
    best = 0 

    for i in range(num_episodes):
        print('episode: ', i,'score: ', score)
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_rewards(reward)
            observation = observation_
            score += reward
            if score >=400 :
                env.render()
        score_history.append(score)
        agent.train()    
        if score >= best : 
            agent.save_checkpoint()
            best = score 
        # agent.save_checkpoint()
        # agent.load_checkpoint()
    env.close()
    #agent.learn()
    