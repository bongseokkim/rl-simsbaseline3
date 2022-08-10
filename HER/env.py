import numpy as np

class Env(object):
    def __init__(self, n_bits):
        self.n_bits = n_bits 
        self.state = np.random.randint(2, size = self.n_bits) # ex) array([1, 0, 1, 1, 1])
        self.goal = np.random.randint(2, size= self.n_bits)

    def reset(self):
        self.state = np.random.randint(2, size = self.n_bits) # ex) array([1, 0, 1, 1, 1])
        self.goal = np.random.randint(2, size= self.n_bits)
        return self.state, self.goal 

    def step(self, action):
        if self.state[action] == 1 :
            self.state[action] = 0 
        else : 
            self.state[action] = 1
        
        done = False 
        if np.array_equal(self.state, self.goal):
            done = True 
            reward = 0
        
        else : 
            reward = -1 

        return self.state, reward, done 

    
