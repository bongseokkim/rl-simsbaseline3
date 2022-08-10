from sre_constants import SUCCESS
from her_dqn import Agent
from env import Env
import numpy as np 

def main():
    checkpoint_dir ="./checkpoint"
    n_bits = 5
    env = Env(n_bits = n_bits)
    agent = Agent(lr =0.0001, input_dims=[n_bits], batch_size=64,gamma=0.99,epsilon=0.1, n_actions=n_bits)
    episode = 50000
    sucess = 0
    for epi in range(episode):
        # sample goal g and initial state s0
        state, goal = env.reset() 
        done = False 
        transition = [] 

        for t in range(n_bits):
            # sample an action a_t using behavior policy
            action = agent.choose_action(state, goal)
            # Execute the action a_t and observe a new state s_{t+1}
            next_state, reward, done = env.step(action)
            transition.append((state, action, reward,next_state, done, goal))
            
            state = next_state
            
            agent.learn()
            if done :
                sucess+=1
                
    
        for i in range(len(transition)):
             # store the transtion in replay buffer (standard experience replay)
            agent.replay_buffer.store_transition(*transition[i])

            # HER
            if not done : 
                state, action, next_state, reward, done, goal = transition[i]
                new_goal = transition[-1][-1] # terminal state become new goal
                if np.array_equal(next_state, new_goal) : 
                    agent.replay_buffer.store_transition(state, action, next_state, 0, True, new_goal)

    
                agent.replay_buffer.store_transition(state, action, next_state, -1,False, new_goal )
        agent.learn()
        if epi %50 ==0 :
            print(f'episode :{epi}, success rate :{sucess/50 :.3f}')
            sucess = 0







if __name__ =='__main__':
    main()