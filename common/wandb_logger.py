import wandb 
from argparser import base_prser
import datetime
import os 
import numpy as np

class base_logger(object):
    def __init__(self,project_name, arg_parser):
        # make project folder to save asset
        self.experiment_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.project_name = project_name
        self.directory = './'+self.project_name+'_'+self.experiment_id
        self.arg_parser = arg_parser

        try : 
            if not os.path.exists(self.directory) :
                os.makedirs(self.directory)

        except OSError : 
                print ('Error: Creating directory. ' +  self.directory)
    
    def wandb_start(self):
        wandb.init(
            project= self.project_name,
            name = f'experiment_{self.directory}',
        )
        wandb.config.update(self.arg_parser.args)
    
    def wandb_end(self):
        wandb.finish()
    
    def wandb_log(self, reward_buffer,add_info:dict =None):
        wandb.log({'mean_reward': np.mean(reward_buffer)})

        if len(add_info) > 0 : 
            for name,value in add_info.items() :
                wandb.log({name: value})
            
