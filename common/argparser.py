import argparse 

class base_prser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--lr", default=0.0001,type=float)
        self.parser.add_argument("--episode_train", default=3000,type=int)
        self.parser.add_argument("--episode_test", default=100,type=int)
        self.args = self.parser.parse_args()

    # add additional argument to argparser
    def add_argument(self,argument : dict):
        for name,default_value in argument.items() :
            self.parser.add_argument("--"+name, default=default_value)
        self.args = self.parser.parse_args()