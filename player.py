import jittor as jt
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # For Running
    parser.add_argument("--data", default = "", help = "Path to the scene")
    parser.add_argument("--load_path", default= "", help = "Path to snapshot which need be load")
    parser.add_argument("--save_path", default= "", help = "Path where we saved snapshot")
    parser.add_argument("--test", action= "store_true", help = "If test, we load data from data/transform_test.json")
    
    # For Training
    parser.add_argument("--n_steps", type = int, default= 10000, help = "Training steps")
    # Noticed that the truly length is sqrt(3)/steps. Noticed that it may be changed for this parameter
    parser.add_argument("--ray_step", type = int, default= 1024, help = "Step length of ray marching")
    parser.add_argument("--filter", action = "store_true", help = "If filter, load filter config from ./configs")
    
    args = Arguements(parser.parse_args())
    return args

class Arguements:
    def __init__(self, args:argparse.ArgumentParser().parse_args()):
        # For Running
        self.data = args.data
        self.lpath = args.load_path
        self.spath = args.save_path
        self.test:bool = args.test
        
        # Training
        self.n_steps = args.n_steps
        self.ray_step = args.ray_step
        self.filter = args.filter
    
if __name__ == "__main__":
    agrs = parse_args()
    
    print("Run JRNeRF!")