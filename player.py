'''
Author: HypoxanthineOvO heyx1@shanghaitech.edu.cn
Date: 2023-04-07 00:58:16
LastEditors: HypoxanthineOvO heyx1@shanghaitech.edu.cn
LastEditTime: 2023-04-07 17:28:58
FilePath: /JRNeRF/player.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import jittor as jt
import os
import argparse

# Load Packages
from loader import SyntheticLoader

class Arguements:
    def __init__(self, args:argparse.ArgumentParser().parse_args()):
        # For Running
        self.data = args.data
        self.datatype = args.datatype
        self.lpath = args.load_path
        self.spath = args.save_path
        self.test:bool = args.test
        
        # Training
        self.n_steps = args.n_steps
        self.ray_step = args.ray_step
        self.filter = args.filter
    
def parse_args() -> Arguements:
    parser = argparse.ArgumentParser()
    # For Running
    parser.add_argument("--data", default = "", help = "Path to the scene")
    parser.add_argument("--datatype", default= "synthetic", help = "Choose in [synthetic, llff, vsp]")
    parser.add_argument("--load_path", default= "", help = "Path to snapshot which need be load")
    parser.add_argument("--save_path", default= "", help = "Path where we saved snapshot")
    parser.add_argument("--test", action= "store_true", help = "If test, we load data from data/transform_test.json")
    
    # For Training
    parser.add_argument("--n_steps", type = int, default= 10000, help = "Training steps")
    # HACK : Noticed that the truly length is sqrt(3)/steps. Noticed that it may be changed for this parameter
    parser.add_argument("--ray_step", type = int, default= 1024, help = "Step length of ray marching")
    parser.add_argument("--filter", action = "store_true", help = "If filter, load filter config from ./configs")
    
    args = Arguements(parser.parse_args())
    return args


if __name__ == "__main__":
    args = parse_args()
    
    if(args.datatype == "synthetic"):
        loader = SyntheticLoader(args.path)
    imgs, poses, [H, W, focal_length], test_indexs = loader.get_data()
    
    # TODO : Following are processing
    print("Run JRNeRF!")