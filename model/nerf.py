import jittor as jt
from jittor import nn

class NeRF(jt.Module):
    def __init__(self, D = 8, W = 256):
        super(NeRF, self).__init__()

        self.D = D
        self.W = W
        # Position Channels and Direction Channels  
        self.pos_ch = 3
        self.dir_ch = 3
        self.skips = [4]
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.pos_ch, W)] + 
            [nn.Linear(W, W) if i not in self.skips 
             else nn.Linear(W + self.pos_ch, W) for i in range(D-1)]
        )
        self.dir_linears = nn.ModuleList([nn.Linear(self.dir_ch + W, W//2)])
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)
    
    def execute(self, x):
        input_pts, input_dir = jt.split(x, [self.pos_ch, self.dir_ch], dim = -1)
        h = input_pts
        for i,l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = nn.relu(h)
            if i in self.skips:
                h = jt.concat([x,h], -1)
        
        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = jt.concat([feature, input_dir], -1)

        for i, l in enumerate(self.dir_linears):
            h = self.dir_linears[i](h)
            h = nn.relu(h)
        rgb = self.rgb_linear(h)
        outputs = jt.concat([rgb, alpha], -1)
        return outputs