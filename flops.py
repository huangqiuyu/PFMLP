import torch, math
from thop import profile, clever_format

from pfmlp import PFMLP

if __name__=="__main__":
    custom_ops = {}
    input = torch.randn(1, 3, 224, 224)

    #model = PFMLP(dims=[48,96,192,384], layers=[3,6,11,3], expand_ratio=3.0, mlp_ratio=3.0, use_dw=True, drop_path_rate=0.05)
    #model = PFMLP(dims=[64,128,256,512], layers=[3,6,11,3], expand_ratio=3.0, mlp_ratio=3.0, use_dw=False, drop_path_rate=0.15)
    #model = PFMLP(dims=[80,160,320,640], layers=[3,6,11,3], expand_ratio=3.0, mlp_ratio=3.0, use_dw=False, drop_path_rate=0.20)
    model = PFMLP(dims=[96,192,384,768], layers=[3,6,11,3], expand_ratio=3.0, mlp_ratio=3.0, use_dw=False, drop_path_rate=0.25)

    model.eval()
    
    macs, params = profile(model, inputs=(input, ), custom_ops=custom_ops)
    macs, params = clever_format([macs, params], "%.3f")
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print('Flops:  ', macs)
    print('Params: ', params)


