# type: ignore
# adapted from https://github.com/nadermx/backgroundremover/
import os
import torch
from enfugue.diffusion.support.background.u2net import u2net

class Net(torch.nn.Module):
    def __init__(self, model_name, path):
        super(Net, self).__init__()
        assert os.path.exists(path), "model path must exist"
        model = {
            'u2netp': (u2net.U2NETP,
                       'e4f636406ca4e2af789941e7f139ee2e',
                       '1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy',
                       'U2NET_PATH'),
            'u2net': (u2net.U2NET,
                      '09fb4e49b7f785c9f855baf94916840a',
                      '1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ',
                      'U2NET_PATH'),
            'u2net_human_seg': (u2net.U2NET,
                                '347c3d51b01528e5c6c071e3cff1cb55',
                                '1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P',
                                'U2NET_PATH')
        }[model_name]

        if model_name == "u2netp":
            net = u2net.U2NETP(3, 1)
        elif model_name == "u2net":
            net = u2net.U2NET(3, 1)
        elif model_name == "u2net_human_seg":
            net = u2net.U2NET(3, 1)
        else:
            print("Choose between u2net, u2net_human_seg or u2netp", file=sys.stderr)

        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.to(dtype=torch.float32, non_blocking=True)
        net.eval()
        self.net = net
        self.forward = self.net.forward

    def to(self, device: torch.device):
        self.net.to(device=device)
