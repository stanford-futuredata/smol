import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import custom_resnets


class NormImage(nn.Module):
    def __init__(
            self,
            model,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
    ):
        super().__init__()
        self.model = model
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        print(x.size())
        x[:, 0, ...] -= self.mean[0]
        x[:, 1, ...] -= self.mean[1]
        x[:, 2, ...] -= self.mean[2]
        x[:, 0, ...] /= self.std[0]
        x[:, 1, ...] /= self.std[1]
        x[:, 2, ...] /= self.std[2]
        print(x.size())
        return self.model(x)


class ResizeImage(nn.Module):
    def __init__(self, model, dim=224):
        super().__init__()
        self.model = model
        self.dim = dim

    def forward(self, x):
        print(x.size())
        x = F.interpolate(x, (self.dim, self.dim), mode='bilinear')
        print(x.size())
        return self.model(x)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', required=True)
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--do_norm', type=str2bool, required=True)
    parser.add_argument('--do_resize', type=str2bool, required=True)
    args = parser.parse_args()

    depth_to_net = {
            18: custom_resnets.resnet18,
            34: custom_resnets.resnet34,
            50: custom_resnets.resnet50,
            101: custom_resnets.resnet101,
            152: custom_resnets.resnet152}

    dummy_input = torch.randn(args.batch_size, 3, 224, 224).cuda()
    model = depth_to_net[args.depth](pretrained=False, bs=args.batch_size)
    sd = torch.load(args.in_path)['state_dict']
    nb_out, nb_in = sd['fc.weight'].shape
    model.fc = nn.Linear(nb_in, nb_out)
    model.load_state_dict(sd)

    if args.do_resize:
        print('Doing resize')
        model = ResizeImage(model)
    if args.do_norm:
        print('Doing norm')
        model = NormImage(model)

    model.cuda()
    model.eval()

    out_fname = args.out_path
    torch.onnx.export(
            model,
            dummy_input,
            out_fname,
            verbose=True
    )

if __name__ == '__main__':
    main()
