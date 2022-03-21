import torch
import torch.nn as nn
import torch.nn.functional as F

from models import resnext

def init_weights(m):
  print(m)
  if type(m) == nn.Linear:
    print(m.weight)
  else:
    print('error')

class MMTM(nn.Module):
  def __init__(self, dim_rgb, dim_depth, ratio):
    super(MMTM, self).__init__()
    dim = dim_rgb + dim_depth
    dim_out = int(2*dim/ratio)
    self.fc_squeeze = nn.Linear(dim, dim_out)

    self.fc_rgb = nn.Linear(dim_out, dim_rgb)
    self.fc_depth = nn.Linear(dim_out, dim_depth)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

    # initialize
    with torch.no_grad():
      self.fc_squeeze.apply(init_weights)
      self.fc_rgb.apply(init_weights)
      self.fc_depth.apply(init_weights)

  def forward(self, rgb, depth):
    squeeze_array = []
    for tensor in [rgb, depth]:
      tview = tensor.view(tensor.shape[:2] + (-1,))
      squeeze_array.append(torch.mean(tview, dim=-1))
    squeeze = torch.cat(squeeze_array, 1)

    excitation = self.fc_squeeze(squeeze)
    excitation = self.relu(excitation)

    vis_out = self.fc_rgb(excitation)
    sk_out = self.fc_depth(excitation)

    vis_out = self.sigmoid(vis_out)
    sk_out = self.sigmoid(sk_out)

    dim_diff = len(rgb.shape) - len(vis_out.shape)
    vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

    dim_diff = len(depth.shape) - len(sk_out.shape)
    sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

    return rgb * vis_out, depth * sk_out


class MMTNet(nn.Module):
  def __init__(self, args):
    super(MMTNet, self).__init__()
    self.rgb = None
    self.depth = None
    self.final_pred = None

    self.mmtm1 = MMTM(256, 256, 4)
    self.mmtm2 = MMTM(512, 512, 4)
    self.mmtm3 = MMTM(1024, 1024, 4)
    self.mmtm4 = MMTM(2048, 2048, 4)

    self.return_interm_feas = False
    self.return_both = False
    if hasattr(args, 'fc_final_preds') and args.fc_final_preds:
      self.final_pred = nn.Linear(args.num_classes*2, args.num_classes)

  def get_mmtm_params(self):
    parameters = [
                {'params': self.mmtm1.parameters()},
                {'params': self.mmtm2.parameters()},
                {'params': self.mmtm3.parameters()},
                {'params': self.mmtm4.parameters()}
                         ]
    return parameters

  def get_rgb_params(self):
    parameters = [
                {'params': self.rgb.parameters()},
                {'params': self.mmtm1.parameters()},
                {'params': self.mmtm2.parameters()},
                {'params': self.mmtm3.parameters()},
                {'params': self.mmtm4.parameters()}
                         ]
    return parameters

  def get_depth_params(self):
    parameters = [
                {'params': self.depth.parameters()},
                {'params': self.mmtm1.parameters()},
                {'params': self.mmtm2.parameters()},
                {'params': self.mmtm3.parameters()},
                {'params': self.mmtm4.parameters()}
                         ]
    return parameters

  def set_rgb_depth_nets(self, rgb, depth, return_interm_feas=False):
    self.rgb = rgb
    self.depth = depth
    self.return_interm_feas = return_interm_feas

  def set_return_both(self, p):
    self.return_both = p

  def forward(self, x):
    rgb_x = x[:, :-1, :, :, :]
    depth_x = x[:, -1, :, :, :].unsqueeze(1)

    # rgb INIT BLOCK
    rgb_x = self.rgb.conv1(rgb_x)
    rgb_x = self.rgb.bn1(rgb_x)
    rgb_x = self.rgb.relu(rgb_x)
    rgb_x = self.rgb.maxpool(rgb_x)

    # depth INIT BLOCK
    depth_x = self.depth.conv1(depth_x)
    depth_x = self.depth.bn1(depth_x)
    depth_x = self.depth.relu(depth_x)
    depth_x = self.depth.maxpool(depth_x)

    # MMTM
    rgb_features, depth_features = [], []

    rgb_x = self.rgb.layer1(rgb_x)
    depth_x = self.depth.layer1(depth_x)
    rgb_x, depth_x = self.mmtm1(rgb_x, depth_x)
    rgb_features.append(rgb_x)
    depth_features.append(depth_x)
    
    rgb_x = self.rgb.layer2(rgb_x)
    depth_x = self.depth.layer2(depth_x)
    rgb_x, depth_x = self.mmtm2(rgb_x, depth_x)
    rgb_features.append(rgb_x)
    depth_features.append(depth_x)

    rgb_x = self.rgb.layer3(rgb_x)
    depth_x = self.depth.layer3(depth_x)
    rgb_x, depth_x = self.mmtm3(rgb_x, depth_x)
    rgb_features.append(rgb_x)
    depth_features.append(depth_x)

    rgb_x = self.rgb.layer4(rgb_x)
    depth_x = self.depth.layer4(depth_x)
    rgb_x, depth_x = self.mmtm4(rgb_x, depth_x)
    rgb_features.append(rgb_x)
    depth_features.append(depth_x)

    rgb_x = self.rgb.avgpool(rgb_x)
    rgb_x = rgb_x.view(rgb_x.size(0), -1)
    rgb_x = self.rgb.fc(rgb_x)
    depth_x = self.depth.avgpool(depth_x)
    depth_x = depth_x.view(depth_x.size(0), -1)
    depth_x = self.depth.fc(depth_x)
    rgb_features.append(rgb_x)
    depth_features.append(depth_x)

    if self.return_interm_feas:
      return rgb_features, depth_features

    ### LATE FUSION
    if self.final_pred is None:
      pred = (rgb_x + depth_x)/2
    else:
      pred = self.final_pred(torch.cat([rgb_x, depth_x], dim=-1))

    if self.return_both:
      return rgb_x, depth_x

    return pred


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")