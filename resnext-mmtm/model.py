import torch
from torch import nn

from models import c3d, squeezenet, mobilenet, shufflenet, mobilenetv2, shufflenetv2, resnext, resnet, resnetl, mmtnet
import pdb

def generate_model(opt):
    assert opt.model in ['c3d', 'squeezenet', 'mobilenet', 'resnext', 'resnet', 'resnetl',
                         'shufflenet', 'mobilenetv2', 'shufflenetv2', 'mmtnet']


    if opt.model == 'c3d':
        from models.c3d import get_fine_tuning_parameters
        model = c3d.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)
    elif opt.model == 'squeezenet':
        from models.squeezenet import get_fine_tuning_parameters
        model = squeezenet.get_model(
            version=opt.version,
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)
    elif opt.model == 'shufflenet':
        from models.shufflenet import get_fine_tuning_parameters
        model = shufflenet.get_model(
            groups=opt.groups,
            width_mult=opt.width_mult,
            num_classes=opt.n_classes)
    elif opt.model == 'shufflenetv2':
        from models.shufflenetv2 import get_fine_tuning_parameters
        model = shufflenetv2.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)
    elif opt.model == 'mobilenet':
        from models.mobilenet import get_fine_tuning_parameters
        model = mobilenet.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)
    elif opt.model == 'mobilenetv2':
        from models.mobilenetv2 import get_fine_tuning_parameters
        model = mobilenetv2.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)
    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]
        from models.resnext import get_fine_tuning_parameters
        model = resnext.get_model(opt.model_depth,
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            cardinality=opt.resnext_cardinality,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)
    elif opt.model == 'resnetl':
        assert opt.model_depth in [10]

        from models.resnetl import get_fine_tuning_parameters

        if opt.model_depth == 10:
            model = resnetl.resnetl10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        from models.resnet import get_fine_tuning_parameters
        model = resnet.get_model(opt.model_depth,
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)

    elif opt.model == 'mmtnet':
        from models.mmtnet import get_fine_tuning_parameters
        opt.model = 'resnext'
        opt.modality = 'RGB'
        opt.no_cuda = True
        pretrain_dataset = opt.pretrain_dataset
        if opt.pretrain_dataset == opt.dataset:
            opt.pretrain_dataset = ''
        rgb = generate_model(opt)[0]
        opt.modality = 'Depth'
        depth = generate_model(opt)[0]
        opt.model = 'mmtnet'
        opt.modality = 'RGB-D'
        opt.pretrain_dataset = pretrain_dataset
        opt.no_cuda = False
        model = mmtnet.MMTNet(opt)
        model.set_return_both(True)
        model.set_rgb_depth_nets(rgb, depth)


    if not opt.no_cuda:
        model = nn.DataParallel(model, device_ids=None)
        # pytorch_total_params = sum(p.numel() for p in model.parameters() if
        #                        p.requires_grad)
        # print("Total number of trainable parameters: ", pytorch_total_params)
        # if opt.force_change_firstlayer:
        #    model = _modify_first_conv_layer(model,3,3)

        if opt.pretrain_path and opt.model not in ['mmtnet']:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            # pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
            pretrain = torch.load(opt.pretrain_path)
            # print(opt.arch)
            # print(pretrain['arch'])
            # assert opt.arch == pretrain['arch']
            # print('opt.sample_duration', opt.sample_duration)
           # if opt.pretrain_dataset == opt.dataset:
           #    model.load_state_dict(pretrain['state_dict'])
            if opt.pretrain_dataset == 'jester':
                if opt.sample_duration < 32 and opt.model not in ['c3d', 'squeezenet', 'mobilenet','shufflenet', 'mobilenetv2', 'shufflenetv2', 'resnext']:
                    print('_modify_first_conv_layer')
                    model = _modify_first_conv_layer(model,3,3)
                if opt.model in  ['mobilenetv2', 'shufflenetv2']:
                    del pretrain['state_dict']['module.classifier.1.weight']
                    del pretrain['state_dict']['module.classifier.1.bias']
                else:
                    del pretrain['state_dict']['module.fc.weight']
                    del pretrain['state_dict']['module.fc.bias']
                model.load_state_dict(pretrain['state_dict'],strict=False)
        
        if opt.model=='mmtnet':
            if opt.pretrain_dataset == 'hcigesture_allbutnone':
                print('loading pretrained model {}'.format(opt.pretrain_path))
                # pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
                pretrain = torch.load(opt.pretrain_path)
                del pretrain['state_dict']['module.rgb.fc.weight']
                del pretrain['state_dict']['module.rgb.fc.bias']
                del pretrain['state_dict']['module.depth.fc.weight']
                del pretrain['state_dict']['module.depth.fc.bias']
                model.load_state_dict(pretrain['state_dict'],strict=False)
            elif opt.pretrain_dataset == opt.dataset:
                print('loading pretrained model {}'.format(opt.pretrain_path))
                # pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
                pretrain = torch.load(opt.pretrain_path)
                model.load_state_dict(pretrain['state_dict'])
            # if opt.pretrain_dataset == 'imagenet' and opt.model == 'i3d':
                
#            elif opt.pretrain_dataset in ['egogesture', 'nvgesture', 'denso']:
#                del pretrain['state_dict']['module.fc.weight']
#                del pretrain['state_dict']['module.fc.bias']
#                model.load_state_dict(pretrain['state_dict'],strict=False)

        if opt.force_change_firstlayer:
            model = _modify_first_conv_layer(model,3,3)
        if opt.model not in ['mmtnet']:
            model = modify_kernels(opt, model, opt.modality)
            # model.load_state_dict(pretrain['state_dict'])
            if opt.pretrain_dataset == opt.dataset:
                model.load_state_dict(pretrain['state_dict'])
            elif opt.pretrain_dataset in ['egogesture', 'nvgesture', 'denso', 'hcigesture_allbutnone']:
            # elif opt.pretrain_dataset:
                del pretrain['state_dict']['module.fc.weight']
                del pretrain['state_dict']['module.fc.bias']
                model.load_state_dict(pretrain['state_dict'],strict=False)
            
            if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2']:
                model.module.classifier = nn.Sequential(
                                nn.Dropout(0.5),
                                nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes))
                model.module.classifier = model.module.classifier.cuda()
            elif opt.model == 'squeezenet':
                model.module.classifier = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Conv3d(model.module.classifier[1].in_channels, opt.n_finetune_classes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.AvgPool3d((1,4,4), stride=1))
                model.module.classifier = model.module.classifier.cuda()
            elif opt.model == 'c3d':# CHECK HERE
                model.module.fc = nn.Linear(
                    model.module.fc[0].in_features, model.module.fc[0].out_features)
                model.module.fc = model.module.fc.cuda()
            # elif opt.model =='i3d':
            #     model.module.replace_logits(opt.n_finetune_classes, device='cuda:0')
            else:
                model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
                model.module.fc = model.module.fc.cuda()

            # model = modify_kernels(opt, model, opt.modality)
        # else:
        #     model = modify_kernels(opt, model, opt.modality)

        parameters = get_fine_tuning_parameters(model, opt.ft_portion)
        model = model.cuda()
        return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path, map_location='cpu')
            from collections import OrderedDict
            pretrain_new = OrderedDict()
            for k, v in pretrain['state_dict'].items():
                name = k[7:]  # delete `module.`
                pretrain_new[name] = v

            # if opt.model not in ['mmtnet']:
            #     model = modify_kernels(opt, model, opt.pretrain_modality)
            # model.load_state_dict(pretrain['state_dict'])
            # model.load_state_dict(pretrain_new, strict=False)
            if opt.pretrain_dataset == opt.dataset:
                model.load_state_dict(pretrain_new)
            elif opt.pretrain_dataset in ['jester', 'egogesture', 'nvgesture', 'denso']:
                del pretrain_new['fc.weight']
                del pretrain_new['fc.bias']
                model.load_state_dict(pretrain_new,strict=False)

            if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2']:
                model.classifier = nn.Sequential(
                                nn.Dropout(0.9),
                                nn.Linear(model.classifier[1].in_features, opt.n_finetune_classes)
                                )
            elif opt.model == 'squeezenet':
                model.classifier = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Conv3d(model.classifier[1].in_channels, opt.n_finetune_classes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.AvgPool3d((1,4,4), stride=1))
            else:
                model.fc = nn.Linear(model.fc.in_features, opt.n_finetune_classes)
            if opt.model not in ['mmtnet']:
                model = modify_kernels(opt, model, opt.modality)
            parameters = get_fine_tuning_parameters(model, opt.ft_portion)
            return model, parameters
        else:
            if opt.model not in ['mmtnet']:
                model = modify_kernels(opt, model, opt.modality)

    return model, model.parameters()


def _construct_depth_model(base_model):
    # modify the first convolution kernels for Depth input
    modules = list(base_model.modules())

    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                 list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]

    # modify parameters, assume the first blob contains the convolution kernels
    motion_length = 1
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (1*motion_length,  ) + kernel_size[2:]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

    new_conv = nn.Conv3d(1, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride,
                         conv_layer.padding, bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

    # replace the first convlution layer
    setattr(container, layer_name, new_conv)

    return base_model

def _construct_rgbdepth_model(base_model):
    # modify the first convolution kernels for RGB-D input
    modules = list(base_model.modules())

    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                           list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]
    # modify parameters, assume the first blob contains the convolution kernels
    motion_length = 1
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (1 * motion_length,) + kernel_size[2:]
    new_kernels = torch.mul(torch.cat((params[0].data, params[0].data.mean(dim=1,keepdim=True).expand(new_kernel_size).contiguous()), 1), 0.6)
    new_kernel_size = kernel_size[:1] + (3 + 1 * motion_length,) + kernel_size[2:]
    new_conv = nn.Conv3d(4, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride,
                         conv_layer.padding, bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data  # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

    # replace the first convolution layer
    setattr(container, layer_name, new_conv)
    return base_model

def _modify_first_conv_layer(base_model, new_kernel_size1, new_filter_num):
    modules = list(base_model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                               list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]
 
    new_conv = nn.Conv3d(new_filter_num, conv_layer.out_channels, kernel_size=(new_kernel_size1,7,7),
                         stride=(1,2,2), padding=(1,3,3), bias=False)
    layer_name = list(container.state_dict().keys())[0][:-7]

    setattr(container, layer_name, new_conv)
    return base_model

def modify_kernels(opt, model, modality):
    if modality == 'RGB' and opt.model not in ['c3d', 'squeezenet', 'mobilenet','shufflenet', 'mobilenetv2', 'shufflenetv2', 'magnifynet']:
        print("[INFO]: RGB model is used for init model")
        if opt.dataset != 'jester' and not opt.no_change_firstlayer:
            model = _modify_first_conv_layer(model,3,3) ##### Check models trained (3,7,7) or (7,7,7)
    elif modality == 'Depth':
        print("[INFO]: Converting the pretrained model to Depth init model")
        model = _construct_depth_model(model)
        print("[INFO]: Done. Flow model ready.")
    elif modality == 'RGB-D':
        print("[INFO]: Converting the pretrained model to RGB+D init model")
        model = _construct_rgbdepth_model(model)
        if opt.no_change_firstlayer:
            model = _modify_first_conv_layer(model,3,4)
        print("[INFO]: Done. RGB-D model ready.")
    modules = list(model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                               list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    if conv_layer.kernel_size[0]> opt.sample_duration:
       model = _modify_first_conv_layer(model,int(opt.sample_duration/2),1)
    return model
