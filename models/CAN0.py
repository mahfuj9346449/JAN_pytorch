import argparse
import pdb
import os
import shutil
import time
import itertools
import functools
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import math

from losses import *
from utils import *


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return nn.parallel.data_parallel(self.model, input)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        # create model
        if args.fromcaffe:
            print("=> using pre-trained model from caffe '{}'".format(args.arch))
            import models.caffe_resnet as resnet
            model = resnet.__dict__[args.arch]()
            state_dict = torch.load("models/"+args.arch+".pth")
            model.load_state_dict(state_dict)
        elif args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            self.feature_dim = model.classifier[6].in_features
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        elif args.arch.startswith('densenet'):
            self.feature_dim = model.classifier.in_features
            model = nn.Sequential(*list(model.children())[:-1])
        else:
            self.feature_dim = model.fc.in_features
            model = nn.Sequential(*list(model.children())[:-1])

        self.origin_feature = torch.nn.DataParallel(model)
        self.model = args.model
        self.arch = args.arch

        self.fcb = nn.Linear(self.feature_dim, args.bottleneck)
        self.fcb.weight.data.normal_(0, 0.005)
        self.fcb.bias.data.fill_(0.1)
        self.fc = nn.Linear(args.bottleneck, args.classes)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0.0)

        args.SGD_param = [
            {'params': self.origin_feature.parameters(), 'lr': 1,},
            {'params': self.fcb.parameters(), 'lr': 10},
            {'params': self.fc.parameters(), 'lr': 10}
        ]

    def forward(self, x):
        x = self.origin_feature(x)
        if self.arch.startswith('densenet'):
            x = F.relu(x, inplace=True)
            x = F.avg_pool2d(x, kernel_size=7)
        x = x.view(x.size(0), -1)
        x = self.fcb(x)
        y = self.fc(x)

        return y, x



def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='instance', use_dropout=False):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    netG.cuda()
    netG.apply(weights_init)
    return netG


def convert_state_dict(state_dict):
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.encode('utf-8')] = v
    return new_state_dict


origin_mean = torch.autograd.Variable(torch.Tensor([0.5, 0.5, 0.5])).view(-1, 1, 1).cuda()
origin_std = torch.autograd.Variable(torch.Tensor([0.5, 0.5, 0.5])).view(-1, 1, 1).cuda()
new_mean = torch.autograd.Variable(torch.Tensor([0.485, 0.456, 0.406])).view(-1, 1, 1).cuda()
new_std = torch.autograd.Variable(torch.Tensor([0.229, 0.224, 0.225])).view(-1, 1, 1).cuda()


def renormalize(input):
    input = input * origin_std + origin_mean
    return (input - new_mean) / new_std


def train_val(source_loader, target_loader, val_loader, model, criterion, optimizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    source_cycle = itertools.cycle(source_loader)
    target_cycle = itertools.cycle(target_loader)

    end = time.time()
    netG_A = define_G(3, 3, 64, 'resnet_9blocks')
    state_dict = convert_state_dict(torch.load('/home/sun/pytorch-CycleGAN-and-pix2pix/checkpoints/office-cycle_gan/latest_net_G_A.pth'))
    netG_A.load_state_dict(state_dict)
    netG_A.eval()
    #netG_B = define_G(3, 3, 64, 'resnet_9blocks')
    #state_dict = convert_state_dict(torch.load('/home/sun/pytorch-CycleGAN-and-pix2pix/checkpoints/office-cycle_gan/latest_net_G_B.pth'))
    #netG_B.load_state_dict(state_dict)
    #netG_B.eval()
    model.train()
    for i in range(args.train_iter):
        global global_iter
        global_iter = i
        adjust_learning_rate(optimizer, i, args)
        data_time.update(time.time() - end)

        source_input, label = next(source_cycle)
        target_input, _ = next(target_cycle)
        if source_input.size()[0] < args.batch_size or target_input.size()[0] < args.batch_size:
            source_cycle = iter(source_loader)
            target_cycle = iter(target_loader)
            source_input, label = next(source_cycle)
            target_input, _ = next(target_cycle)

        label = label.cuda(async=True)
        source_var = torch.autograd.Variable(source_input).cuda()
        target_var = torch.autograd.Variable(target_input).cuda()
        label_var = torch.autograd.Variable(label)

        fake_target_var = netG_A(source_var)
        inputs = torch.cat([fake_target_var, target_var], 0)
        outputs, features = model(renormalize(inputs))
        fake_target_output, target_output = outputs.chunk(2, 0)
        fake_target_feature, target_feature = features.chunk(2, 0)

        acc_loss = criterion(fake_target_output, label_var)
        softmax = nn.Softmax()
        jmmd_loss = JMMDLoss([fake_target_feature, softmax(fake_target_output)],
                             [target_feature, softmax(target_output)])

        loss = acc_loss + jmmd_loss

        prec1, _ = accuracy(fake_target_output.data, label, topk=(1, 5))

        losses.update(loss.data[0], args.batch_size)
        loss1 = jmmd_loss.data[0]
        loss2 = acc_loss.data[0]
        top1.update(prec1[0], args.batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Iter: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss1:.4f}/{loss2:.4f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, args.train_iter, batch_time=batch_time,
                loss=losses, top1=top1, loss1=loss1, loss2=loss2))

        if i % args.test_iter == 0 and i != 0:
            validate(val_loader, model, criterion, args)
            model.train()
            batch_time.reset()
            data_time.reset()
            losses.reset()
            top1.reset()


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output, _ = model(renormalize(input_var))
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg
