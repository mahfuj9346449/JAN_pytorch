import argparse
import pdb
import os
import shutil
import time
import itertools

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

global_iter = 0


class GRLayer(torch.autograd.Function):
    def __init__(self, max_iter=2000, alpha=10., high=1.):
        super(GRLayer, self).__init__()
        self.total = float(max_iter)
        self.alpha = alpha
        self.high = high

    def forward(self, input):
        return input.view_as(input)

    def backward(self, gradOutput):
        prog = global_iter / self.total
        lr = 2.*self.high / (1 + math.exp(-self.alpha * prog)) - self.high
        return (-lr) * gradOutput


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Generator,self).__init__()
        self.conv1 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv7 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv8 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1)
        self.dconv6 = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1)
        self.dconv7 = nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1)
        self.dconv8 = nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1)

        self.batch_norm = nn.BatchNorm2d(ngf)
        self.batch_norm2 = nn.BatchNorm2d(ngf * 2)
        self.batch_norm4 = nn.BatchNorm2d(ngf * 4)
        self.batch_norm8 = nn.BatchNorm2d(ngf * 8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        e1 = self.conv1(input)
        # state size is (ngf) x 128 x 128
        e2 = self.batch_norm2(self.conv2(self.leaky_relu(e1)))
        # state size is (ngf x 2) x 64 x 64
        e3 = self.batch_norm4(self.conv3(self.leaky_relu(e2)))
        # state size is (ngf x 4) x 32 x 32
        e4 = self.batch_norm8(self.conv4(self.leaky_relu(e3)))
        # state size is (ngf x 8) x 16 x 16
        e5 = self.batch_norm8(self.conv5(self.leaky_relu(e4)))
        # state size is (ngf x 8) x 8 x 8
        e6 = self.batch_norm8(self.conv6(self.leaky_relu(e5)))
        # state size is (ngf x 8) x 4 x 4
        e7 = self.batch_norm8(self.conv7(self.leaky_relu(e6)))
        # state size is (ngf x 8) x 2 x 2
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 1 x 1
        d1_ = self.dropout(self.batch_norm8(self.dconv1(self.relu(e8))))
        # state size is (ngf x 8) x 2 x 2
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.dropout(self.batch_norm8(self.dconv2(self.relu(d1))))
        # state size is (ngf x 8) x 4 x 4
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.dropout(self.batch_norm8(self.dconv3(self.relu(d2))))
        # state size is (ngf x 8) x 8 x 8
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.batch_norm8(self.dconv4(self.relu(d3)))
        # state size is (ngf x 8) x 16 x 16
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.batch_norm4(self.dconv5(self.relu(d4)))
        # state size is (ngf x 4) x 32 x 32
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.batch_norm2(self.dconv6(self.relu(d5)))
        # state size is (ngf x 2) x 64 x 64
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.batch_norm(self.dconv7(self.relu(d6)))
        # state size is (ngf) x 128 x 128
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.relu(d7))
        # state size is (nc) x 256 x 256
        output = self.tanh(d8)
        return output


class Discriminator(nn.Module):
    def __init__(self,input_nc,output_nc,ndf):
        super(Discriminator,self).__init__()
        # 256 x 256
        self.layer1 = nn.Sequential(nn.Conv2d(input_nc+output_nc,ndf,kernel_size=4,stride=2,padding=1),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 128 x 128
        self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*2),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 64 x 64
        self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*4),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 32 x 32
        self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=1,padding=1),
                                 nn.BatchNorm2d(ndf*8),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 31 x 31
        self.layer5 = nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=4,stride=1,padding=1),
                                 nn.Sigmoid())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


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

        self.fcb_s = nn.Linear(self.feature_dim, args.bottleneck)
        self.fcb_s.weight.data.normal_(0, 0.005)
        self.fcb_s.bias.data.fill_(0.1)
        # self.fcb_t = nn.Linear(self.feature_dim, args.bottleneck)
        # self.fcb_t.weight.data.normal_(0, 0.005)
        # self.fcb_t.bias.data.fill_(0.1)
        self.fcb_t = self.fcb_s
        self.fc_s = nn.Linear(args.bottleneck, args.classes)
        self.fc_s.weight.data.normal_(0, 0.01)
        self.fc_s.bias.data.fill_(0.0)
        # self.fc_t = nn.Linear(args.bottleneck, args.classes)
        # self.fc_t.weight.data.normal_(0, 0.01)
        # self.fc_t.bias.data.fill_(0.0)
        self.fc_t = self.fc_s

        self.W_st = create_W([args.bottleneck, args.bottleneck])
        self.W_ts = create_W([args.bottleneck, args.bottleneck])

        self.D_s = create_D([args.bottleneck, 1024, 1024, 1])
        self.D_t = create_D([args.bottleneck, 1024, 1024, 1])

        self.grl_ss = GRLayer()
        self.grl_tt = GRLayer()
        self.grl_st = GRLayer()
        self.grl_ts = GRLayer()

        args.SGD_param = [
            {'params': self.origin_feature.parameters(), 'lr': 1,},
            {
                'params': itertools.chain(self.fcb_s.parameters(),
                                          self.fcb_t.parameters()),
                'lr': 10
            },
            {
                'params': itertools.chain(self.fc_s.parameters(),
                                          self.fc_t.parameters()),
                'lr': 10
            },
            {
                'params': itertools.chain(self.W_ts.parameters(),
                                          self.W_st.parameters()),
                'lr': 10
            },
            {
                'params': itertools.chain(self.D_s.parameters(),
                                          self.D_t.parameters()),
                'lr': 10
            }
        ]

    def forward(self, x, train=True):
        x = self.origin_feature(x)
        if self.arch.startswith('densenet'):
            x = F.relu(x, inplace=True)
            x = F.avg_pool2d(x, kernel_size=7)
        x = x.view(x.size(0), -1)
        x = torch.autograd.Variable(x.data)
        if train:
            orign_feature_s, orign_feature_t = x.chunk(2, 0)
            feature_s = self.fcb_s(orign_feature_s)
            feature_t = self.fcb_t(orign_feature_t)
            output_s = self.fc_s(feature_s)
            output_t = self.fc_t(feature_t)
            fake_feature_t = self.W_st(feature_s)
            fake_feature_s = self.W_ts(feature_t)
            cycle_s = self.W_ts(fake_feature_t)
            cycle_t = self.W_st(fake_feature_s)
            fake_output_t = self.fc_t(fake_feature_t)
            discriminate_s = self.D_s(torch.cat([self.grl_ss(fake_feature_s),
                                                 self.grl_st(feature_s)], 0))
            discriminate_t = self.D_t(torch.cat([self.grl_ts(fake_feature_t),
                                                 self.grl_tt(feature_t)], 0))
            return (feature_s, feature_t), \
                   (cycle_s, cycle_t), \
                   (output_s, output_t),\
                   (fake_output_t,), \
                   (discriminate_s, discriminate_t)
        else:
            return self.fc_t(self.fcb_t(x))


def L2loss(source, target):
    return torch.sum((source - target) ** 2)


def train_val(source_loader, target_loader, val_loader, model, criterion, optimizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    source_cycle = itertools.cycle(source_loader)
    target_cycle = itertools.cycle(target_loader)

    end = time.time()
    model.eval()
    cycle_criterion = L2loss
    discriminate_criterion = nn.BCELoss()
    for i in range(args.train_iter):
        global global_iter
        global_iter = i
        adjust_learning_rate(optimizer, i, args)
        data_time.update(time.time() - end)
        source_input, label = source_cycle.next()
        if source_input.size()[0] < args.batch_size:
            source_input, label = source_cycle.next()
        target_input, _ = target_cycle.next()
        if target_input.size()[0] < args.batch_size:
            target_input, _ = target_cycle.next()
        domain_label = torch.autograd.Variable(
            torch.cat([torch.zeros(source_input.size()[0]),
                       torch.ones(source_input.size()[0])], 0)).cuda()
        label = label.cuda(async=True)
        source_var = torch.autograd.Variable(source_input)
        target_var = torch.autograd.Variable(target_input)
        label_var = torch.autograd.Variable(label)

        inputs = torch.cat([source_var, target_var], 0)
        (feature_s, feature_t), (cycle_s, cycle_t), \
            (output_s, output_t), (fake_output_t,), \
            (discriminate_s, discriminate_t) = model(inputs)

        acc_loss = criterion(output_s, label_var) \
            + criterion(fake_output_t, label_var)
        cycle_loss = cycle_criterion(feature_s, cycle_s) \
            + cycle_criterion(feature_t, cycle_t)
        discriminate_loss = discriminate_criterion(discriminate_s, domain_label) \
            + discriminate_criterion(discriminate_t, domain_label)

        loss = acc_loss + args.alpha * cycle_loss + args.beta * discriminate_loss
        # loss = discriminate_loss

        prec1, _ = accuracy(output_s.data, label, topk=(1, 5))

        losses.update(loss.data[0], args.batch_size)
        loss1 = acc_loss.data[0]
        loss2 = cycle_loss.data[0]
        loss3 = discriminate_loss.data[0]
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
                  'Loss {loss1:.4f} {loss2:.4f} {loss3:.4f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, args.train_iter, batch_time=batch_time,
                loss=losses, top1=top1, loss1=loss1, loss2=loss2, loss3=loss3))

        if i % args.test_iter == 0 and i != 0:
            validate(val_loader, model, criterion, args)
            model.eval()
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
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var, train=False)
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
