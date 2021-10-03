# modified from https://github.com/lorenmt/mtan/blob/master/im2im_pred/model_resnet_mtan/resnet_mtan.py
#           and https://github.com/lorenmt/mtan/blob/master/visual_decathlon/model_wrn_mtan.py

import torch, sys
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('./model')
import resnet
from resnet_dilated import ResnetDilated
from aspp import DeepLabHead
from resnet import Bottleneck, conv1x1


class Model_h(nn.Module):
    def __init__(self, task_num):
        super(Model_h, self).__init__()
        self.loss_combine = nn.Parameter(torch.FloatTensor(task_num))
        
        nn.init.constant_(self.loss_combine, 1/task_num)
        
    def forward(self, loss):
        self.weight = torch.softmax(self.loss_combine, dim=0)
        final_loss = torch.sum(torch.mul(self.weight, loss))
        return final_loss


class DeepLabv3(nn.Module):
    def __init__(self):
        super(DeepLabv3, self).__init__()
        self.backbone = ResnetDilated(resnet.__dict__['resnet50'](pretrained=True))
        ch = [256, 512, 1024, 2048]

        self.class_nb = 13

        self.tasks = ['segmentation', 'depth', 'normal']
        self.num_out_channels = {'segmentation': self.class_nb, 'depth': 1, 'normal': 3}
        
        self.decoders = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x):
        img_size  = x.size()[-2:]
        x = self.backbone(x)
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](x), img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
    
    def get_share_params(self):
        return self.backbone.parameters()


class MTANDeepLabv3(nn.Module):
    def __init__(self):
        super(MTANDeepLabv3, self).__init__()
        backbone = ResnetDilated(resnet.__dict__['resnet50'](pretrained=True))
        ch = [256, 512, 1024, 2048]
        
        self.class_nb = 13
        
        self.tasks = ['segmentation', 'depth', 'normal']
        self.num_out_channels = {'segmentation': 13, 'depth': 1, 'normal': 3}
        
        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.maxpool)

        # We will apply the attention over the last bottleneck layer in the ResNet. 
        self.shared_layer1_b = backbone.layer1[:-1] 
        self.shared_layer1_t = backbone.layer1[-1]

        self.shared_layer2_b = backbone.layer2[:-1]
        self.shared_layer2_t = backbone.layer2[-1]

        self.shared_layer3_b = backbone.layer3[:-1]
        self.shared_layer3_t = backbone.layer3[-1]

        self.shared_layer4_b = backbone.layer4[:-1]
        self.shared_layer4_t = backbone.layer4[-1]

        # Define task specific attention modules using a similar bottleneck design in residual block
        # (to avoid large computations)
        self.encoder_att_1 = nn.ModuleList([self.att_layer(ch[0], ch[0] // 4, ch[0]) for _ in self.tasks])
        self.encoder_att_2 = nn.ModuleList([self.att_layer(2 * ch[1], ch[1] // 4, ch[1]) for _ in self.tasks])
        self.encoder_att_3 = nn.ModuleList([self.att_layer(2 * ch[2], ch[2] // 4, ch[2]) for _ in self.tasks])
        self.encoder_att_4 = nn.ModuleList([self.att_layer(2 * ch[3], ch[3] // 4, ch[3]) for _ in self.tasks])

        # Define task shared attention encoders using residual bottleneck layers
        # We do not apply shared attention encoders at the last layer,
        # so the attended features will be directly fed into the task-specific decoders.
        self.encoder_block_att_1 = self.conv_layer(ch[0], ch[1] // 4)
        self.encoder_block_att_2 = self.conv_layer(ch[1], ch[2] // 4)
        self.encoder_block_att_3 = self.conv_layer(ch[2], ch[3] // 4)
        
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList([DeepLabHead(2048, self.num_out_channels[t]) for t in self.tasks])
        
    def forward(self, x):
        img_size  = x.size()[-2:]
        # Shared convolution
        x = self.shared_conv(x)
        
        # Shared ResNet block 1
        u_1_b = self.shared_layer1_b(x)
        u_1_t = self.shared_layer1_t(u_1_b)

        # Shared ResNet block 2
        u_2_b = self.shared_layer2_b(u_1_t)
        u_2_t = self.shared_layer2_t(u_2_b)

        # Shared ResNet block 3
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)
        
        # Shared ResNet block 4
        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)

        # Attention block 1 -> Apply attention over last residual block
        a_1_mask = [att_i(u_1_b) for att_i in self.encoder_att_1]  # Generate task specific attention map
        a_1 = [a_1_mask_i * u_1_t for a_1_mask_i in a_1_mask]  # Apply task specific attention map to shared features
        a_1 = [self.down_sampling(self.encoder_block_att_1(a_1_i)) for a_1_i in a_1]
        
        # Attention block 2 -> Apply attention over last residual block
        a_2_mask = [att_i(torch.cat((u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)]
        a_2 = [a_2_mask_i * u_2_t for a_2_mask_i in a_2_mask]
        a_2 = [self.encoder_block_att_2(a_2_i) for a_2_i in a_2]
        
        # Attention block 3 -> Apply attention over last residual block
        a_3_mask = [att_i(torch.cat((u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_att_3)]
        a_3 = [a_3_mask_i * u_3_t for a_3_mask_i in a_3_mask]
        a_3 = [self.encoder_block_att_3(a_3_i) for a_3_i in a_3]
        
        # Attention block 4 -> Apply attention over last residual block (without final encoder)
        a_4_mask = [att_i(torch.cat((u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, self.encoder_att_4)]
        a_4 = [a_4_mask_i * u_4_t for a_4_mask_i in a_4_mask]
        
        # Task specific decoders
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](a_4[i]), size=img_size, mode='bilinear', align_corners=True)
            if t == 'segmentation':
                out[i] = F.log_softmax(out[i], dim=1)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out
    
    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid())
        
    def conv_layer(self, in_channel, out_channel):
        downsample = nn.Sequential(conv1x1(in_channel, 4 * out_channel, stride=1),
                                   nn.BatchNorm2d(4 * out_channel))
        return Bottleneck(in_channel, out_channel, downsample=downsample)
    
    def get_share_params(self):
        p = []
        p += self.shared_conv.parameters()
        p += self.shared_layer1_b.parameters()
        p += self.shared_layer2_b.parameters()
        p += self.shared_layer3_b.parameters()
        p += self.shared_layer4_b.parameters()
        p += self.shared_layer1_t.parameters()
        p += self.shared_layer2_t.parameters()
        p += self.shared_layer3_t.parameters()
        p += self.shared_layer4_t.parameters()
        p += self.encoder_att_1.parameters()
        p += self.encoder_att_2.parameters()
        p += self.encoder_att_3.parameters()
        p += self.encoder_att_4.parameters()
        p += self.encoder_block_att_1.parameters()
        p += self.encoder_block_att_2.parameters()
        p += self.encoder_block_att_3.parameters()
        p += self.down_sampling.parameters()
        return p


######################



class DMTL(nn.Module):
    def __init__(self, task_num, base_net='resnet18', hidden_dim=512, class_num=31):
        super(DMTL, self).__init__()
        # base network
        self.base_network = resnet.__dict__[base_net](pretrained=True)
        # shared layer
        self.avgpool = self.base_network.avgpool
        self.hidden_layer_list = [nn.Linear(512, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.hidden_layer = nn.Sequential(*self.hidden_layer_list)
        # task-specific layer
        self.classifier_parameter = nn.Parameter(torch.FloatTensor(task_num, hidden_dim, class_num))

        # initialization
        self.hidden_layer[0].weight.data.normal_(0, 0.005)
        self.hidden_layer[0].bias.data.fill_(0.1)
        self.classifier_parameter.data.normal_(0, 0.01)

        # self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))

    def forward(self, inputs, task_index):
        features = self.base_network(inputs)
        features = torch.flatten(self.avgpool(features), 1)
        hidden_features = self.hidden_layer(features)
        outputs = torch.mm(hidden_features, self.classifier_parameter[task_index])
        return outputs

    def predict(self, inputs, task_index):
        return self.forward(inputs, task_index)

    def get_share_params(self):
        p = []
        p += self.base_network.parameters()
        p += self.hidden_layer.parameters()
        return p

class MTAN_ResNet(nn.Module):
    def __init__(self, task_num, num_classes):
        super(MTAN_ResNet, self).__init__()
        backbone = resnet.__dict__['resnet18'](pretrained=True)
        self.task_num = task_num
        filter = [64, 128, 256, 512]
#         filter = [256, 512, 1024, 2048]

        self.conv1, self.bn1, self.relu1, self.maxpool = backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.linear = nn.ModuleList([nn.Linear(filter[-1], num_classes) for _ in range(self.task_num)])

        # attention modules
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])

        for j in range(self.task_num):
            if j < self.task_num-1:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))

            for i in range(3):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))

        for i in range(3):
            if i < 2:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

    def forward(self, x, k):
        g_encoder = [0] * 4

        atten_encoder = [0] * 4
        for i in range(4):
            atten_encoder[i] = [0] * 3

        # shared encoder
        x = self.maxpool(self.relu1(self.bn1(self.conv1(x))))
        g_encoder[0] = self.layer1(x)
        g_encoder[1] = self.layer2(g_encoder[0])
        g_encoder[2] = self.layer3(g_encoder[1])
        g_encoder[3] = self.layer4(g_encoder[2])

        # apply attention modules
        for j in range(4):
            if j == 0:
                atten_encoder[j][0] = self.encoder_att[k][j](g_encoder[0])
                atten_encoder[j][1] = (atten_encoder[j][0]) * g_encoder[0]
                atten_encoder[j][2] = self.encoder_block_att[j](atten_encoder[j][1])
                atten_encoder[j][2] = F.max_pool2d(atten_encoder[j][2], kernel_size=2, stride=2)
            else:
                atten_encoder[j][0] = self.encoder_att[k][j](torch.cat((g_encoder[j], atten_encoder[j - 1][2]), dim=1))
                atten_encoder[j][1] = (atten_encoder[j][0]) * g_encoder[j]
                atten_encoder[j][2] = self.encoder_block_att[j](atten_encoder[j][1])
                if j < 3:
                    atten_encoder[j][2] = F.max_pool2d(atten_encoder[j][2], kernel_size=2, stride=2)

        pred = self.avgpool(atten_encoder[-1][-1])
        pred = pred.view(pred.size(0), -1)

        out = self.linear[k](pred)
        return out
    
    def conv_layer(self, channel):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channel[1]),
            nn.ReLU(inplace=True),
        )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def get_share_params(self):
        p = []
        p += self.conv1.parameters()
        p += self.bn1.parameters()
        p += self.layer1.parameters()
        p += self.layer2.parameters()
        p += self.layer3.parameters()
        p += self.layer4.parameters()
        p += self.encoder_att.parameters()
        p += self.encoder_block_att.parameters()
        return p
