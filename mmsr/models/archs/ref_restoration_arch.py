import pdb

import mmsr.models.archs.arch_util as arch_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmsr.models.archs.DCNv2.DCN.dcn_v2 import DCN_sep_pre_multi_offset as DynAgg
from mmsr.models.archs.DCNv2.DCN.dcn_v2 import DCN_pre_offset as DynWarp

class ContentExtractor(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=nf)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.default_init_weights([self.conv_first], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body(feat)

        return feat


class RestorationNet(nn.Module):

    def __init__(self, ngf=64, n_blocks=16, groups=8):
        super(RestorationNet, self).__init__()
        self.content_extractor = ContentExtractor(in_nc=3, out_nc=3, nf=ngf, n_blocks=n_blocks)
        self.dyn_agg_restore = DynamicAggregationRestoration(ngf, n_blocks, groups)

        arch_util.srntt_init_weights(self, init_type='normal', init_gain=0.02)
        self.re_init_dcn_offset()

    def re_init_dcn_offset(self):
        self.dyn_agg_restore.small_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.small_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.medium_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.medium_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.large_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.large_dyn_agg.conv_offset_mask.bias.data.zero_()

        # self.dyn_agg_restore.small_deform_conv.conv_offset_mask.weight.data.zero_()
        # self.dyn_agg_restore.small_deform_conv.conv_offset_mask.bias.data.zero_()
        # self.dyn_agg_restore.medium_deform_conv.conv_offset_mask.weight.data.zero_()
        # self.dyn_agg_restore.medium_deform_conv.conv_offset_mask.bias.data.zero_()
        # self.dyn_agg_restore.large_deform_conv.conv_offset_mask.weight.data.zero_()
        # self.dyn_agg_restore.large_deform_conv.conv_offset_mask.bias.data.zero_()



    def forward(self, x, pre_offset, img_ref_feat,max_val):
        """
        Args:
            x (Tensor): the input image of SRNTT.
            maps (dict[Tensor]): the swapped feature maps on relu3_1, relu2_1
                and relu1_1. depths of the maps are 256, 128 and 64
                respectively.
        """
        #pdb.set_trace()
        base = F.interpolate(x, None, 4, 'bilinear', False)
        content_feat = self.content_extractor(x) # LR
        # pdb.set_trace()
        upscale_restore = self.dyn_agg_restore(content_feat, pre_offset, img_ref_feat,max_val)
        return upscale_restore + base # SR


class DynamicAggregationRestoration(nn.Module):

    def __init__(self, ngf=64, n_blocks=16, groups=8):
        super(DynamicAggregationRestoration, self).__init__()
        # self.small_deform_conv = DynWarp(256,256,3,stride=1,padding=1,dilation=1,deformable_groups=1,extra_offset_mask=False)
        # dynamic aggregation module for relu3_1 reference feature
        self.small_offset_conv1 = nn.Conv2d(ngf + 256, 256, 3, 1, 1, bias=True)  # concat for diff
        self.small_offset_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=True)
        self.small_dyn_agg = DynAgg(256,256,3,stride=1,padding=1,dilation=1,deformable_groups=groups, extra_offset_mask=True)

        # for small scale restoration
        #self.small_3_1 = nn.Conv2d(256*3, 256, 3, 1, 1, bias=True)
        self.head_small = nn.Sequential(nn.Conv2d(ngf + 256*2, ngf, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1, True))
        self.body_small = arch_util.make_layer(arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.tail_small = nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))

        # self.medium_deform_conv = DynWarp(128, 128, 3, stride=1, padding=1, dilation=1, deformable_groups=1,extra_offset_mask=False)
        # dynamic aggregation module for relu2_1 reference feature
        self.medium_offset_conv1 = nn.Conv2d(ngf + 128, 128, 3, 1, 1, bias=True)
        self.medium_offset_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.medium_dyn_agg = DynAgg(64*2,64*2,3,stride=1,padding=1,dilation=1,deformable_groups=groups,extra_offset_mask=True)

        # for medium scale restoration
        #self.medium_3_1 = nn.Conv2d(128 * 3, 128, 3, 1, 1, bias=True)
        self.head_medium = nn.Sequential(nn.Conv2d(ngf + 128*2, ngf, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1, True))
        self.body_medium = arch_util.make_layer(arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.tail_medium = nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))

        # self.large_deform_conv = DynWarp(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=1,extra_offset_mask=False)
        # dynamic aggregation module for relu1_1 reference feature
        self.large_offset_conv1 = nn.Conv2d(ngf + 64, 64, 3, 1, 1, bias=True)
        self.large_offset_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.large_dyn_agg = DynAgg(32*2,32*2,3,stride=1,padding=1,dilation=1,deformable_groups=groups,extra_offset_mask=True)

        # for large scale
        #self.large_3_1 = nn.Conv2d(64 * 3, 64, 3, 1, 1, bias=True)
        self.head_large = nn.Sequential(nn.Conv2d(ngf + 64*2, ngf, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.1, True))
        self.body_large = arch_util.make_layer(arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.tail_large = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1))

        self.lrelu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)


        # k = 2
        # self.small_offset_conv12 = nn.Conv2d(ngf + 256, 256, 3, 1, 1, bias=True)  # concat for diff
        # self.small_offset_conv22 = nn.Conv2d(256, 256, 3, 1, 1, bias=True)
        # self.small_dyn_agg2 = DynAgg(128, 128, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
        #                             extra_offset_mask=True)
        #
        # # for small scale restoration
        # # self.small_3_1 = nn.Conv2d(256*3, 256, 3, 1, 1, bias=True)
        # self.head_small2 = nn.Sequential(nn.Conv2d(ngf + 256, ngf, kernel_size=3, stride=1, padding=1),
        #                                 nn.LeakyReLU(0.1, True))
        # self.body_small2 = arch_util.make_layer(arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        # self.tail_small2 = nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1), nn.PixelShuffle(2),
        #                                 nn.LeakyReLU(0.1, True))
        #
        # # self.medium_deform_conv = DynWarp(128, 128, 3, stride=1, padding=1, dilation=1, deformable_groups=1,extra_offset_mask=False)
        # # dynamic aggregation module for relu2_1 reference feature
        # self.medium_offset_conv12 = nn.Conv2d(ngf + 128, 128, 3, 1, 1, bias=True)
        # self.medium_offset_conv22 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        # self.medium_dyn_agg2 = DynAgg(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
        #                              extra_offset_mask=True)
        #
        # # for medium scale restoration
        # # self.medium_3_1 = nn.Conv2d(128 * 3, 128, 3, 1, 1, bias=True)
        # self.head_medium2 = nn.Sequential(nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1),
        #                                  nn.LeakyReLU(0.1, True))
        # self.body_medium2 = arch_util.make_layer(arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        # self.tail_medium2 = nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
        #                                  nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))
        #
        # # self.large_deform_conv = DynWarp(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=1,extra_offset_mask=False)
        # # dynamic aggregation module for relu1_1 reference feature
        # self.large_offset_conv12 = nn.Conv2d(ngf + 64, 64, 3, 1, 1, bias=True)
        # self.large_offset_conv22 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        # self.large_dyn_agg2 = DynAgg(32, 32, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
        #                             extra_offset_mask=True)
        #
        # # for large scale
        # # self.large_3_1 = nn.Conv2d(64 * 3, 64, 3, 1, 1, bias=True)
        # self.head_large2 = nn.Sequential(nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1),
        #                                 nn.LeakyReLU(0.1, True))
        # self.body_large2 = arch_util.make_layer(arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        # self.tail_large2 = nn.Sequential(
        #     nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1))

        self.lrelu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(256, 128, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)

    def forward(self, x, pre_offset, img_ref_feat , max_val):
        # dynamic aggregation for relu3_1 reference feature
        # pdb.set_trace()
        # x.size = [9,64,40,40]
        # img_ref_feat = [9,256,40,40],[9,256,40,40],[9,256,40,40],[9,256,40,40]

        # MASA [9,3,160,160] -> [9,64,160,160]
        # C2-matching [9,3,160,160] -> [9,256,40,40]
        input = x
        k1, k2, k3, k4, k5 = pre_offset['relu3_1'].size()
        topk = 2
        # pdb.set_trace()
        pre_offset_relu3_1 = pre_offset['relu3_1'].view(k1 // topk, topk, k2, k3, k4, k5)
        k1, k2, k3, k4, k5 = pre_offset['relu2_1'].size()
        pre_offset_relu2_1 = pre_offset['relu2_1'].view(k1 // topk, topk, k2, k3, k4, k5)
        k1, k2, k3, k4, k5 = pre_offset['relu1_1'].size()
        pre_offset_relu1_1 = pre_offset['relu1_1'].view(k1 // topk, topk, k2, k3, k4, k5)
        max_val = torch.softmax(max_val*10 , dim = 1)
        # pdb.set_trace()
        # for i in range(topk):
        x = input
        # img_ref_1 = self.small_deform_conv(img_ref_feat['relu3_1'],pre_offset_relu3_1[:,i,:,:,:,:])
        # relu3_offset = torch.cat([x, img_ref_1], 1)
        relu3_offset = torch.cat([x, img_ref_feat['relu3_1']], 1)
        relu3_offset = self.lrelu1(self.small_offset_conv1(relu3_offset))
        relu3_offset = self.lrelu1(self.small_offset_conv2(relu3_offset))
        # relu3_offset = self.conv1(relu3_offset)
        # img_ref_feat['relu3_1'] = self.conv1(img_ref_feat['relu3_1'])

        relu3_swapped_feat_1 = self.lrelu1(self.small_dyn_agg([img_ref_feat['relu3_1'], relu3_offset], pre_offset_relu3_1[:,0,:,:,:,:]))
        relu3_swapped_feat_2 = self.lrelu2(self.small_dyn_agg([img_ref_feat['relu3_1'], relu3_offset], pre_offset_relu3_1[:,1,:,:,:,:]))
        h = torch.cat([x, relu3_swapped_feat_1,relu3_swapped_feat_2], 1)

        # pdb.set_trace()
        h = self.head_small(h)
        h = self.body_small(h) + x
        x = self.tail_small(h)

        # dynamic aggregation for relu2_1 reference feature
        relu2_offset = torch.cat([x, img_ref_feat['relu2_1']], 1)
        relu2_offset = self.lrelu1(self.medium_offset_conv1(relu2_offset))
        relu2_offset = self.lrelu1(self.medium_offset_conv2(relu2_offset))
        # relu2_offset = self.conv2(relu2_offset)
        # img_ref_feat['relu2_1'] = self.conv2(img_ref_feat['relu2_1'])
        relu2_swapped_feat_1 = self.lrelu1(self.medium_dyn_agg([img_ref_feat['relu2_1'], relu2_offset], pre_offset_relu2_1[:,0,:,:,:,:]))
        relu2_swapped_feat_2 = self.lrelu1(self.medium_dyn_agg([img_ref_feat['relu2_1'], relu2_offset], pre_offset_relu2_1[:,1,:,:,:,:]))

        h = torch.cat([x, relu2_swapped_feat_1,relu2_swapped_feat_2], 1)
        h = self.head_medium(h)
        h = self.body_medium(h) + x
        x = self.tail_medium(h)

        # img_ref_3 = self.large_deform_conv(img_ref_feat['relu1_1'], pre_offset_relu1_1[:, i, :, :, :, :])
        # relu1_offset = torch.cat([x, img_ref_3], 1)
        relu1_offset = torch.cat([x, img_ref_feat['relu1_1']], 1)
        relu1_offset = self.lrelu1(self.large_offset_conv1(relu1_offset))
        relu1_offset = self.lrelu1(self.large_offset_conv2(relu1_offset))
        # relu1_offset = self.conv3(relu1_offset)
        # img_ref_feat['relu1_1'] = self.conv3(img_ref_feat['relu1_1'])
        relu1_swapped_feat_1 = self.lrelu1(self.large_dyn_agg([img_ref_feat['relu1_1'], relu1_offset], pre_offset_relu1_1[:,0,:,:,:,:]))
        relu1_swapped_feat_2 = self.lrelu1(self.large_dyn_agg([img_ref_feat['relu1_1'], relu1_offset], pre_offset_relu1_1[:,1,:,:,:,:]))
        h = torch.cat([x, relu1_swapped_feat_1,relu1_swapped_feat_2], 1)
        h = self.head_large(h)
        h = self.body_large(h) + x
        x = self.tail_large(h)
        # pdb.set_trace()
        return x
