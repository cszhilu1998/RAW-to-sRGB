import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from . import losses as L
from pwc import pwc_net
from . import ispjoint_model
from util.util import *


class SRRAWJOINTModel(BaseModel):
	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		return parser

	def __init__(self, opt):
		super(SRRAWJOINTModel, self).__init__(opt)

		self.opt = opt
		self.visual_names = ['dslr_warp', 'dslr_mask', 'data_out', 'AlignNet_out']

		# self.loss_names = ['AlignNet_L1', 'ISPNet_L1', 'ISPNet_SSIM', 'ISPNet_VGG', 'Total', 'GAN', 'Total_D'] # 
		# self.model_names = ['ISPNet', 'AlignNet', 'Discriminator'] #  will rename in subclasses
		# self.optimizer_names = ['ISPNet_optimizer_%s' % opt.optimizer,
		# 						'AlignNet_optimizer_%s' % opt.optimizer,
		# 						'Discriminator_optimizer_%s' % opt.optimizer]

		self.loss_names = ['AlignNet_L1', 'ISPNet_L1', 'ISPNet_SSIM', 'ISPNet_VGG', 'Total'] # 
		self.model_names = ['ISPNet', 'AlignNet'] #  will rename in subclasses
		self.optimizer_names = ['ISPNet_optimizer_%s' % opt.optimizer,
								'AlignNet_optimizer_%s' % opt.optimizer]

		isp = SRResNet(opt)
		self.netISPNet= N.init_net(isp, opt.init_type, opt.init_gain, opt.gpu_ids)

		align = ispjoint_model.AlignNet(opt)
		self.netAlignNet = N.init_net(align, opt.init_type, opt.init_gain, opt.gpu_ids)

		# align2 = AlignNet2(opt)
		# self.netAlignNet2 = N.init_net(align2, opt.init_type, opt.init_gain, opt.gpu_ids)

		pwcnet = pwc_net.PWCNET()
		self.netPWCNET = N.init_net(pwcnet, opt.init_type, opt.init_gain, opt.gpu_ids)
		self.set_requires_grad(self.netPWCNET, requires_grad=False)

		# discriminator = Discriminator()
		# self.netDiscriminator = N.init_net(discriminator, opt.init_type, opt.init_gain, opt.gpu_ids)

		if self.isTrain:		
			self.optimizer_ISPNet = optim.Adam(self.netISPNet.parameters(),
										  lr=opt.lr,
										  betas=(opt.beta1, opt.beta2),
										  weight_decay=opt.weight_decay)
			self.optimizer_AlignNet = optim.Adam(self.netAlignNet.parameters(),
										  lr=opt.lr,
										  betas=(opt.beta1, opt.beta2),
										  weight_decay=opt.weight_decay)
		
			# self.optimizer_D = optim.Adam(self.netDiscriminator.parameters(),
			# 						lr=opt.lr,
			# 						betas=(opt.beta1, opt.beta2),
			# 						weight_decay=opt.weight_decay)

			self.optimizers = [self.optimizer_ISPNet, self.optimizer_AlignNet] #  , self.optimizer_D

			self.criterionL1 = N.init_net(L.L1Loss(), gpu_ids=opt.gpu_ids)
			self.criterionSSIM = N.init_net(L.SSIMLoss(), gpu_ids=opt.gpu_ids)
			self.criterionVGG = N.init_net(L.VGGLoss(), gpu_ids=opt.gpu_ids)
			self.criterionGAN = N.init_net(L.GANLoss(), gpu_ids=opt.gpu_ids)

		self.isp_coord = {}

	def set_input(self, input):
		self.data_raw = input['raw'].to(self.device)
		self.data_raw_demosaic = input['raw_demosaic'].to(self.device)
		self.data_dslr = input['dslr'].to(self.device)
		self.align_coord = input['coord'].to(self.device)
		self.wb = input['wb'].to(self.device)
		self.image_paths = input['fname']

	def forward(self):
		N, C, H, W = self.data_raw_demosaic.size()
		down_dslr = nn.functional.interpolate(input=self.data_dslr, size=(H, W), 
											  mode='bilinear', align_corners=True)
		self.AlignNet_out = self.netAlignNet(self.data_raw_demosaic, down_dslr, self.align_coord)
		self.AlignNet_out = self.post_wb(self.AlignNet_out)
		# flow = self.get_flow(self.AlignNet_out, down_dslr, self.netPWCNET)
		
		# up_flow = nn.functional.interpolate(input=flow, size=(2*H, 2*W), 
		# 									mode='bilinear', align_corners=True) * 2
		
		# down_dslr_x2 = nn.functional.interpolate(input=self.data_dslr, size=(2*H, 2*W), 
		# 									  mode='bilinear', align_corners=True)
		# self.dslr_warp, self.dslr_mask = self.get_backwarp('/', down_dslr_x2, self.netPWCNET, up_flow)
		
		# self.down_dslr_mask = nn.functional.interpolate(input=self.dslr_mask, size=(H, W), 
		# 							mode='bilinear', align_corners=True)
		# self.down_dslr_warp = nn.functional.interpolate(input=self.dslr_warp, size=(H, W), 
		# 							mode='bilinear', align_corners=True)

		# self.dslr_warp, self.dslr_mask = self.get_backwarp('/', down_dslr, self.netPWCNET, flow)
		# self.down_dslr_warp = self.dslr_warp
		# self.down_dslr_mask = self.dslr_mask

		# N, C, H, W = self.data_raw_demosaic.size()
		# down_dslr_in = nn.functional.interpolate(input=self.data_dslr, size=(448, 448), 
		# 									  mode='bilinear', align_corners=True)
		# down_dslr = nn.functional.interpolate(input=self.data_dslr, size=(H, W), 
		# 									  mode='bilinear', align_corners=True)
		# AlignNet_in = nn.functional.interpolate(input=self.data_raw_demosaic, size=(448, 448), 
		# 									  mode='bilinear', align_corners=True)
		# self.AlignNet_out = self.netAlignNet(AlignNet_in, down_dslr_in)
		# self.AlignNet_out = self.post_wb(self.AlignNet_out)
		# self.AlignNet_out_down = nn.functional.interpolate(input=self.AlignNet_out, size=(H, W), 
		# 									  mode='bilinear', align_corners=True)
		# flow = self.get_flow(self.AlignNet_out_down, down_dslr, self.netPWCNET)

		# up_flow = nn.functional.interpolate(input=flow, size=(H*4, W*4), 
		# 									mode='bilinear', align_corners=True) * 4
		# self.dslr_warp, self.dslr_mask = self.get_backwarp('/', self.data_dslr, self.netPWCNET, up_flow)
		
		# self.down_dslr_mask = nn.functional.interpolate(input=self.dslr_mask, size=(H, W), 
		# 							mode='bilinear', align_corners=True)
		# self.down_dslr_warp = nn.functional.interpolate(input=self.dslr_warp, size=(H, W), 
		# 							mode='bilinear', align_corners=True)

		# N, C, H, W = self.data_raw.size()
		# index = str(self.data_raw.size()) + '_' + str(self.data_raw.device)
		# if index not in self.isp_coord:
		# 	isp_coord = get_coord(H=H//2, W=W//2)
		# 	isp_coord = np.expand_dims(isp_coord, axis=0)
		# 	isp_coord = np.tile(isp_coord, (N, 1, 1, 1))
		# 	# print(isp_coord.size())
		# 	self.isp_coord[index] = torch.from_numpy(isp_coord).to(self.data_raw.device)
		
		self.data_out = self.netISPNet(self.data_raw) #, self.isp_coord[index])
		self.data_out = self.post_wb(self.data_out)
		
		# self.data_out = nn.functional.interpolate(input=self.data_out, size=(4*H, 4*W), 
		# 								mode='bilinear', align_corners=True)

		if self.isTrain:
			self.AlignNet_out = self.AlignNet_out * self.down_dslr_mask
			self.data_out = self.data_out * self.dslr_mask
		else:
			# self.data_out = torch.pow(self.data_raw_demosaic, 1/2.2)

			flow = self.get_flow(nn.functional.interpolate(input=self.data_out, size=(2*H,2*W), mode='bilinear', align_corners=True), 
								 nn.functional.interpolate(input=self.data_dslr, size=(2*H,2*W), mode='bilinear', align_corners=True), self.netPWCNET)
			up_flow = nn.functional.interpolate(input=flow, size=(4*H, 4*W), mode='bilinear', align_corners=True) * 2
			self.dslr_warp, self.dslr_mask = self.get_backwarp(self.data_out, self.data_dslr, self.netPWCNET, up_flow) # down_dslr_x2 self.data_dslr down_dslr
			
			# self.dslr_warp =self.dslr_mask = self.data_dslr

	def backward(self):  
		self.loss_AlignNet_L1 = self.criterionL1(self.AlignNet_out, self.down_dslr_warp).mean()
		self.loss_ISPNet_L1 = self.criterionL1(self.data_out, self.dslr_warp).mean()
		self.loss_ISPNet_SSIM = 1 - self.criterionSSIM(self.data_out, self.dslr_warp).mean()
		self.loss_ISPNet_VGG = self.criterionVGG(self.data_out, self.dslr_warp).mean()
		self.loss_Total = self.loss_AlignNet_L1 + self.loss_ISPNet_L1 + self.loss_ISPNet_VGG + self.loss_ISPNet_SSIM * 0.15
		self.loss_Total.backward()

	# def backward_D(self):
	# 	predict_fake = self.netDiscriminator(self.data_out.detach())
	# 	lossGAN_fake = self.criterionGAN(predict_fake, False).mean()

	# 	predict_real = self.netDiscriminator(self.dslr_warp)
	# 	lossGAN_real = self.criterionGAN(predict_real, True).mean()

	# 	self.loss_Total_D = 0.5 * (lossGAN_fake + lossGAN_real)
	# 	self.loss_Total_D.backward()

	# def backward_G(self):
	# 	predict_fake = self.netDiscriminator(self.data_out)

	# 	self.loss_AlignNet_L1 = self.criterionL1(self.AlignNet_out, self.down_dslr_warp).mean()
	# 	self.loss_ISPNet_L1 = self.criterionL1(self.data_out, self.dslr_warp).mean()
	# 	self.loss_ISPNet_SSIM =  1 - self.criterionSSIM(self.data_out, self.dslr_warp).mean()
	# 	self.loss_ISPNet_VGG = self.criterionVGG(self.data_out, self.dslr_warp).mean()
	# 	# self.loss_Total = self.loss_AlignNet_L1 + self.loss_ISPNet_L1 + self.loss_ISPNet_VGG + self.loss_ISPNet_SSIM * 0.15
	# 	self.loss_GAN = self.criterionGAN(predict_fake, True).mean()
	# 	# print(self.loss_L1, self.loss_SSIM, self.loss_VGG, self.loss_GAN)

	# 	self.loss_Total = self.loss_AlignNet_L1 + self.loss_ISPNet_L1 + self.loss_ISPNet_VGG * 1 + self.loss_ISPNet_SSIM * 0.15 + self.loss_GAN * 0.001
	# 	self.loss_Total.backward()

	def optimize_parameters(self):
		# self.forward()
		# # update D
		# self.set_requires_grad(self.netDiscriminator, True)
		# self.optimizer_D.zero_grad()
		# self.backward_D()
		# self.optimizer_D.step()
		# # update G
		# self.set_requires_grad(self.netDiscriminator, False)
		# self.optimizer_ISPNet.zero_grad()
		# self.optimizer_AlignNet.zero_grad()
		# self.backward_G()
		# self.optimizer_ISPNet.step()
		# self.optimizer_AlignNet.step()

		self.forward()
		self.optimizer_ISPNet.zero_grad()
		self.optimizer_AlignNet.zero_grad()
		self.backward()
		self.optimizer_ISPNet.step()
		self.optimizer_AlignNet.step()
	
	def post_wb(self, img):
		# print(self.wb.size())
		img[:,0,...] *= torch.pow(self.wb[:,0,...], 1/2.2)
		img[:,1,...] *= torch.pow(self.wb[:,1,...], 1/2.2)
		img[:,2,...] *= torch.pow(self.wb[:,3,...], 1/2.2)
		return img      

# class AlignNet2(nn.Module):
# 	def __init__(self, opt):
# 		super(AlignNet2, self).__init__()
# 		self.opt = opt
# 		self.ch_1 = 32
# 		self.ch_2 = 64
# 		guide_input_channels = 8
# 		align_input_channels = 5
# 		self.alignnet_coord = opt.alignnet_coord

# 		if not self.alignnet_coord:
# 			guide_input_channels = 6
# 			align_input_channels = 3
		
# 		self.guide_net = N.seq(
# 			N.conv(guide_input_channels, self.ch_1, 7, stride=2, padding=0, mode='CR'),
# 			N.conv(self.ch_1, self.ch_1, kernel_size=3, stride=1, padding=1, mode='CRC'),
# 			nn.AdaptiveAvgPool2d(1),
# 			N.conv(self.ch_1, self.ch_2, 1, stride=1, padding=0, mode='C')
# 		)

# 		self.align_head = N.conv(align_input_channels, self.ch_2, 1, padding=0, mode='CR')

# 		self.align_base = N.seq(
# 			N.conv(self.ch_2, self.ch_2, kernel_size=1, stride=1, padding=0, mode='CRCRCR')
# 		)
# 		self.align_tail = N.seq(
# 			N.conv(self.ch_2, 3, 1, padding=0, mode='C')
# 		)

# 	def forward(self, demosaic_raw, dslr, coord=None):
# 		# demosaic_raw = torch.pow(demosaic_raw, 1/2.2)
# 		if self.alignnet_coord:
# 			guide_input = torch.cat((demosaic_raw, dslr, coord), 1)
# 			base_input = torch.cat((demosaic_raw, coord), 1)
# 		else:
# 			guide_input = torch.cat((demosaic_raw, dslr), 1)
# 			base_input = demosaic_raw

# 		guide = self.guide_net(guide_input)
	
# 		out = self.align_head(base_input)
# 		out = guide * out + out
# 		out = self.align_base(out)
# 		out = self.align_tail(out) + demosaic_raw
		
# 		return out

def pad(x, kernel_size=3, dilation=1):
	"""For stride = 2 or stride = 3"""
	pad_total = dilation * (kernel_size - 1)
	pad_beg = pad_total // 2
	pad_end = pad_total - pad_beg
	x_padded = torch.nn.functional.pad(x, pad=(pad_beg, pad_end, pad_beg, pad_end))
	return x_padded

class Residual_Block(nn.Module):
	def __init__(self):
		super(Residual_Block, self).__init__()
		self.conv_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=0, bias=False)
		self.BatchNorm = nn.InstanceNorm2d(64, affine=True) #  nn.BatchNorm2d(64, eps=0.001, affine=True) #
		self.Prelu = nn.PReLU(num_parameters=64)
		self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
		self.BatchNorm_1 = nn.InstanceNorm2d(64, affine=True) # nn.BatchNorm2d(64, eps=0.001, affine=True) # 

	def forward(self, x):
		identity_data = x
		x = pad(x, kernel_size=4)
		output = self.Prelu(self.BatchNorm(self.conv_1(x)))
		output = self.BatchNorm_1(self.conv_2(output))
		# output = self.Prelu(self.conv_1(x))
		# output = self.conv_2(output)
		output = output + identity_data
		return output 

class SRResNet(nn.Module):
	def __init__(self, opt):
		super(SRResNet, self).__init__()

		self.input_stage = nn.Sequential(
			nn.Conv2d(in_channels=4, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True),
			nn.PReLU(num_parameters=64)
		)
		
		for i in range(1, 16+1):
			setattr(self, 'resblock_%d'%i, Residual_Block())

		self.resblock_output = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm2d(64, affine=True)
			# nn.BatchNorm2d(64, eps=0.001, affine=True)
		)

		# self.deconv_stage1 = nn.Sequential(
		# 	nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
		# 	nn.PixelShuffle(upscale_factor=2),
		# 	nn.PReLU(num_parameters=64)
		# )

		# self.deconv_stage2 = nn.Sequential(
		# 	nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
		# 	nn.PixelShuffle(upscale_factor=2),
		# 	nn.PReLU(num_parameters=64)
		# )

		# self.deconv_stage3 = nn.Sequential(
		# 	nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
		# 	nn.PixelShuffle(upscale_factor=2),
		# 	nn.PReLU(num_parameters=64)
		# )

		# self.deconv_output_stage = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=True)

		self.deconv_stage1 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
			nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, 
							   padding=1, output_padding=(1,1), bias=True),
			nn.PReLU(num_parameters=256)
		)

		self.deconv_stage2 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
			nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2,
							   padding=1, output_padding=(1,1), bias=True),
			nn.PReLU(num_parameters=256)
		)

		self.deconv_stage3 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
			nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, 
							   padding=1, output_padding=(1,1), bias=True),
			nn.PReLU(num_parameters=256)
		)

		self.deconv_output_stage = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=9, stride=1, padding=4, bias=True)

		# self.load_state_dict(torch.load('./ckpt/cobi/net.pth')['state_dict'])

	def forward(self, x, coord=None):
		# identity_data = x
		input_stage = self.input_stage(x)
		stage1_output = input_stage # 64
		
		for i in range(1, 16+1):
			stage1_output = getattr(self, 'resblock_%d'%i)(stage1_output)

		resblock_output = self.resblock_output(stage1_output)
		resblock_output = resblock_output + input_stage
		
		out = self.deconv_stage1(resblock_output)
		out = self.deconv_stage2(out)
		out = self.deconv_stage3(out)

		out = self.deconv_output_stage(out)
		return out

class Discriminator(nn.Module): # LAST CHANGE CONV & CHANGE PLACE
	"""Defines a PatchGAN discriminator"""
	def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
		"""Construct a PatchGAN discriminator
		Parameters:
			input_nc (int)  -- the number of channels in input images
			ndf (int)       -- the number of filters in the last conv layer
			n_layers (int)  -- the number of conv layers in the discriminator
			norm_layer      -- normalization layer
		"""
		super(Discriminator, self).__init__()
		use_bias = False

		kw = 4
		padw = 1
		sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, n_layers):  # gradually increase the number of filters
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			]

		nf_mult_prev = nf_mult
		nf_mult = min(2 ** n_layers, 8)
		sequence += [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]

		sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
		self.model = nn.Sequential(*sequence)

	def forward(self, input):
		return self.model(input)

# class CSRNET(nn.Module):
# 	def __init__(self, opt):
# 		super(CSRNET, self).__init__()
# 		self.opt = opt
# 		self.ch_1 = 32
# 		self.ch_2 = 64
		
# 		stride = 2
# 		pad = 0

# 		self.guide_net = N.seq(
# 			N.conv(3*2, self.ch_1, 7, stride=2, padding=0, mode='CR'),
# 			N.conv(self.ch_1, self.ch_1, kernel_size=3, stride=1, padding=1, mode='CRC'),
# 			nn.AdaptiveAvgPool2d(1),
# 			N.conv(self.ch_1, self.ch_1*2, 1, stride=1, padding=0, mode='C'),
# 		)
		
# 		self.align_head = N.conv(3, self.ch_2, 1, padding=0, mode='CR')

# 		self.align_base = N.seq(
# 			N.conv(self.ch_2, self.ch_2, kernel_size=1, stride=1, padding=0, mode='CRCRCR')
# 		)
# 		self.align_tail = N.seq(
# 			N.conv(self.ch_2, 3, 1, padding=0, mode='C')
# 		)

# 	def forward(self, raw, dslr=None, index=None):
# 		raw = torch.pow(raw, 1/2.2)
# 		input = torch.cat((raw, dslr), 1)

# 		guide = self.guide_net(input)
# 		out = self.align_head(raw)
# 		out = guide * out + out

# 		out = self.align_base(out) 

# 		out = self.align_tail(out) + raw
# 		return out 


# class _Residual_Block(nn.Module):
# 	def __init__(self):
# 		super(_Residual_Block, self).__init__()
# 		self.conv_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=0, bias=False)
# 		self.BatchNorm = nn.BatchNorm2d(64, eps=0.001, affine=True, track_running_stats=False)
# 		self.Prelu = nn.PReLU(num_parameters=64)
# 		self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
# 		self.BatchNorm_1 = nn.BatchNorm2d(64, eps=0.001, affine=True, track_running_stats=False)

# 	def forward(self, x):
# 		identity_data = x
# 		x = pad(x, kernel_size=4)
# 		output = self.Prelu(self.BatchNorm(self.conv_1(x)))
# 		output = pad(output, kernel_size=3)
# 		output = self.BatchNorm_1(self.conv_2(output))
# 		output = output + identity_data
# 		return output 

# class SRResNet(nn.Module):
# 	def __init__(self, opt):
# 		super(SRResNet, self).__init__()

# 		self.input_stage = nn.Sequential(
# 			nn.Conv2d(in_channels=4, out_channels=64, kernel_size=9, stride=1, padding=0, bias=True),
# 			nn.PReLU(num_parameters=64)
# 		)
		
# 		for i in range(1, 16+1):
# 			setattr(self, 'resblock_%d'%i, _Residual_Block())

# 		self.resblock_output = nn.Sequential(
# 			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
# 			nn.BatchNorm2d(64, eps=0.001, affine=True, track_running_stats=False)
# 		)

# 		self.deconv_stage1 = nn.Sequential(
# 			nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
# 			nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=(1,1), bias=True),
# 			nn.PReLU(num_parameters=256)
# 		)

# 		self.deconv_stage2 = nn.Sequential(
# 			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
# 			nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=(1,1), bias=True),
# 			nn.PReLU(num_parameters=256)
# 		)

# 		self.deconv_stage3 = nn.Sequential(
# 			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
# 			nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=(1,1), bias=True),
# 			nn.PReLU(num_parameters=256)
# 		)

# 		self.deconv_output_stage = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=9, stride=1, padding=0, bias=True)

# 	def forward(self, x, coord=None):
# 		identity_data = x
# 		x = pad(x, kernel_size=9)
# 		input_stage = self.input_stage(x)
# 		stage1_output = input_stage # 64
		
# 		for i in range(1, 16+1):
# 			input_stage = getattr(self, 'resblock_%d'%i)(input_stage)

# 		input_stage = pad(input_stage, kernel_size=3)
# 		resblock_output = self.resblock_output(input_stage)
		
# 		resblock_output = resblock_output + stage1_output
		
# 		out = self.deconv_stage1(resblock_output)
# 		out = self.deconv_stage2(out)
# 		out = self.deconv_stage3(out)

# 		out = pad(out, kernel_size=9)
# 		out = self.deconv_output_stage(out)
# 		return out