import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from . import losses as L
from pwc import pwc_net
from . import ispjoint_model
# from util.util import *


class SRRAWJOINTModel(BaseModel):
	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		return parser

	def __init__(self, opt):
		super(SRRAWJOINTModel, self).__init__(opt)

		self.opt = opt
		self.visual_names = ['dslr_warp', 'dslr_mask', 'data_out', 'GCMModel_out']

		self.loss_names = ['GCMModel_L1', 'LiteISPNet_L1', 'LiteISPNet_SSIM', 'LiteISPNet_VGG', 'Total']
		self.visual_names = ['dslr_warp', 'dslr_mask', 'data_out', 'GCMModel_out'] 
		self.model_names = ['LiteISPNet', 'GCMModel'] 
		self.optimizer_names = ['LiteISPNet_optimizer_%s' % opt.optimizer,
								'GCMModel_optimizer_%s' % opt.optimizer]
								
		# self.loss_names = ['GCMModel_L1', 'ISPNet_L1', 'ISPNet_SSIM', 'ISPNet_VGG', 'Total', 'GAN', 'Total_D'] # 
		# self.model_names = ['ISPNet', 'GCMModel', 'Discriminator'] #  will rename in subclasses
		# self.optimizer_names = ['ISPNet_optimizer_%s' % opt.optimizer,
		# 						'GCMModel_optimizer_%s' % opt.optimizer,
		# 						'Discriminator_optimizer_%s' % opt.optimizer]

		self.loss_names = ['GCMModel_L1', 'ISPNet_L1', 'ISPNet_SSIM', 'ISPNet_VGG', 'Total'] # 
		self.model_names = ['ISPNet', 'GCMModel'] #  will rename in subclasses
		self.optimizer_names = ['ISPNet_optimizer_%s' % opt.optimizer,
								'GCMModel_optimizer_%s' % opt.optimizer]

		isp = SRResNet(opt)
		self.netISPNet= N.init_net(isp, opt.init_type, opt.init_gain, opt.gpu_ids)

		align = ispjoint_model.GCMModel(opt)
		self.netGCMModel = N.init_net(align, opt.init_type, opt.init_gain, opt.gpu_ids)

		# align2 = GCMModel2(opt)
		# self.netGCMModel2 = N.init_net(align2, opt.init_type, opt.init_gain, opt.gpu_ids)

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
			self.optimizer_GCMModel = optim.Adam(self.netGCMModel.parameters(),
										  lr=opt.lr,
										  betas=(opt.beta1, opt.beta2),
										  weight_decay=opt.weight_decay)
		
			# self.optimizer_D = optim.Adam(self.netDiscriminator.parameters(),
			# 						lr=opt.lr,
			# 						betas=(opt.beta1, opt.beta2),
			# 						weight_decay=opt.weight_decay)

			self.optimizers = [self.optimizer_ISPNet, self.optimizer_GCMModel] #  , self.optimizer_D

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
		self.GCMModel_out = self.netGCMModel(self.data_raw_demosaic, down_dslr, self.align_coord)
		self.GCMModel_out = self.post_wb(self.GCMModel_out)
		# flow = self.get_flow(self.GCMModel_out, down_dslr, self.netPWCNET)
		
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
		# GCMModel_in = nn.functional.interpolate(input=self.data_raw_demosaic, size=(448, 448), 
		# 									  mode='bilinear', align_corners=True)
		# self.GCMModel_out = self.netGCMModel(GCMModel_in, down_dslr_in)
		# self.GCMModel_out = self.post_wb(self.GCMModel_out)
		# self.GCMModel_out_down = nn.functional.interpolate(input=self.GCMModel_out, size=(H, W), 
		# 									  mode='bilinear', align_corners=True)
		# flow = self.get_flow(self.GCMModel_out_down, down_dslr, self.netPWCNET)

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
			self.GCMModel_out = self.GCMModel_out * self.down_dslr_mask
			self.data_out = self.data_out * self.dslr_mask
		else:
			# self.data_out = torch.pow(self.data_raw_demosaic, 1/2.2)

			flow = self.get_flow(nn.functional.interpolate(input=self.data_out, size=(2*H,2*W), mode='bilinear', align_corners=True), 
								 nn.functional.interpolate(input=self.data_dslr, size=(2*H,2*W), mode='bilinear', align_corners=True), self.netPWCNET)
			up_flow = nn.functional.interpolate(input=flow, size=(4*H, 4*W), mode='bilinear', align_corners=True) * 2
			self.dslr_warp, self.dslr_mask = self.get_backwarp(self.data_out, self.data_dslr, self.netPWCNET, up_flow) # down_dslr_x2 self.data_dslr down_dslr
			
			# self.dslr_warp =self.dslr_mask = self.data_dslr

	def backward(self):  
		self.loss_GCMModel_L1 = self.criterionL1(self.GCMModel_out, self.down_dslr_warp).mean()
		self.loss_ISPNet_L1 = self.criterionL1(self.data_out, self.dslr_warp).mean()
		self.loss_ISPNet_SSIM = 1 - self.criterionSSIM(self.data_out, self.dslr_warp).mean()
		self.loss_ISPNet_VGG = self.criterionVGG(self.data_out, self.dslr_warp).mean()
		self.loss_Total = self.loss_GCMModel_L1 + self.loss_ISPNet_L1 + self.loss_ISPNet_VGG + self.loss_ISPNet_SSIM * 0.15
		self.loss_Total.backward()

	def optimize_parameters(self):
		self.forward()
		self.optimizer_ISPNet.zero_grad()
		self.optimizer_GCMModel.zero_grad()
		self.backward()
		self.optimizer_ISPNet.step()
		self.optimizer_GCMModel.step()
	
	def post_wb(self, img):
		img[:,0,...] *= torch.pow(self.wb[:,0,...], 1/2.2)
		img[:,1,...] *= torch.pow(self.wb[:,1,...], 1/2.2)
		img[:,2,...] *= torch.pow(self.wb[:,3,...], 1/2.2)
		return img      


# The network structure is somewhat different from the general SRResNet, 
# which is to be consistent with https://github.com/ceciliavision/zoom-learn-zoom/blob/master/net.py
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
		self.BatchNorm = nn.InstanceNorm2d(64, affine=True) 
		self.Prelu = nn.PReLU(num_parameters=64)
		self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
		self.BatchNorm_1 = nn.InstanceNorm2d(64, affine=True) 

	def forward(self, x):
		identity_data = x
		x = pad(x, kernel_size=4)
		output = self.Prelu(self.BatchNorm(self.conv_1(x)))
		output = self.BatchNorm_1(self.conv_2(output))
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

	def forward(self, x):
		input_stage = self.input_stage(x)
		stage1_output = input_stage 
		
		for i in range(1, 16+1):
			stage1_output = getattr(self, 'resblock_%d'%i)(stage1_output)

		resblock_output = self.resblock_output(stage1_output)
		resblock_output = resblock_output + input_stage
		
		out = self.deconv_stage1(resblock_output)
		out = self.deconv_stage2(out)
		out = self.deconv_stage3(out)

		out = self.deconv_output_stage(out)
		return out
