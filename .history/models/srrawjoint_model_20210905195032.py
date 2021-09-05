import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
from . import losses as L
from pwc import pwc_net
from .zrrjoint_model import GCMModel
from torch.nn.functional import interpolate 

class SRRAWJOINTModel(BaseModel):
	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		return parser

	def __init__(self, opt):
		super(SRRAWJOINTModel, self).__init__(opt)

		self.opt = opt
		self.loss_names = ['GCMModel_L1', 'SRResNet_L1', 'SRResNet_SSIM', 'SRResNet_VGG', 'Total']
		self.visual_names = ['dslr_warp', 'dslr_mask', 'data_out', 'GCMModel_out'] 
		self.model_names = ['SRResNet', 'GCMModel'] 
		self.optimizer_names = ['SRResNet_optimizer_%s' % opt.optimizer,
								'GCMModel_optimizer_%s' % opt.optimizer]

		srresnet = SRResNet(opt)
		self.netSRResNet= N.init_net(srresnet, opt.init_type, opt.init_gain, opt.gpu_ids)

		gcm = GCMModel(opt)
		self.netGCMModel = N.init_net(gcm, opt.init_type, opt.init_gain, opt.gpu_ids)

		pwcnet = pwc_net.PWCNET()
		self.netPWCNET = N.init_net(pwcnet, opt.init_type, opt.init_gain, opt.gpu_ids)
		self.set_requires_grad(self.netPWCNET, requires_grad=False)

		if self.isTrain:		
			self.optimizer_SRResNet = optim.Adam(self.netSRResNet.parameters(),
										  lr=opt.lr,
										  betas=(opt.beta1, opt.beta2),
										  weight_decay=opt.weight_decay)
			self.optimizer_GCMModel = optim.Adam(self.netGCMModel.parameters(),
										  lr=opt.lr,
										  betas=(opt.beta1, opt.beta2),
										  weight_decay=opt.weight_decay)

			self.optimizers = [self.optimizer_SRResNet, self.optimizer_GCMModel]

			self.criterionL1 = N.init_net(L.L1Loss(), gpu_ids=opt.gpu_ids)
			self.criterionSSIM = N.init_net(L.SSIMLoss(), gpu_ids=opt.gpu_ids)
			self.criterionVGG = N.init_net(L.VGGLoss(), gpu_ids=opt.gpu_ids)
			self.criterionGAN = N.init_net(L.GANLoss(), gpu_ids=opt.gpu_ids)

	def set_input(self, input):
		self.data_raw = input['raw'].to(self.device)
		self.data_raw_demosaic = input['raw_demosaic'].to(self.device)
		self.data_dslr = input['dslr'].to(self.device)
		self.gcm_coord = input['coord'].to(self.device)
		self.wb = input['wb'].to(self.device)
		self.image_paths = input['fname']

	def forward(self):
		N, C, H, W = self.data_raw_demosaic.size()
		down_dslr = interpolate(input=self.data_dslr, size=(H, W), 
											  mode='bilinear', align_corners=True)
		self.GCMModel_out = self.netGCMModel(self.data_raw_demosaic, down_dslr, self.gcm_coord)
		self.GCMModel_out = self.post_wb(self.GCMModel_out)
		
		self.data_out = self.netSRResNet(self.data_raw) #, self.isp_coord[index])
		self.data_out = self.post_wb(self.data_out)

		flow = self.get_flow(self.GCMModel_out, down_dslr, self.netPWCNET)
		up_flow = interpolate(input=flow, size=(H*4, W*4), 
							  mode='bilinear', align_corners=True) * 4.
		self.dslr_warp, self.dslr_mask = self.get_backwarp('/', self.data_dslr, self.netPWCNET, up_flow)
		
		self.down_dslr_warp = interpolate(input=self.dslr_warp, size=(H, W), 
									      mode='bilinear', align_corners=True)
		self.down_dslr_mask = interpolate(input=self.dslr_mask, size=(H, W), 
									    mode='bilinear', align_corners=True)

		if self.opt.scale == 1:
			self.dslr_warp = self.down_dslr_warp
			self.dslr_mask = self.down_dslr_mask
		elif self.opt.scale == 2:
			self.dslr_warp = interpolate(input=self.dslr_warp, size=(2*H, 2*W), 
								         mode='bilinear', align_corners=True)
			self.dslr_mask = interpolate(input=self.dslr_mask, size=(2*H, 2*W), 
							             mode='bilinear', align_corners=True)
		
		if self.isTrain:
			self.GCMModel_out = self.GCMModel_out * self.down_dslr_mask
			self.data_out = self.data_out * self.dslr_mask
		else:
			self.data_out = interpolate(input=self.data_out, size=(4*H, 4*W), 
											          mode='bilinear', align_corners=True)
			flow = self.get_flow(interpolate(input=self.data_out, size=(2*H,2*W), mode='bilinear', align_corners=True), 
								 interpolate(input=self.data_dslr, size=(2*H,2*W), mode='bilinear', align_corners=True), self.netPWCNET)
			up_flow = interpolate(input=flow, size=(4*H, 4*W), mode='bilinear', align_corners=True) * 2
			self.dslr_warp, self.dslr_mask = self.get_backwarp(self.data_out, self.data_dslr, self.netPWCNET, up_flow) # down_dslr_x2 self.data_dslr down_dslr
			
			# self.dslr_warp =self.dslr_mask = self.data_dslr

	def backward(self):  
		self.loss_GCMModel_L1 = self.criterionL1(self.GCMModel_out, self.down_dslr_warp).mean()
		self.loss_SRResNet_L1 = self.criterionL1(self.data_out, self.dslr_warp).mean()
		self.loss_SRResNet_SSIM = 1 - self.criterionSSIM(self.data_out, self.dslr_warp).mean()
		self.loss_SRResNet_VGG = self.criterionVGG(self.data_out, self.dslr_warp).mean()
		self.loss_Total = self.loss_GCMModel_L1 + self.loss_SRResNet_L1 + self.loss_SRResNet_VGG + self.loss_SRResNet_SSIM * 0.15
		self.loss_Total.backward()

	def optimize_parameters(self):
		self.forward()
		self.optimizer_SRResNet.zero_grad()
		self.optimizer_GCMModel.zero_grad()
		self.backward()
		self.optimizer_SRResNet.step()
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
