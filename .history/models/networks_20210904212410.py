import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from collections import OrderedDict


def get_scheduler(optimizer, opt):
	if opt.lr_policy == 'linear':
		def lambda_rule(epoch):
			return 1 - max(0, epoch-opt.niter) / max(1, float(opt.niter_decay))
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer,
										step_size=opt.lr_decay_iters,
										gamma=0.5)
	elif opt.lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
												   mode='min',
												   factor=0.2,
												   threshold=0.01,
												   patience=5)
	elif opt.lr_policy == 'cosine':
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
												   T_max=opt.niter,
												   eta_min=0)
	else:
		return NotImplementedError('lr [%s] is not implemented', opt.lr_policy)
	return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
	def init_func(m):  # define the initialization function
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 \
				or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=init_gain)
			elif init_type == 'uniform':
				init.uniform_(m.weight.data, b=init_gain)
			else:
				raise NotImplementedError('[%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, init_gain)
			init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='default', init_gain=0.02, gpu_ids=[]):
	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
	if init_type != 'default' and init_type is not None:
		init_weights(net, init_type, init_gain=init_gain)
	return net


'''
# ===================================
# Advanced nn.Sequential
# reform nn.Sequentials and nn.Modules
# to a single nn.Sequential
# ===================================
'''

def seq(*args):
	if len(args) == 1:
		args = args[0]
	if isinstance(args, nn.Module):
		return args
	modules = OrderedDict()
	if isinstance(args, OrderedDict):
		for k, v in args.items():
			modules[k] = seq(v)
		return nn.Sequential(modules)
	assert isinstance(args, (list, tuple))
	return nn.Sequential(*[seq(i) for i in args])

'''
# ===================================
# Useful blocks
# --------------------------------
# conv (+ normaliation + relu)
# concat
# sum
# resblock (ResBlock)
# resdenseblock (ResidualDenseBlock_5C)
# resinresdenseblock (RRDB)
# ===================================
'''

# -------------------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# -------------------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
		 output_padding=0, dilation=1, groups=1, bias=True,
		 padding_mode='zeros', mode='CBR'):
	L = []
	for t in mode:
		if t == 'C':
			L.append(nn.Conv2d(in_channels=in_channels,
							   out_channels=out_channels,
							   kernel_size=kernel_size,
							   stride=stride,
							   padding=padding,
							   dilation=dilation,
							   groups=groups,
							   bias=bias,
							   padding_mode=padding_mode))
		elif t == 'X':
			assert in_channels == out_channels
			L.append(nn.Conv2d(in_channels=in_channels,
							   out_channels=out_channels,
							   kernel_size=kernel_size,
							   stride=stride,
							   padding=padding,
							   dilation=dilation,
							   groups=in_channels,
							   bias=bias,
							   padding_mode=padding_mode))
		elif t == 'T':
			L.append(nn.ConvTranspose2d(in_channels=in_channels,
										out_channels=out_channels,
										kernel_size=kernel_size,
										stride=stride,
										padding=padding,
										output_padding=output_padding,
										groups=groups,
										bias=bias,
										dilation=dilation,
										padding_mode=padding_mode))
		elif t == 'B':
			L.append(nn.BatchNorm2d(out_channels))
		elif t == 'I':
			L.append(nn.InstanceNorm2d(out_channels, affine=True))
		elif t == 'i':
			L.append(nn.InstanceNorm2d(out_channels))
		elif t == 'R':
			L.append(nn.ReLU(inplace=True))
		elif t == 'r':
			L.append(nn.ReLU(inplace=False))
		elif t == 'S':
			L.append(nn.Sigmoid())
		elif t == 'P':
			L.append(nn.PReLU())
		elif t == 'L':
			L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=True))
		elif t == 'l':
			L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=False))
		elif t == '2':
			L.append(nn.PixelShuffle(upscale_factor=2))
		elif t == '3':
			L.append(nn.PixelShuffle(upscale_factor=3))
		elif t == '4':
			L.append(nn.PixelShuffle(upscale_factor=4))
		elif t == 'U':
			L.append(nn.Upsample(scale_factor=2, mode='nearest'))
		elif t == 'u':
			L.append(nn.Upsample(scale_factor=3, mode='nearest'))
		elif t == 'M':
			L.append(nn.MaxPool2d(kernel_size=kernel_size,
								  stride=stride,
								  padding=0))
		elif t == 'A':
			L.append(nn.AvgPool2d(kernel_size=kernel_size,
								  stride=stride,
								  padding=0))
		else:
			raise NotImplementedError('Undefined type: '.format(t))
	return seq(*L)


class DWTForward(nn.Conv2d):
    def __init__(self, in_channels=64):
        super(DWTForward, self).__init__(in_channels, in_channels*4, 2, 2,
                                  groups=in_channels, bias=False)
        weight = torch.tensor([[[[0.5,  0.5], [ 0.5,  0.5]]],
                               [[[0.5,  0.5], [-0.5, -0.5]]],
                               [[[0.5, -0.5], [ 0.5, -0.5]]],
                               [[[0.5, -0.5], [-0.5,  0.5]]]],
                              dtype=torch.get_default_dtype()
                             ).repeat(in_channels, 1, 1, 1)# / 2
        self.weight.data.copy_(weight)
        self.requires_grad_(False)


class DWTInverse(nn.ConvTranspose2d):
    def __init__(self, in_channels=64):
        super(DWTInverse, self).__init__(in_channels, in_channels//4, 2, 2,
                                  groups=in_channels//4, bias=False)
        weight = torch.tensor([[[[0.5,  0.5], [ 0.5,  0.5]]],
                               [[[0.5,  0.5], [-0.5, -0.5]]],
                               [[[0.5, -0.5], [ 0.5, -0.5]]],
                               [[[0.5, -0.5], [-0.5,  0.5]]]],
                              dtype=torch.get_default_dtype()
                             ).repeat(in_channels//4, 1, 1, 1)# * 2
        self.weight.data.copy_(weight)
        self.requires_grad_(False)


# -------------------------------------------------------
# Channel Attention (CA) Layer
# -------------------------------------------------------
class CALayer(nn.Module):
	def __init__(self, channel=64, reduction=16):
		super(CALayer, self).__init__()

		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.conv_du = nn.Sequential(
			nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
			nn.Sigmoid()
		)

	def forward(self, x):
		y = self.avg_pool(x)
		y = self.conv_du(y)
		return x * y 


# -------------------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# -------------------------------------------------------
class ResBlock(nn.Module):
	def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
				 padding=1, bias=True, mode='CRC'):
		super(ResBlock, self).__init__()

		assert in_channels == out_channels
		if mode[0] in ['R','L']:
			mode = mode[0].lower() + mode[1:]

		self.res = conv(in_channels, out_channels, kernel_size,
						stride, padding=padding, bias=bias, mode=mode)

	def forward(self, x):
		res = self.res(x)
		return x + res


# -------------------------------------------------------
# Residual Channel Attention Block (RCAB)
# -------------------------------------------------------
class RCABlock(nn.Module):
	def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
				 padding=1, bias=True, mode='CRC', reduction=16):
		super(RCABlock, self).__init__()
		assert in_channels == out_channels
		if mode[0] in ['R','L']:
			mode = mode[0].lower() + mode[1:]

		self.res = conv(in_channels, out_channels, kernel_size,
						stride, padding, bias=bias, mode=mode)
		self.ca = CALayer(out_channels, reduction)

	def forward(self, x):
		res = self.res(x)
		res = self.ca(res) 
		return res + x


# -------------------------------------------------------
# Residual Channel Attention Group (RG)
# -------------------------------------------------------
class RCAGroup(nn.Module):
	def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
				 padding=1, bias=True, mode='CRC', reduction=16, nb=12):
		super(RCAGroup, self).__init__()
		assert in_channels == out_channels
		if mode[0] in ['R','L']:
			mode = mode[0].lower() + mode[1:]

		RG = [RCABlock(in_channels, out_channels, kernel_size, stride, padding,
					   bias, mode, reduction) for _ in range(nb)]
		# RG = [ResBlock(in_channels, out_channels, kernel_size, stride, padding,
		#                bias, mode) for _ in range(nb)]
		RG.append(conv(out_channels, out_channels, mode='C'))

		self.rg = nn.Sequential(*RG)

	def forward(self, x):
		res = self.rg(x)
		return res + x

