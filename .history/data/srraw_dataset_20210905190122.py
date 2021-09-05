import numpy as np
import os
from data.base_dataset import BaseDataset
from .imlib import imlib
from util.util import *
import colour_demosaicing
import glob


class SRRAWDataset(BaseDataset):
	def __init__(self, opt, split='train', dataset_name='SRRAW'):
		super(SRRAWDataset, self).__init__(opt, split, dataset_name)
		# if self.root == '':
		# 	rootlist = ['/Data/dataset/SR-RAW']
		# 	for root in rootlist:
		# 		if os.path.isdir(root):
		# 			self.root = root
		# 			break

		self.batch_size = opt.batch_size

		self.patch_size = 512
		self.scale = 4
		self.mode = opt.mode  # RGB, Y or L=
		self.imio = imlib(self.mode, lib=opt.imlib)

		if split == 'train':
			self.raw_dirs, self.dslr_dirs, self.names = self._get_image_dir(self.root, 'train')
			self.out_wbs = self._get_wb(self.raw_dirs, 'train')
			self._getitem = self._getitem_train
			self.len_data = 1000 * self.batch_size  # len(self.names)
		elif split == 'val':
			self.raw_dirs, self.dslr_dirs, self.names = self._get_image_dir(self.root, 'val')
			self.out_wbs = self._get_wb(self.raw_dirs, 'val')
			self._getitem = self._getitem_val
			self.len_data = len(self.names)
		elif split == 'test':
			self.raw_dirs, self.dslr_dirs, self.names = self._get_image_dir(self.root, 'test', flag='00006')
			self.out_wbs = self._get_wb(self.raw_dirs, 'test')
			self._getitem = self._getitem_test
			self.len_data = len(self.names)
		else:
			raise ValueError

		self.coord = get_coord(H=512//4, W=512//4, x=1, y=1)
		self.raw_images = [0] * len(self.names)
		self.dslr_images = [0] * len(self.names)
		read_images(self)

	def __getitem__(self, index):
		return self._getitem(index)

	def __len__(self):
		return self.len_data

	def _getitem_train(self, idx):
		idx = idx % len(self.names)

		raw_image = np.maximum(np.float32(self.raw_images[idx]), 0)
		dslr_image =  np.float32(self.dslr_images[idx]) / 255.0

		cropped_raw, cropped_rgb, coord_index = self._crop_pair(raw_image, dslr_image, croph=self.patch_size, 
		    cropw=self.patch_size, ratio=self.scale, type='random') 
		raw, raw_demosaic = self._get_raw_demosaic(cropped_raw)

		raw, cropped_rgb, raw_demosaic, coord = augment(raw, cropped_rgb, raw_demosaic, self.coord)

		return {'raw': raw,
				'raw_demosaic': raw_demosaic,
				'dslr': cropped_rgb,
				'coord': coord,
				'wb': self.out_wbs[idx],
				'fname': self.names[idx]}

	def _getitem_val(self, idx):
		raw_image = np.maximum(np.float32(self.raw_images[idx]), 0)
		dslr_image = np.float32(self.dslr_images[idx]) / 255.0

		cropped_raw, cropped_rgb, coord_index = self._crop_pair(raw_image, dslr_image, croph=self.patch_size, 
		    cropw=self.patch_size, ratio=self.scale, type='fixed') 

		raw, raw_demosaic = self._get_raw_demosaic(cropped_raw)

		return {'raw': raw,
				'raw_demosaic': raw_demosaic,
				'dslr': cropped_rgb,
				'coord': self.coord,
				'wb': self.out_wbs[idx],
				'fname': self.names[idx]}

	def _getitem_test(self, idx):
		raw_image = np.maximum(np.float32(self.raw_images[idx]), 0)
		dslr_image = np.float32(self.dslr_images[idx]) / 255.0

		raw, raw_demosaic = self._get_raw_demosaic(raw_image)
		coord = get_coord(H=raw_demosaic.shape[1], W=raw_demosaic.shape[2], x=1, y=1)

		return {'raw': raw,
				'raw_demosaic': raw_demosaic,
				'dslr': dslr_image,
				'coord': coord,
				'wb': self.out_wbs[idx],
				'fname': self.names[idx]}

		# cropped_raw, cropped_rgb, coord_index = self._crop_pair(raw_image, dslr_image, 
		#     croph=1600, cropw=2400, ratio=self.scale, type='fixed') #  croph=1024, cropw=1536

		# raw, raw_demosaic = self._get_raw_demosaic(cropped_raw)
		# coord = get_coord(H=raw_demosaic.shape[1], W=raw_demosaic.shape[2], x=1, y=1)

		# return {'raw': raw,
		# 		'raw_demosaic': raw_demosaic,
		# 		'dslr': cropped_rgb,
		# 		'coord': coord,
		# 		'wb': self.out_wbs[idx],
		# 		'fname': self.names[idx]}
	
	def _get_image_dir(self, dataroot, type='train', flag='*'):
		raw_dirs = []
		dslr_dirs = [] # input_x4_raw target_x4_rgb

		image_names = sorted(glob.glob(dataroot + '/' + type + '/' + '00*'))
		for image_name in image_names:
			raw_dir = sorted(glob.glob(image_name + '/input_x4_raw/' + flag + '.npy'))
			dslr_dir = sorted(glob.glob(image_name + '/input_x4_raw/' + flag + '.png'))
			raw_dirs.extend(raw_dir)
			dslr_dirs.extend(dslr_dir)
		
		names = []
		for raw_name in raw_dirs:
			raw_name_sp = raw_name.split('/')
			names.append(raw_name_sp[-3] + '-' + raw_name_sp[-1])
		
		if type == 'val':
			return raw_dirs[0:15], dslr_dirs[0:15], names[0:15]
		# return raw_dirs[0:16], dslr_dirs[0:16], names[0:16]
		return raw_dirs, dslr_dirs, names
	
	def _get_raw_demosaic(self, bayer):
		raw = np.ascontiguousarray(bayer.transpose((2, 0, 1)))
		
		H = bayer.shape[0]
		W = bayer.shape[1]
		newH = int(H*2)
		newW = int(W*2)
		
		bayer_back = np.zeros((newH, newW))
		bayer_back[0:newH:2, 0:newW:2] = bayer[...,0]
		bayer_back[0:newH:2, 1:newW:2] = bayer[...,1]
		bayer_back[1:newH:2, 1:newW:2] = bayer[...,2]
		bayer_back[1:newH:2, 0:newW:2] = bayer[...,3]
		
		raw_demosaic = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(bayer_back, pattern='RGGB')
		raw_demosaic = np.ascontiguousarray(raw_demosaic.astype(np.float32).transpose((2, 0, 1)))
		return raw, raw_demosaic

	def _get_wb(self, raw_dirs):
		out_wbs = []
		for raw_name in raw_dirs:
			raw_name_sp = raw_name.split('/input_x4_raw/')
			key = raw_name_sp[1][:-4]		
			wb_txt = raw_name_sp[0] + '/wb.txt'
			
			out_wb = read_wb(wb_txt, key=key+":")
			out_wb = np.expand_dims(out_wb, axis=0)
			out_wb = out_wb.astype(np.float32).transpose((2, 0, 1))
			out_wbs.append(out_wb)
		return np.array(out_wbs).astype(np.float32)

	def _crop_pair(self, raw, image, croph, cropw, tol=0, raw_tol=0, ratio=4, 
	               type='random', fixx=0.5, fixy=0.5):
		is_pad_h = False
		is_pad_w = False
		# if type == 'central':
		# 	rand_p = rand_gen.rvs(2)
		if type == 'random':
			rand_p = np.random.rand(2)
		elif type == 'fixed':
			rand_p = [fixx,fixy]
		
		height_raw, width_raw = raw.shape[:2]
		height_rgb, width_rgb = image.shape[1:]

		if croph > height_raw * 2 * ratio or cropw > width_raw * 2 * ratio:
			print("Image too small to have the specified crop sizes.")
			return None, None
		croph_rgb = croph + tol * 2
		cropw_rgb = cropw + tol * 2 
		croph_raw = int(croph/(ratio*2)) + raw_tol * 2  
		# add a small offset to deal with boudary case
		cropw_raw = int(cropw/(ratio*2)) + raw_tol * 2  
		# add a small offset to deal with boudary case
		
		if croph_rgb > height_rgb:
			sx_rgb = 0
			sx_raw = int(tol/2.)
			is_pad_h = True
			pad_h1_rgb = int((croph_rgb-height_rgb)/2)
			pad_h2_rgb = int(croph_rgb-height_rgb-pad_h1_rgb)
			pad_h1_raw = int(np.ceil(pad_h1_rgb/(2*ratio)))
			pad_h2_raw = int(np.ceil(pad_h2_rgb/(2*ratio)))
		else:
			sx_rgb = int((height_rgb - croph_rgb) * rand_p[0])
			sx_raw = max(0, int((sx_rgb + tol)/(2*ratio)) - raw_tol) 
			# add a small offset to deal with boudary case
		
		if cropw_rgb > width_rgb:
			sy_rgb = 0 
			sy_raw = int(tol/2.)
			is_pad_w = True
			pad_w1_rgb = int((cropw_rgb-width_rgb)/2)
			pad_w2_rgb = int(cropw_rgb-width_rgb-pad_w1_rgb)
			pad_w1_raw = int(np.ceil(pad_w1_rgb/(2*ratio)))
			pad_w2_raw = int(np.ceil(pad_w2_rgb/(2*ratio)))
		else:
			sy_rgb = int((width_rgb - cropw_rgb) * rand_p[1])
			sy_raw = max(0, int((sy_rgb + tol)/(2*ratio)) - raw_tol)

		raw_cropped = raw
		rgb_cropped = image
		if is_pad_h:
			# print("Pad h with:", (pad_h1_rgb, pad_h2_rgb),(pad_h1_raw, pad_h2_raw))
			rgb_cropped = np.pad(image, pad_width=((pad_h1_rgb, pad_h2_rgb),(0, 0),(0,0)),
				mode='constant', constant_values=0)
			raw_cropped = np.pad(raw, pad_width=((pad_h1_raw, pad_h2_raw),(0, 0),(0,0)),
				mode='constant', constant_values=0)
		if is_pad_w:
			# print("Pad w with:", (pad_w1_rgb, pad_w2_rgb),(pad_w1_raw, pad_w2_raw))
			rgb_cropped = np.pad(image, pad_width=((0, 0),(pad_w1_rgb, pad_w2_rgb),(0,0)),
				mode='constant', constant_values=0)
			raw_cropped = np.pad(raw, pad_width=((0, 0),(pad_w1_raw, pad_w2_raw),(0,0)),
				mode='constant', constant_values=0)
		
		raw_cropped = raw_cropped[sx_raw:sx_raw+croph_raw, sy_raw:sy_raw+cropw_raw,...]
		rgb_cropped = rgb_cropped[..., sx_rgb:sx_rgb+croph_rgb, sy_rgb:sy_rgb+cropw_rgb]
		return  raw_cropped, rgb_cropped, \
		        [sx_raw, sx_raw+croph_raw, sy_raw, sy_raw+cropw_raw]


def iter_obj(num, objs):
	for i in range(num):
		yield (i, objs)

def imreader(arg):
	i, obj = arg
	for _ in range(3):
		try:
			obj.raw_images[i] = np.load(obj.raw_dirs[i])
			obj.dslr_images[i] = obj.imio.read(obj.dslr_dirs[i])
			failed = False
			break
		except:
			failed = True
	if failed: print('%s fails!' % obj.names[i])

def read_images(obj):
	# may use `from multiprocessing import Pool` instead, but less efficient and
	# NOTE: `multiprocessing.Pool` will duplicate given object for each process.
	from multiprocessing.dummy import Pool
	from tqdm import tqdm
	print('Starting to load images via multiple imreaders')
	pool = Pool() # use all threads by default
	for _ in tqdm(pool.imap(imreader, iter_obj(len(obj.names), obj)), total=len(obj.names)):
		pass
	pool.close()
	pool.join()

# if __name__ == '__main__':
	# import os
	# map = {'00001.png':'00005.png',
	#        '00002.png':'00006.png',
	# 	     '00003.png':'00007.png',}
	# raw_dirs = sorted(glob.glob('/Data/dataset/SRRAW/test/*'))
	# for raw_dir in raw_dirs:
	# 	names = sorted(glob.glob(raw_dir + '/target_x4_rgb/*.png'))
	# 	for name in names:
	# 		pre_name = name[-9:]
	# 		post_name = map[pre_name]
	# 		os.rename(name, name[:-9] + post_name)