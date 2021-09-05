import numpy as np
import os
from data.base_dataset import BaseDataset
from .imlib import imlib
from util.util import *


# 
class ZRRDataset(BaseDataset):
	"Zurich RAW to RGB (ZRR) dataset"
	def __init__(self, opt, split='train', dataset_name='ZRR'):
		super(ZRRDataset, self).__init__(opt, split, dataset_name)

		# if self.root == '':
		# 	rootlist = ['/data/dataset/Zurich-RAW-to-DSLR']
		# 	for root in rootlist:
		# 		if os.path.isdir(root):
		# 			self.root = root
		# 			break

		self.batch_size = opt.batch_size
		self.mode = opt.mode  # RGB, Y or L=
		self.imio = imlib(self.mode, lib=opt.imlib)
		self.raw_imio = imlib('RAW', fmt='HWC', lib='cv2')

		if split == 'train':
			self.raw_dir = os.path.join(self.root, 'train', 'huawei_raw')
			self.dslr_dir = os.path.join(self.root, 'train', 'canon')
			self.names = ['%s'%i for i in range(0, 46839)] 
			self._getitem = self._getitem_train

		elif split == 'val':
			self.raw_dir = os.path.join(self.root, 'test', 'huawei_raw')
			self.dslr_dir = os.path.join(self.root, 'test', 'canon')
			self.names = ['%s'%i for i in range(0, 1204)] 
			self._getitem = self._getitem_test

		elif split == 'test':
			self.raw_dir = os.path.join(self.root, 'test', 'huawei_raw')
			self.dslr_dir = os.path.join(self.root, 'test', 'canon')
			self.names = ['%s'%i for i in range(0, 1204)]  
			self._getitem = self._getitem_test

		elif split == 'visual':
			self.raw_dir = os.path.join(self.root, 'full_resolution/huawei_raw')
			self.names = ['1072', '1096', '1167']
			self._getitem = self._getitem_visual

		else:
			raise ValueError

		self.len_data = len(self.names)

		self.raw_images = [0] * self.len_data
		self.coord = get_coord(H=448, W=448, x=1, y=1)
		read_images(self)

	def __getitem__(self, index):
		return self._getitem(index)

	def __len__(self):
		return self.len_data

	def _getitem_train(self, idx):
		raw_combined, raw_demosaic = self._process_raw(self.raw_images[idx])
		
		dslr_image = self.imio.read(os.path.join(self.dslr_dir, self.names[idx] + ".jpg"))
		dslr_image = np.float32(dslr_image) / 255.0

		raw_combined, raw_demosaic, dslr_image, coord = augment(
			raw_combined, raw_demosaic, dslr_image, self.coord)

		return {'raw': raw_combined,
				'raw_demosaic': raw_demosaic,
				'dslr': dslr_image,
				'coord': coord,
				'fname': self.names[idx]}

	def _getitem_test(self, idx):
		raw_combined, raw_demosaic = self._process_raw(self.raw_images[idx])
		
		dslr_image = self.imio.read(os.path.join(self.dslr_dir, self.names[idx] + ".jpg"))
		dslr_image = np.float32(dslr_image) / 255.0

		return {'raw': raw_combined,
				'raw_demosaic': raw_demosaic,
				'dslr': dslr_image,
				'coord': self.coord,
				'fname': self.names[idx]}

	def _getitem_visual(self, idx):
		raw_combined, raw_demosaic = self._process_raw(self.raw_images[idx])
		h, w = raw_demosaic.shape[-2:]
		coord = get_coord(H=h, W=w, x=1, y=1)

		return {'raw': raw_combined,
				'raw_demosaic': raw_demosaic,
				'dslr': raw_combined,
				'coord': coord,
				'fname': self.names[idx]}

	def _process_raw(self, raw):
		raw = remove_black_level(raw)
		raw_combined = extract_bayer_channels(raw)
		raw_demosaic = get_raw_demosaic(raw)
		return raw_combined, raw_demosaic

def iter_obj(num, objs):
	for i in range(num):
		yield (i, objs)

def imreader(arg):
	# Due to the memory (32 GB) limitation, here we only preload the raw images. 
	# If you have enough memory, you can also modify the code to preload the sRGB images to speed up the training process.
	i, obj = arg
	for _ in range(3):
		try:
			obj.raw_images[i] = obj.raw_imio.read(os.path.join(obj.raw_dir, obj.names[i] + '.png'))
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
	for _ in tqdm(pool.imap(imreader, iter_obj(obj.len_data, obj)), total=obj.len_data):
		pass
	pool.close()
	pool.join()

if __name__ == '__main__':
	pass