import numpy as np
import cv2
import torch
import math
from tqdm import tqdm
import lpips
import glob
from skimage.metrics import structural_similarity as ssim
import argparse
import sys

def calc_psnr_np(sr, hr, range=255.):
	diff = (sr.astype(np.float32) - hr.astype(np.float32)) / range
	mse = np.power(diff, 2).mean()
	return -10 * math.log10(mse)

def lpips_norm(img):
	img = img[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
	img = img / (255. / 2.) - 1
	return torch.Tensor(img).to(device)

def calc_lpips(out, target, loss_fn_alex):
	lpips_out = lpips_norm(out)
	lpips_target = lpips_norm(target)
	LPIPS = loss_fn_alex(lpips_out, lpips_target)
	return LPIPS.detach().cpu().item()

def calc_metrics(out, target, loss_fn_alex):
	psnr = calc_psnr_np(out, target)
	SSIM = ssim(out, target, win_size=11, data_range=255, multichannel=True, gaussian_weights=True)
	LPIPS = calc_lpips(out, target, loss_fn_alex)
	return np.array([psnr, SSIM, LPIPS], dtype=float)


if __name__ == '__main__':

	map = {'00001.png':'00005.png',
	       '00001.png':'00005.png',
		   '00001.png':'00005.png',}
	raw_dirs = sorted(glob.glob('/Data/dataset/SRRAW/test/*'))
	for raw_dir in raw_dirs:
		names = sorted(glob.glob(raw_dir + '/target_x4_rgb/*.png'))
		for name in names:
			pre_name = name[-9:]
			post_name = int(pre_name) + 

	# parser = argparse.ArgumentParser(description='Metrics for argparse')
	# parser.add_argument('--name', type=str, required=True,
	# 		            help='Name of the folder to save models and logs.')	
	# parser.add_argument('--dataroot', type=str, default='')
	# parser.add_argument('--device', default="0")
	# args = parser.parse_args()

	# device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
	# loss_fn_alex_v1 = lpips.LPIPS(net='alex', version='0.1').to(device)

	# root = sys.path[0]
	# files = [
	# 	root + '/ckpt/' + args.name,
	# ]
	
	# raw_dirs = sorted(glob.glob(files[0] + '/output/' + '*.png'))
	# image_names = []
	# for i, raw_name in enumerate(raw_dirs):
	# 	raw_name_sp = raw_name.split('/')
	# 	image_names.append(raw_name_sp[-1][:-4])
	
	# for file in files:
	# 	print('Start to measure images in %s...' % (file))
	# 	metrics = np.zeros([len(image_names), 3*2])
	# 	log_dir = '%s/log_metrics.txt' % (file)
	# 	f = open(log_dir, 'a')
	# 	i = 0
	# 	for name in tqdm(image_names): 
	# 		gt = cv2.imread(args.dataroot + '/test/canon/' + name + '.jpg')[..., ::-1]
	# 		output = cv2.imread(file + '/output/' + name + '.png')[..., ::-1]
	# 		warp_gt = cv2.imread(file + '/warp_gt/' + name + '.png')[..., ::-1]
	# 		warp_gt_mask = cv2.imread(file + '/warp_gt_mask/' + name + '.png')[..., ::-1] / 255.0
		
	# 		metrics[i, 0:3] = calc_metrics(output, gt, loss_fn_alex_v1)
	# 		metrics[i, 3:6] = calc_metrics(output * warp_gt_mask, warp_gt, loss_fn_alex_v1)

	# 		i = i + 1
		
	# 	mean_metrics = np.mean(metrics, axis=0)
		
	# 	print('\n        File        :\t %s \n' % (file))
	# 	print('   Original    GT   :\t PSNR = %.2f, SSIM = %.4f, LPIPS = %.3f \n' 
	# 	        % (mean_metrics[0], mean_metrics[1], mean_metrics[2]))
	# 	print('Align GT with result:\t PSNR = %.2f, SSIM = %.4f, LPIPS = %.3f \n' 
	# 	        % (mean_metrics[3], mean_metrics[4], mean_metrics[5]))

	# 	f.write('\n        File        :\t %s \n' % (file))
	# 	f.write('   Original    GT   :\t PSNR = %.2f, SSIM = %.4f, LPIPS = %.3f \n' 
	# 	        % (mean_metrics[0], mean_metrics[1], mean_metrics[2]))
	# 	f.write('Align GT with result:\t PSNR = %.2f, SSIM = %.4f, LPIPS = %.3f \n' 
	# 	        % (mean_metrics[3], mean_metrics[4], mean_metrics[5]))

	# 	f.flush()
	# 	f.close()
