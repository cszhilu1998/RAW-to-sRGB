import numpy as np
import cv2
import torch
import math
from tqdm import tqdm
import lpips
import glob
from skimage.metrics import structural_similarity as ssim
import argparse


def calc_psnr_np(sr, hr, range=255.):
	diff = (sr.astype(np.float32) - hr.astype(np.float32)) / range
	mse = np.power(diff, 2).mean()
	return -10 * math.log10(mse)

# def rgb2gray(rgb):
# 	r, g, b=rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
# 	gray = 0.2989*r + 0.5870*g + 0.1140*b
# 	return gray

def lpips_norm(img):
	img = img[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
	img = img / (255. / 2.) - 1
	return torch.Tensor(img).to(device)

def calc_lpips(x_mask_out, x_canon, loss_fn_alex):
	lpips_mask_out = lpips_norm(x_mask_out)
	lpips_canon = lpips_norm(x_canon)
	LPIPS = loss_fn_alex(lpips_mask_out, lpips_canon)
	return LPIPS.detach().cpu().item()

def calc_metrics(x_mask_out, x_canon, loss_fn_alex):
	psnr = calc_psnr_np(x_mask_out, x_canon)
	SSIM = ssim(x_mask_out, x_canon, win_size=11, data_range=255, multichannel=True, gaussian_weights=True)
	LPIPS = calc_lpips(x_mask_out, x_canon, loss_fn_alex)
	return np.array([psnr, SSIM, LPIPS], dtype=float)

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Metrics for argparse')
	parser.add_argument('--name', type=str, required=True,
			            help='Name of the folder to save models and logs.')	
	parser.add_argument('--dataroot', type=str, default='')
	parser.add_argument('--device', default="0")
	args = parser.parse_args()

	device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
	loss_fn_alex_v1 = lpips.LPIPS(net='alex', version='0.1').to(device)

	# path_list = np.load("./to.npy")
	# names = ['%s'%i for i in path_list]  1204
	# names = ['%s'%i for i in range(0, 1204)]
	
	psnrs = []
	psnr_masks = []
	LPIPS_outs = []
	SSIMs = []
	SSIMs_mask = []
	LPIPS_out_masks = []

	files = [
		#  '/home/zzl/Code/ISP/MW-ISPNet/mwisp_43200/results/',
		# '/home/zzl/Code/ISP/AWNet_43200/result_4c_2/',
		# '/home/zzl/Code/ISP/AdaISP/AdaISP_rebuttal/ckpt/ispjoint_gamma_coord/',
		# '/home/zzl/Code/ISP/PyNET-PyTorch-43200/results/'
		'/Code/ISP/AdaISP/RAW-to-sRGB/RAW-to-sRGB/ckpt/zrrjoint/'
		# '/home/zzl/Code/ISP/SRMethods/PyTorch-SRGAN-master/test_all/',
		# '/home/zzl/Code/ISP/SRMethods/ESRGAN-master/test_all/',
		# '/home/zzl/Code/ISP/SRMethods/SPSR-master/test_all/',
		# '/home/zzl/Code/ISP/SRMethods/RealSR-master/test_all/',
		# '/home/zzl/Code/ISP/AWNet_43200/result_3c/'
	]
	
	# warp_target = '/Data/dataset/Zurich-RAW-to-DSLR/warp_test/'
	ori_target = argparse.dataroot + '/test/'
	# ori_target = '/home/zzl/Code/ISP/AdaISP/AdaISP_0208/ckpt/srraw_gt/'

	raw_dirs = sorted(glob.glob(files[0] + 'output/' + '*.png'))
	names = []
	for i, raw_name in enumerate(raw_dirs):
		raw_name_sp = raw_name.split('/')
		names.append(raw_name_sp[-1][:-4])
	
	ori_metrics = np.zeros([len(names), 3*2])
	# print(names)

	test_canon = []
	warp_test_canon = []
	warp_test_mask = []
	for name in tqdm(names): 
		test_canon.append(cv2.imread(ori_target + 'canon/' + name + '.jpg')[..., ::-1]) 
		# warp_test_canon.append(cv2.imread(warp_target + 'canon/' + name + '.png')[..., ::-1])
		# warp_test_mask.append(cv2.imread(warp_target + 'mask/' + name + '.png')[..., ::-1] / 255.0)

	# for name in tqdm(names): 
	# 	test_canon.append(cv2.imread(ori_target + name + '.png')[..., ::-1])

	for file in files:
		log_dir = '%s/log_metrics.txt' % (file)
		# os.makedirs(os.path.split(log_dir)[0], exist_ok=True)
		f = open(log_dir, 'a')
		i = 0
		for name in tqdm(names): 
			# canon = './ckpt/srraw_gt/' + name + '.png'
			x_out = cv2.imread(file + 'output/' + name + '.png')[..., ::-1]
			warp_gt_canon = cv2.imread(file + 'warp_gt/' + name + '.png')[..., ::-1] # [32:-32,32:-32, ::-1]
			warp_gt_mask = cv2.imread(file + 'warp_gt_mask/' + name + '.png')[..., ::-1] / 255.0
		
			# x_out = exposure.match_histograms(x_out*warp_gt_mask, warp_gt_canon, multichannel=True)

			ori_metrics[i, 0:4] = calc_metrics(x_out, test_canon[i], loss_fn_alex_v1)
			ori_metrics[i, 4:8] = calc_metrics(x_out*warp_test_mask[i], warp_test_canon[i], loss_fn_alex_v1)
			ori_metrics[i, 8:12] = calc_metrics(x_out*warp_gt_mask, warp_gt_canon, loss_fn_alex_v1)

			# f.write('name: %s, metrics: %s \n' % (name, ori_metrics[i]))
			i = i + 1
		
		f.write('\n file: %s \n mean: %s \n' % (file, np.mean(ori_metrics, axis=0)))
		print('\n file: %s \n mean: %s \n' % (file, np.mean(ori_metrics, axis=0)))
		# print('PSNR = %.2f, SSIM = %.2f, lpips_out = %.3f' % (np.mean(psnrs), np.mean(SSIMs), np.mean(LPIPS_outs)))
		# print('PSNR = %.2f, SSIM = %.4f, lpips_out = %.3f' % (np.mean(psnr_masks), np.mean(SSIMs_mask), np.mean(LPIPS_out_masks)))
	f.flush()
	f.close()
