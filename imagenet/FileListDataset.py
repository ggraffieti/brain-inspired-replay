import torch.utils.data as data

from PIL import Image
import os
import os.path
import multiprocessing as mp


def default_loader(path):
	return Image.open(path).convert('RGB')


def is_image_valid(path):
	try:
		with Image.open(path) as pil_img:
			pil_img.verify()
		return True
	except:
		return False


def default_flist_reader(flist, root, exclusion_list=None):
	"""
	flist format: impath label\nimpath label\n ...(same to caffe's filelist)
	"""
	imlist = []
	if exclusion_list is None:
		exclusion_list = []
	exclusion_list = set(exclusion_list)

	with open(flist, 'r') as rf:
		all_lines = rf.readlines()
		all_lines = [i for j, i in enumerate(all_lines) if j not in exclusion_list]

	N = mp.cpu_count()
	with mp.Pool(processes=N) as p:
		validity_check = p.map(is_image_valid, [os.path.join(root, impath) for impath in all_lines])
	bad_images_idx = [j for j, i in enumerate(validity_check) if i]
	print('Found', len(bad_images_idx), 'bad images')
	for bad_image_idx in bad_images_idx:
		print(os.path.join(root, all_lines[bad_image_idx]))
	all_lines = [i for j, i in enumerate(all_lines) if j not in bad_images_idx]

	for line_idx, line in enumerate(all_lines):
		impath, imlabel = line.strip().split()
		complete_path = os.path.join(root, impath)
		imlist.append((complete_path, int(imlabel)))

	return imlist


class ImageFilelist(data.Dataset):
	def __init__(self, root, flist, transform=None, target_transform=None,
			flist_reader=default_flist_reader, loader=default_loader):
		self.root = root
		self.imgs = flist_reader(flist, root)
		self.targets = [img_data[1] for img_data in self.imgs]
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

	def __getitem__(self, index):
		impath, target = self.imgs[index]
		img = self.loader(os.path.join(self.root,impath))
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		
		return img, target

	def __len__(self):
		return len(self.imgs)
