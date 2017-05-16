from skimage import io, feature, color, transform
import numpy as np
import random
import os

VARIATION_CNT = 5

random.seed(1337)

for case_name in ["bark", "bikes", "boat", "graf", "leuven", "trees", "wall"]:
#for case_name in ["leuven"]:
	imgs = []

	for i in range(1,7):
		tmp_img = " "
		try:
			tmp_img = io.imread("samples/raw/%s/img%d.ppm" % (case_name, i,))
		except FileNotFoundError:
			tmp_img = io.imread("samples/raw/%s/img%d.pgm" % (case_name, i,))
		imgs += [tmp_img]
	transforms = [np.eye(3, dtype=np.float_)]

	for i in range(2,7):
		tab = []
		with open("samples/raw/%s/H1to%dp" % (case_name, i, ), "r") as f:
			for l in f:
				if l.strip():
					tab += [[float(sub) for sub in filter(lambda x: x, l.strip().split(" "))]]
		transforms += [np.array(tab)]

		
	pt_cnt = 0

	SPAN = 32
	MARGIN = 20

	collected = []

	os.makedirs("samples/%s" % (case_name, ), exist_ok = True)
	out_txt = open("samples/%s/matches.csv" % (case_name, ), "w")
	
	for row, col in feature.corner_peaks(feature.corner_harris(color.rgb2gray(imgs[0]))):
		if row < SPAN or imgs[0].shape[0] - row <= SPAN or col < SPAN or imgs[0].shape[1] - col <= SPAN:
			continue
		good = True
		for arow, acol in collected:
			drow, dcol = row-arow, col-acol
			if drow*drow+dcol*dcol < MARGIN*MARGIN: good = False; break
		if not good: continue
		start_cnt = pt_cnt
		for variation in range(VARIATION_CNT):
			offr, offc = (0, 0) if (variation == 0) else (random.uniform(-10,10)//3, random.uniform(-10,10)//3)
			offrot = 0 if (variation == 0) else (random.gauss(0,60))
			offscale = 1 if (variation == 0) else (2**(random.random()*2-1))
			for i in range(6):
				r, c = row + offr, col + offc
				if i > 0:
					c, r, tmp = np.dot(transforms[i], np.array([c,r,1])).flatten()
					r, c = int(r/tmp), int(c/tmp)
				off_span = SPAN*offscale*1.4 # 1.4 ~= sqrt(2)
				if r < off_span or imgs[i].shape[0] - r <= off_span or c < off_span or imgs[i].shape[1] - c <= off_span:
					continue
				r, c = int(r*offscale), int(c*offscale)
				transimg = transform.rotate(transform.rescale(imgs[i], offscale, mode="reflect"), offrot, center=(c,r), mode="reflect")
				if r < SPAN or transimg.shape[0] - r <= SPAN or c < SPAN or transimg.shape[1] - c <= SPAN:
					continue
				sample_slice = transimg[r-SPAN:r+SPAN,c-SPAN:c+SPAN]
				io.imsave("samples/%s/%05d.png" % (case_name, pt_cnt, ), sample_slice)
				for j in range(start_cnt, pt_cnt):
					out_txt.write("%d,%d\n" % (pt_cnt, j))
				pt_cnt += 1
		collected += [(row, col)]
		
	out_txt.close()
