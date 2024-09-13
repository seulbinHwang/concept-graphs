from PIL import Image
import numpy as np

# 이미지 파일 경로
image_path = 'frame000000.jpg'

# 이미지 열기
image = Image.open(image_path)

# 이미지 데이터를 numpy 배열로 변환
image_pil = np.array(image)

import cv2
import numpy as np


# 이미지 읽기 (BGR 순서로 읽힘)
image_cv2 = cv2.imread(image_path)



import matplotlib.pyplot as plt
import numpy as np


# 이미지 읽기 (RGB 순서로 읽힘)
image = plt.imread(image_path)

# numpy 배열로 이미지를 가져옴
image_plt = np.array(image)

import imageio.v2 as imageio  # imageio 버전에 따라 v2 모듈을 사용할 수 있습니다.
import numpy as np


# 이미지 읽기 (RGB 순서로 읽힘)
image = imageio.imread(image_path)

# numpy 배열로 이미지를 가져옴
image_imageio= np.array(image)

from skimage import io
import numpy as np


# 이미지 읽기 (RGB 순서로 읽힘)
image = io.imread(image_path)

# numpy 배열로 이미지를 가져옴
image_skimage = np.array(image)


import cv2
# show 5 images in a row
stacked_images = np.hstack([image_cv2, image_pil, image_plt, image_imageio, image_skimage])
cv2.imshow('5 images', stacked_images)
cv2.waitKey(0)

