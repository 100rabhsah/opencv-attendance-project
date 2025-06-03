import cv2
import numpy as np
img = np.zeros((100, 100, 3), dtype=np.uint8)
img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
print("Success!", img2.shape)