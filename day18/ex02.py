#%%
import numpy as np
import cv2 as cv
from ultralytics import YOLO,checks

from IPython.display import display
import PIL.Image as Image
checks()
# %%
img = cv.imread("bus.jpg")
#display(Image.fromarray(img)) converting을 안해서 색이 이상하게 나옴
display(Image.fromarray(cv.cvtColor(img,cv.COLOR_BGR2RGB)))
# %% 이번 모델은 전위학습 시키기 위한 모델
#model = YOLO('yolo11n-seg.pt')  
model = YOLO('yolo11x-seg.pt') #튀는것을 줄이기 위해 더 큰 모델 사용
results = model(img)
print(results)
# %%
result = results[0]
result_img = img.copy()
# %%
print(result.masks)
# %%
for mask_data in result.masks.data :
    mask_img = mask_data.cpu().numpy()
    mask_img = np.where(mask_img > 0.5, 255,0).astype('uint8')
    display(Image.fromarray(cv.cvtColor(mask_img,cv.COLOR_BGR2RGB)))    
# %%
img_h, img_w,_ = result_img.shape

for _segment in result.masks.xyn :
    np_cnt = (_segment *(img_w,img_h)).astype(np.int32)
    cv.polylines(result_img,[np_cnt],True,(0,255,0),2)
        
display(Image.fromarray(cv.cvtColor(result_img,cv.COLOR_BGR2RGB)))
# %%
