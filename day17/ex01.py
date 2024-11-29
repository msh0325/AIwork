#%%
from ultralytics import YOLO

#load a pretrained YOLO11n model
model = YOLO("yolo11m.pt")

#run inference on 'aname.jpg/ wotj arguments
model.predict("Makati_intersection.jpg",save=True,imgsz=320,conf=0.5)
# %%
