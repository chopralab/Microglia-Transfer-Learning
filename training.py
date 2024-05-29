from ultralytics import YOLO
import json

model = YOLO("yolov8x.yaml")
model = YOLO("/scratch/gilbreth/jpfinley/ultralytics/runs/detect/train39/weights/best.pt")


#hyps
n_tune = "/scratch/gilbreth/jpfinley/ultralytics/runs/detect/tune1/_tune_2024-02-19_18-03-26/_tune_167cf_00009_9_box=0.0623,cls=0.2717,copy_paste=0.9517,degrees=6.7196,fliplr=0.9701,flipud=0.3774,hsv_h=0.0835,hsv_s=0.6179,h_2024-02-19_18-03-26/params.json"
s_tune = "/scratch/gilbreth/jpfinley/ultralytics/runs/detect/tune2/_tune_2024-02-19_23-25-24/_tune_1160c_00005_5_box=0.1882,cls=2.6863,copy_paste=0.1336,degrees=7.9790,fliplr=0.8386,flipud=0.4181,hsv_h=0.0156,hsv_s=0.6181,h_2024-02-19_23-25-24/params.json"
l_tune = '/scratch/gilbreth/jpfinley/ultralytics/runs/detect/tune3/_tune_2024-02-20_19-08-34/_tune_5aa2a_00008_8_box=0.0499,cls=2.3028,copy_paste=0.0246,degrees=25.3505,fliplr=0.3324,flipud=0.8894,hsv_h=0.0311,hsv_s=0.4042,_2024-02-20_19-08-34/params.json'
m_tune = "/scratch/gilbreth/jpfinley/ultralytics/runs/detect/tune4/_tune_2024-02-20_20-51-47/_tune_c5d79_00009_9_box=0.0631,cls=2.5456,copy_paste=0.7217,degrees=6.1297,fliplr=0.4284,flipud=0.2787,hsv_h=0.0828,hsv_s=0.1120,h_2024-02-20_20-51-47/params.json"
x_tune = "/scratch/gilbreth/jpfinley/ultralytics/runs/detect/tune5/_tune_2024-02-20_22-45-10/_tune_9d05f_00009_9_box=0.0201,cls=1.3836,copy_paste=0.1825,degrees=26.1948,fliplr=0.6723,flipud=0.7239,hsv_h=0.0696,hsv_s=0.3685,_2024-02-20_22-45-11/params.json"

best_string = n_tune

with open(best_string) as hyps_file:
    hyps = json.load(hyps_file)
    hyps_file.close()

model.train(epochs=50, 
box = hyps['box'],
cls = hyps['cls'],
copy_paste = hyps['copy_paste'],
data = hyps['data'],
degrees = hyps['degrees'],
fliplr = hyps['fliplr'],
flipud = hyps['flipud'],
hsv_h = hyps['hsv_h'],
hsv_s = hyps['hsv_s'],
hsv_v = hyps['hsv_v'],
lr0 = hyps['lr0'],
lrf = hyps['lrf'],
mixup = hyps['mixup'],
momentum = hyps['momentum'],
mosaic = hyps['mosaic'],
perspective = hyps['perspective'],
scale = hyps['scale'],
shear = hyps['shear'],
translate = hyps['translate'],
warmup_epochs = hyps['warmup_epochs'],
warmup_momentum = hyps['warmup_momentum'],
weight_decay = hyps['weight_decay'])

metrics = model.val()

print(metrics)