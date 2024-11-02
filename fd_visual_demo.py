from ultralytics import YOLO

model = YOLO(r'model\former_best.pt')

source = PATH
source2 = SECOND_PATH

results = model.predict(source=0, show = True, conf=0.6, stream = True)
for r in results:
    boxes = r.boxes
    masks = r.masks
    probs = r.probs
