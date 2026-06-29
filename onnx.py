import torch
from models.yolo import Model

device = 'cpu'
model_path = '/home/jndx01/yang/yolov5-master/runs/train/Dynamic/*****C3_dybase_A_599_act_546(41s)/weights/best.pt'

model = Model('/home/jndx01/yang/yolov5-master/models/test/C3_dybase_A.yaml', ch=3, nc=4)
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['model'].state_dict())

model.eval().to(device)


dummy = torch.randn(1, 3, 2048, 2048).to(device)

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy,
        '/home/jndx01/yang/yolov5-master/test.onnx',
        opset_version=12,
        input_names=['images'],
        output_names=['output']
    )

    print("Done")
