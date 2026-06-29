# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run inference with a trained YOLOv5 model and save both detections and intermediate feature map visualizations.

Example:
    python infer_visualize.py --weights runs/train/exp/weights/best.pt --source data/images --imgsz 640
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import LOGGER, check_img_size, check_yaml, increment_path, non_max_suppression, print_args, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path
    source=ROOT / "data/images",  # file/dir path
    data=ROOT / "data/coco128.yaml",  # dataset yaml for class names
    imgsz=640,  # inference size
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    device="",  # cuda device or cpu
    half=False,  # FP16 inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    augment=False,  # augmented inference
    line_thickness=2,  # bounding box thickness
    project=ROOT / "runs/feature_vis",  # save directory root
    name="exp",  # run name
    exist_ok=False,  # allow existing project/name
):
    source = str(source)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run directory
    preds_dir = save_dir / "predictions"
    features_dir = save_dir / "features"
    preds_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    data = check_yaml(data)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = [imgsz] * 2 if isinstance(imgsz, int) else imgsz
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model.eval()

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))  # warmup model

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if im.ndimension() == 3:
            im = im[None]  # expand for batch dim

        feature_save_dir = features_dir / Path(path).stem
        feature_save_dir.mkdir(parents=True, exist_ok=True)

        pred = model(im, augment=augment, visualize=feature_save_dir)
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

        im0 = im0s.copy()
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        det = pred[0]
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in det:
                c = int(cls)
                label = f"{names[c]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(c, True))

        result = annotator.result()
        save_path = preds_dir / Path(path).name
        cv2.imwrite(str(save_path), result)
        LOGGER.info(f"{Path(path).name}: detections -> {save_path}, feature maps -> {feature_save_dir}")

    LOGGER.info(f"All results saved to {save_dir}")
    return str(save_dir)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default='/home/jndx01/yang/yolov5-master/runs/train/paper3/C2f_triple12(599)_599_546_noamp/weights/best.pt', help="model path")
    parser.add_argument("--source", type=str, default='/home/jndx01/yang/yolov5-master/展示图/599', help="file/dir for inference")
    parser.add_argument("--data", type=str, default='/home/jndx01/yang/yolov5-master/data/599.yaml', help="dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[2048], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--line-thickness", default=10, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--project", default=ROOT / "runs/feature_vis", help="directory to save results")
    parser.add_argument("--name", default="exp", help="run name")
    parser.add_argument("--exist-ok", action="store_true", help="overwrite existing project/name")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand if only one dim is provided
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    main(parse_opt())
