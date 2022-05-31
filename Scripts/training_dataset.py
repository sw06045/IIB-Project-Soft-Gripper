from __future__ import annotations 
import random 
import json

import cv2
import numpy as np
import re
from numpy.core.records import record

from pkg_resources import get_distribution
from portalocker.utils import Filename

from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.structures.boxes import BoxMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import Instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import os


from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer,VisImage


from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import logging
import torch
import time
import datetime


import re

def get_dicts(directory):
    classes = ['Mango', "Lime", "Cucumber"]
    dataset_dicts = []
    #image_id = 0
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)
            #print(img_anns)

        record = {}
        #print(img_anns.keys())
        #print(img_anns["imageData"])
        filename = os.path.join(directory, img_anns["imagePath"])
        
        record["file_name"] = filename
        #print(filename)
        record["image_id"] = re.findall("\d+", filename)[0]
        print(record["image_id"])
        print (re.findall("\d+", filename))
        #image_id += 1
        record["height"] = img_anns["imageHeight"]
        record["width"] = img_anns["imageWidth"]
      
        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']]
            py = [a[1] for a in anno['points']]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


    #Registering of classes
for d in ["train", "val"]:
    #register_coco_instances(d, {}, f"images/TMR/{d}/trainval.json", f"images/TMR/{d}")
    DatasetCatalog.register(d, lambda d=d: get_dicts("images/Final/" + d))
    # Set the class name
    MetadataCatalog.get(d).set(thing_classes= ["Mango", "Lime", "Cucumber"])
produce_meta_data = MetadataCatalog.get("train")


dataset_dicts = get_dicts("images/Final/")
for d in dataset_dicts:
    img: np.ndarray = cv2.imread(d["file_name"])
    v = Visualizer(img[:, :, ::-1],
                    metadata = produce_meta_data,
                    scale = 0.5)
    result: VisImage = v.draw_dataset_dict(d)
    result_image: np.ndarray = result.get_image()[:, :, ::-1]

    out_file_name: str = re.search(r"(.*)\.", d["file_name"][-10:-3]).group(0)[:-1]
    out_file_name += "_processed.png"
    out = v.draw_dataset_dict(d)
    cv2.imwrite("images/Final/train_processed/" + out_file_name, result_image)

# For validation dataset
class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)





#Train
class DefaultTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name = "val", output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    def build_hooks(self):
        print(self.cfg.DATASETS.TEST)
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST,
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train",) 
cfg.DATASETS.TEST = ("val",)
cfg.TEST.EVAL_PERIOD = 100
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.SOLVER.STEPS = []   
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.DEVICE = "cpu"






os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer= DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
#trainer.scheduler.milestones=cfg.SOLVER.STEPS



##### COMMENT IN TO TRAIN
#trainer.train()




import json
import matplotlib.pyplot as plt

#Plotting of training and validation loss

experiment_folder = './output/model_iter4000_lr0005_wf1_date2020_03_20__05_16_45'
metrics_location = os.path.join(cfg.OUTPUT_DIR, "metrics.json")
def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

experiment_metrics = load_json_arr(metrics_location)
#print(experiment_metrics)
total_iteration_list = []
total_loss_list = []
validation_iteration_list = []
validation_loss_list = []
for x in experiment_metrics[:-1]:
    print(x)
    print(x.keys())
#    if "total_loss" in x.keys():   
#        #print(x)
        #print(x["iteration"])
        #print(x["total_loss"])
#        total_iteration_list.append(x["iteration"])
#        total_loss_list.append(x["total_loss"])
#        if "validation_loss" in x.keys():
#            validation_iteration_list.append(x["iteration"])    
#            validation_loss_list.append(x["validation_loss"])

plt.plot(
    [x['iteration'] for x in experiment_metrics[:-1] if "total_loss" in x],
    [x['total_loss'] for x in experiment_metrics[:-1] if "total_loss" in x])
plt.plot(
    [x['iteration'] for x in experiment_metrics[:-1] if 'validation_loss' in x], 
    [x['validation_loss'] for x in experiment_metrics[:-1] if 'validation_loss' in x])
plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
plt.xlabel('Iterations')
plt.ylabel('Loss')
#plt.show()
#inference_on_dataset(trainer.model, val_loader, evaluator)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained


print(os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # set a custom testing threshold
predictor = DefaultPredictor(cfg)




evaluator = COCOEvaluator("val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "val")
#val_loader = build_detection_test_loader(cfg, "val")
inf = inference_on_dataset(predictor.model, val_loader, evaluator)
#print(inf)


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))

    mask1_area = np.count_nonzero(masks1 == 1)
    mask2_area = np.count_nonzero(masks2 == 1)
    #print(mask1_area, mask2_area)

    intersection = np.count_nonzero(np.logical_and( masks1==1,  masks2==1 ))
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou



from pickle import HIGHEST_PROTOCOL
#Determining Precision and Recall based on a certain Intersection of Union threshold
#confidence score of 90%

import pycocotools.mask as mask_util


TP = 0
FP = 0
FN = 0
iou_threshold = 0.5

id = 1


dataset_dicts = get_dicts("images/Final/Combine") #dataset_dicts = get_dicts("images/Final/val")


for d in dataset_dicts:
    img = cv2.imread(d["file_name"])
    img_label = cv2.imread(d["file_name"])
    img_pred = cv2.imread(d["file_name"])


    #print(d)
    height = d['height']
    width = d['width']
    v = Visualizer(img_label[:, :, ::-1], metadata=produce_meta_data)

    #Visualise actual labels
    out_labels = v.draw_dataset_dict(d)
    cv2.imwrite("images/Final/processed/" + str(d["file_name"][-10:-3]) +"_label.png", out_labels.get_image()[:, :, ::-1])
    
    #conversion of the polygons to mask data
    polygons = []
    for pgs in d["annotations"]:


        if pgs["category_id"] == id:
            continue
        current_mask = pgs['segmentation']
        #print(current_mask)
        rle = mask_util.frPyObjects(current_mask, height, width)
        rle = mask_util.merge(rle)
        decode = mask_util.decode(rle)[:, :]
      #print(np.count_nonzero(decode == 1))
        polygons.append(decode)

    mask_labels = np.asarray(polygons)

    
    #Visualise predicted labels
    outputs = predictor(img_pred)
    v = Visualizer(img_label[:, :, ::-1], metadata=produce_meta_data)
    out_predictions = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    mask_predictions = np.asarray(outputs["instances"].pred_masks.to("cpu").numpy())*1

    #print(mask_predictions)

    cv2.imwrite("images/Final/processed/" + str(d["file_name"][-10:-3]) + "_predict.png", out_predictions.get_image()[:, :, ::-1])
    
    #Show photo
    #cv2_imshow(out_labels.get_image()[:, :, ::-1])
    #cv2_imshow(out_predictions.get_image()[:, :, ::-1])
    

    #Determining True Positives and False Positives
    print("number of predictions", len(mask_predictions))
    print("number of actual labels", len(mask_labels))
    for mp_id, mp in enumerate(mask_predictions):
      #skip if not evaluating the right object
      if outputs["instances"].pred_classes[mp_id].to("cpu").numpy() == id:
        continue
      IoU = 0
      for ml in mask_labels:
        IoU = compute_overlaps_masks(mp, ml)
        #print(IoU)
        if IoU > iou_threshold:
          TP += 1
          break
      #print("----------------")

      if IoU < iou_threshold:
        FP +=1

    #Determinig False Negatives

    for ml in mask_labels:
      HIGHEST_IoU = 0
      for mp in mask_predictions:

        #skip if not evaluating for the right object
        #if outputs["instances"].pred_classes[mp_id].to("cpu").numpy() != id:
        #  continue
        IoU = compute_overlaps_masks(mp, ml)
        if IoU > HIGHEST_IoU:
          HIGHEST_IoU = IoU
      if HIGHEST_IoU < iou_threshold:
        FN +=1 

print("True Positives", TP)
print("False Positives", FP)
print("False Negatives", FN)

precision = TP/(TP + FP)
recall = TP/(TP + FN)

print("Precision", precision)
print("Recall", recall)



# Change all Final to TMR
