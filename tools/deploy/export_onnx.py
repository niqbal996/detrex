#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
import sys
import time
import torch
import numpy as np
from torch.nn.parallel import DataParallel, DistributedDataParallel
import onnxruntime

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import (
    STABLE_ONNX_OPSET_VERSION,
    TracingAdapter,
    dump_torchscript_IR,
    scripting_with_instances,
)
from detectron2.modeling import GeneralizedRCNN, RetinaNet, build_model
from detectron2.modeling.postprocessing import detector_postprocess
# from detectron2.projects.point_rend import add_pointrend_config
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

logger = logging.getLogger("detrex")

def scale_boxes(boxes, current_size=(800, 1067), new_size=(1536, 2048)):
    x_factor = new_size[0] / current_size[0]
    y_factor = new_size[1] / current_size[1]
    boxes[:, 0] = boxes[:, 0] * x_factor
    boxes[:, 2] = boxes[:, 2] * x_factor
    boxes[:, 1] = boxes[:, 1] * y_factor
    boxes[:, 3] = boxes[:, 3] * y_factor
    return boxes

def export_tracing(torch_model, inputs):
    assert TORCH_VERSION >= (1, 8)
    image = inputs[0]["image"]
    inputs = [{"image": image}]  # remove other unused keys

    if isinstance(torch_model, GeneralizedRCNN):

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly

    traceable_model = TracingAdapter(torch_model, inputs, inference)

    # if args.format == "torchscript":
    #     ts_model = torch.jit.trace(traceable_model, (image,))
    #     with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
    #         torch.jit.save(ts_model, f)
    #     dump_torchscript_IR(ts_model, args.output)
    # elif args.format == "onnx":
    with PathManager.open(os.path.join('output/dino_r50_4scale_12ep', "model.onnx"), "wb") as f:
        torch.onnx.export(traceable_model, (image,), f, opset_version=16)
    logger.info("Inputs schema: " + str(traceable_model.inputs_schema))
    logger.info("Outputs schema: " + str(traceable_model.outputs_schema))

    # if args.format != "torchscript":
    #     return None
    # if not isinstance(torch_model, (GeneralizedRCNN, RetinaNet)):
    #     return None

    def eval_wrapper(inputs):
        """
        The exported model does not contain the final resize step, which is typically
        unused in deployment but needed for evaluation. We add it manually here.
        """
        input = inputs[0]
        instances = traceable_model.outputs_schema(ts_model(input["image"]))[0]["instances"]
        postprocessed = detector_postprocess(instances, input["height"], input["width"])
        return [{"instances": postprocessed}]

    return eval_wrapper

def infer_onnx(model_file, sample_input):
    import cv2
    exec_providers = onnxruntime.get_available_providers()
    exec_provider = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in exec_providers else ['CPUExecutionProvider']

    session = onnxruntime.InferenceSession(model_file, sess_options=None, providers=exec_provider)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    pred = session.run(None, {input_name: sample_input[0]['image'].numpy()})
    # pred = session.run(None, {input_name: cv2.imread('/home/niqbal/98.png')})
    conf_inds = np.where(pred[2] > 0.50)
    filtered = {}
    filtered[0] = pred[0][conf_inds]
    filtered[1] = pred[1][conf_inds]
    filtered[2] = pred[2][conf_inds]
    filtered[3] = pred[3]
    filtered[0] = scale_boxes(filtered[0],
                               current_size=(sample_input[0]['image'].shape[2],
                                             sample_input[0]['image'].shape[1]),
                               new_size=(sample_input[0]['width'],
                                         sample_input[0]['height']))
    # filtered[0] = scale_boxes(filtered[0],
    #                            current_size=(sample_input[0]['image'].shape[2],
    #                                          sample_input[0]['image'].shape[1]),
    #                            new_size=(sample_input[0]['width'],
    #                                      sample_input[0]['height']))
    orig_image = cv2.imread('/home/niqbal/98.png')
    # orig_image = sample_input[0]['image'].numpy()
    # orig_image = np.transpose(orig_image, (1,2,0))
    class_ids = {0: 'weeds', 1: 'maize'}

    for obj in range(filtered[0].shape[0]):
        box = filtered[0][obj, :]
        if filtered[1][obj] == 0:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        cv2.rectangle(orig_image,
                      pt1=(int(box[0]), int(box[1])),
                      pt2=(int(box[2]), int(box[3])),
                      color=color,
                      thickness=2)
        cv2.putText(orig_image,
                    '{:.2f} {}'.format(filtered[2][obj], class_ids[filtered[1][obj]]),
                    org=(int(box[0]), int(box[1] - 10)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    thickness=2,
                    color=color)
    # cv2.imwrite("./output_data/{:04}.png".format(count), orig)
    cv2.imshow('figure', orig_image, )
    cv2.waitKey()


def get_sample_inputs(cfg, device='cuda:0'):
    original_image = detection_utils.read_image('/home/niqbal/98.png', format='BGR')
    # Do same preprocessing as DefaultPredictor
    aug = T.ResizeShortestEdge([800, 800], 1333)
    height, width = original_image.shape[:2]
    image = aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(device)

    inputs = {"image": image, "height": height, "width": width}

    # Sample ready
    sample_inputs = [inputs]
    return sample_inputs

def register_maize():
    from detectron2.data.datasets import register_coco_instances

    register_coco_instances("maize_syn_v2_train", {},
                            "/media/niqbal/T7/datasets/Corn_syn_dataset/2022_GIL_Paper_Dataset_V2/coco_anns/instances_train_2022_2.json",
                            "/media/niqbal/T7/datasets/Corn_syn_dataset/2022_GIL_Paper_Dataset_V2/camera_main_camera/rect")
    register_coco_instances("maize_real_v2_val", {},
                            "/media/niqbal/T7/datasets/Corn_syn_dataset/2022_GIL_Paper_Dataset_V2/coco_anns/instances_val_2022.json",
                            "/media/niqbal/T7/datasets/Corn_syn_dataset/GIL_dataset/all_days/data")
def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    register_maize()
    cfg.train.device = 'cpu'

    # Initiate the model with weights
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()

    sample_inputs = get_sample_inputs(cfg, cfg.train.device)
    # exported_model = export_scripting(torch_model)
    exported_model = export_tracing(model, sample_inputs)
    infer_onnx('output/dino_r50_4scale_12ep/model.onnx', sample_inputs)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
