import gc

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

models = {
  'vit_b': './checkpoints/sam_vit_b_01ec64.pth',
  'vit_l': './checkpoints/sam_vit_l_0b3195.pth',
  'vit_h': './checkpoints/sam_vit_h_4b8939.pth'
}


def get_sam_predictor(model_type='vit_h', device=None, image=None):
  if device is None and torch.cuda.is_available():
    device = 'cuda'
  elif device is None:
    device = 'cpu'
  # sam model
  sam = sam_model_registry[model_type](checkpoint=models[model_type])
  sam = sam.to(device)

  predictor = SamPredictor(sam)
  if image is not None:
    predictor.set_image(image)
  return predictor


def run_inference(predictor: SamPredictor, input_x, selected_points,
                  multi_object: bool = False):
  # Process the image to produce an image embedding
  # points
  points = torch.Tensor(
      [p for p, _ in selected_points]
  ).to(predictor.device).unsqueeze(1)

  labels = torch.Tensor(
      [int(l) for _, l in selected_points]
  ).to(predictor.device).unsqueeze(1)

  transformed_points = predictor.transform.apply_coords_torch(
      points, input_x.shape[:2])
  # print(transformed_points.shape)
  # predict segmentation according to the boxes
  masks, scores, logits = predictor.predict_torch(
    point_coords=transformed_points,
    point_labels=labels,
    multimask_output=True,
  )

  masks = masks[labels[:, 0] == 1, 0].cpu().detach().numpy()
  if not multi_object:
    masks: np.ndarray = masks.max(axis=0, keepdims=True)
  gc.collect()
  torch.cuda.empty_cache()
  
  return [(mask, f'mask_{i}') for i, mask in enumerate(masks)]
