import os
from typing import List

import cv2
import gradio as gr
import numpy as np

from inference import SamPredictor, get_sam_predictor, run_inference

# points color and marker
COLORS = [(255, 0, 0), (0, 255, 0)]
MARKERS = [1, 5]


def save_masks(o_masks):
  o_files = []
  for mask, name in o_masks:
    o_mask = np.uint8(mask * 255)
    o_file = os.path.join('temp', name) + '.png'
    cv2.imwrite(o_file, o_mask)
    o_files.append(o_file)
  return o_files


def select_point(predictor: SamPredictor,
                 original_img: np.ndarray,
                 multi_object: List,
                 sel_pix: list,
                 point_type: str,
                 evt: gr.SelectData):
  """When the user clicks on the image, show points and update the mask.

  Args:
      predictor (SamPredictor): Sam predictor.
      original_img (np.ndarray): Input image.
      sel_pix (list): List of selected points.
      point_type (str): Point type (foreground/background).
      evt (gr.SelectData): Event data.

  Returns:
      np.ndarray: Annotated image.
      np.ndarray: Image with mask.
      np.ndarray: Mask.
  """
  img = original_img.copy()
  if point_type == 'foreground_point':
    sel_pix.append((evt.index, 1))   # append the foreground_point
  elif point_type == 'background_point':
    sel_pix.append((evt.index, 0))    # append the background_point
  else:
    sel_pix.append((evt.index, 1))    # default foreground_point
  # run inference
  o_masks = run_inference(predictor, original_img, sel_pix, multi_object)
  # draw points
  for point, label in sel_pix:
    cv2.drawMarker(img, point, COLORS[label], markerType=MARKERS[label],
                   markerSize=5, thickness=2)
  o_files = save_masks(o_masks)
  return img, (img, o_masks), o_files


# undo the selected point
def undo_points(predictor, orig_img, multi_object, sel_pix):
  temp = orig_img.copy()
  # draw points
  if len(sel_pix) != 0:
    sel_pix.pop()
    for point, label in sel_pix:
      cv2.drawMarker(temp, point, COLORS[label], markerType=MARKERS[label],
                     markerSize=5, thickness=2)
  o_masks = run_inference(predictor, orig_img, sel_pix, multi_object)
  o_files = save_masks(o_masks)
  return temp, (temp, o_masks), o_files


# once user upload an image, the original image is stored in `original_image`
def reset_image(predictor, img, rgb_vals, pre_processing=[]):
  # when new image is uploaded, `selected_points` should be empty
  preprocessed_image = img.copy()
  for c in 'RGB':
    if c not in rgb_vals:
      preprocessed_image[:, :, 'RGB'.index(c)] = 0
  if "CLAHE" in pre_processing:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    preprocessed_image = np.stack(
      [clahe.apply(preprocessed_image[:, :, i]) for i in range(3)],
      axis=-1)
  if "Median" in pre_processing:
    preprocessed_image = cv2.medianBlur(preprocessed_image, 5)
  if "Bilateral" in pre_processing:
    preprocessed_image = cv2.bilateralFilter(preprocessed_image, 9, 75, 75)
  predictor.set_image(preprocessed_image)
  return (
      img, preprocessed_image, preprocessed_image, [],
      (preprocessed_image, []))


with gr.Blocks() as demo:
  # store Sam predictor
  predictor = gr.State(value=get_sam_predictor())
  # store original image without points, default None
  selected_points = gr.State(value=[])
  original_image = gr.State(value=None)
  preprocessed_image = gr.State(value=None)
  # title
  with gr.Row():
    gr.Markdown("# Segment anything at home 🏠\n"
                "Barebone image-only segment anything at home.")
    with gr.Row():
      # select model
      model_type = gr.Dropdown(
          choices=["vit_b", "vit_l", "vit_h"],
          value='vit_b',
          label="Select Model")
      # select device
      device = gr.Dropdown(
          choices=["cpu", "cuda"],
          value='cuda',
          label="Select Device")

      model_type.change(
          get_sam_predictor,
          [model_type, device, original_image],
          [predictor]
      )
      device.change(
          get_sam_predictor,
          [model_type, device, original_image],
          [predictor]
      )

  # Segment image
  with gr.Row():
    with gr.Column():
      # input image
      input_image = gr.Image(type="numpy", label='Input image')
      # Set height of widget to 500 pixels
      input_image.style(height=600)
      with gr.Accordion(label="Image options"):
        with gr.Row():
          rgb_checkbox_group = gr.CheckboxGroup(
              ["R", "G", "B"],
              label="Input image",
              info="Select which channel(s) to use as input image.",
              value=["R", "G", "B"]
          )
          pre_processing_checkbox_group = gr.CheckboxGroup(
              ["CLAHE", "Median", "Bilateral"],
              label="Pre-processing",
              info="Whether to apply post-processing.",
          )

      # point prompt
      undo_button = gr.Button('Undo point')
      with gr.Row():
        fg_bg_radio = gr.Radio(
            ['foreground_point', 'background_point'],
            info="Select foreground or background point",
            label='Point label')
        with gr.Column():
          multi_object = gr.CheckboxGroup(
              ['Multi-object'],
              info="Whether each point correspond to a single object?",
              label='Multi-object',
              default=False
          )
      gr.Markdown('You can click on the image to select points prompt. '
                  'Default: `foreground_point`.')

    # show only mask
    with gr.Column():
      output_mask = gr.AnnotatedImage(show_progress='minimal')
      output_file = gr.File(
          label='Save output mask',
          interactive=False,
          description='Save output mask to local file system.')

  input_image.upload(
      reset_image,
      [predictor, input_image, rgb_checkbox_group,
       pre_processing_checkbox_group],
      [original_image, preprocessed_image, input_image, selected_points,
       output_mask]
  )
  rgb_checkbox_group.change(
      reset_image,
      [predictor, original_image, rgb_checkbox_group,
       pre_processing_checkbox_group],
      [original_image, preprocessed_image, input_image, selected_points,
       output_mask])
  pre_processing_checkbox_group.change(
      reset_image,
      [predictor, original_image, rgb_checkbox_group,
       pre_processing_checkbox_group],
      [original_image, preprocessed_image, input_image, selected_points,
       output_mask])

  undo_button.click(
      undo_points,
      [predictor, preprocessed_image, multi_object, selected_points],
      [input_image, output_mask, output_file]
  )

  input_image.select(
      select_point,
      [predictor, preprocessed_image, multi_object, selected_points,
       fg_bg_radio],
      [input_image, output_mask, output_file])


demo.queue().launch(debug=True, enable_queue=True)
