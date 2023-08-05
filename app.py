from typing import List

import cv2
import gradio as gr
import numpy as np

from inference import SamPredictor, get_sam_predictor, run_inference

# points color and marker
COLORS = [(255, 0, 0), (0, 255, 0)]
MARKERS = [1, 5]


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
  print(multi_object)
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
  return img, (img, o_masks)


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
  return temp, (temp, o_masks)


# once user upload an image, the original image is stored in `original_image`
def reset_image(predictor, img):
  # when new image is uploaded, `selected_points` should be empty
  predictor.set_image(img)
  return img, [], (img, [])


with gr.Blocks() as demo:
  # store Sam predictor
  predictor = gr.State(value=get_sam_predictor())
  # store original image without points, default None
  selected_points = gr.State(value=[])
  original_image = gr.State(value=None)
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
      input_image.style(height=500)

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
    output_mask = gr.AnnotatedImage(show_progress='minimal')

  input_image.upload(
      reset_image,
      [predictor, input_image],
      [original_image, selected_points, output_mask]
  )

  undo_button.click(
      undo_points,
      [predictor, original_image, multi_object, selected_points],
      [input_image, output_mask]
  )

  input_image.select(
      select_point,
      [predictor, original_image, multi_object, selected_points, fg_bg_radio],
      [input_image, output_mask])

  multi_object.change(
      select_point,
      [predictor, original_image, multi_object, selected_points, fg_bg_radio],
      [input_image, output_mask])


demo.queue().launch(debug=True, enable_queue=True)
