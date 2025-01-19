import numpy as np
import threading

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn  # Assuming you've defined MNISTNet class earlier
import torch
torch.set_num_threads(1)

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # Calculate the correct input size for fc1 dynamically
        self.fc1 = nn.Linear(1600, 128)  # Changed from 9216 to 1600
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x) # The issue was here
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

# Instantiate the model
mnist_model = MNISTModel()


# Load the saved state dictionary
mnist_model.load_state_dict(torch.load('models/mnist_model.pth'))

# Set the model to evaluation mode
mnist_model.eval()

# Now you can use the loaded model for inference
# For example, if you have an input tensor 'input_tensor':
# output = model(input_tensor)
# ... process the output ...

def predict_digit(pixel_array, model):
    # Ensure the pixel_array is of a supported type (e.g., uint8 or float32)
    pixel_array = pixel_array.astype(np.float32)  # Convert to float32 or uint8

    # Assuming pixel_array is a numpy array of shape (28, 28)
    # Convert numpy array to PyTorch tensor, add batch and channel dimensions
    processed_tensor = torch.from_numpy(center_image(pixel_array)).float().unsqueeze(0).unsqueeze(0) / 255

    # Ensure model is in evaluation mode
    model.eval()

    # Disable gradient computation for inference
    with torch.no_grad():
        # Forward pass through the model
        output = model(processed_tensor)
        # Get the index of the max log-probability
        prediction = output.argmax(dim=1).item()

    return prediction


from scipy.ndimage import center_of_mass, shift
def center_image(image):
    cy, cx = center_of_mass(image)
    shift_y = 14 - cy
    shift_x = 14 - cx
    centered_image = shift(image, shift=(shift_y, shift_x), mode='constant', cval=0)
    return centered_image


from PIL import Image
import numpy as np
import requests
from io import BytesIO

image_path = "https://raw.githubusercontent.com/adidror005/canvas/refs/heads/main/background_image_28.png"
response = requests.get(image_path)
response.raise_for_status()

img = Image.open(BytesIO(response.content))
img = np.array(img)

from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px

fig = px.imshow(img,height=420)
fig.update_layout(
    dragmode="drawopenpath",
    newshape=dict(line_color='red', line_width=45)  # Set color to red and width to 5
)
fig.update_layout(
    dragmode="drawopenpath",
    newshape=dict(line_color='red', line_width=45),
    margin=dict(l=0, r=0, t=0, b=0),  # Remove all margins
    xaxis=dict(visible=False, fixedrange=True),  # Hide x-axis
    yaxis=dict(visible=False, fixedrange=True)   # Hide y-axis
)


layout = html.Div([
    html.H1("Draw a Digit!", style={'text-align': 'center'}),
    dbc.Row([dbc.Col(dcc.Graph(
        id="graph-picture",
        figure=fig,
        config={"modeBarButtonsToAdd": ['eraseshape']},
    ), xs=11, sm=9, md=5, lg=5, xl=6)]),
    html.H1(id="annotations-data", style={'text-align': 'center'}),
    dbc.Button("Reset", id="reset-button", color="primary", style={'text-align': 'center'})
])

layout = dbc.Container([
    html.H1("Draw a Digit!", style={'text-align': 'center'}),
    dbc.Row(
        dbc.Col(
            dcc.Graph(
                id="graph-picture",
                figure=fig,
                config={"modeBarButtonsToAdd": ['eraseshape']},
                style={
                     "padding": "0",
                     "margin": "0",
        "width": "100%",
        "height": "100%",
                }
            ),
            width={"size": 12, "offset": 0}  # Full width on all screens
        ),
        justify="left"
    ),
    html.H1(id="annotations-data", style={'text-align': 'center'}),
    dbc.Row(
        dbc.Col(
            dbc.Button("Reset", id="reset-button", color="primary"),
            width={"size": 12},
            className="text-center mt-3"
        )
    )
], fluid=True)


from dash import Input, Output, callback
import json
import numpy as np

@callback(
    Output("annotations-data", "children"),
    Input("graph-picture", "relayoutData"),
    prevent_initial_call=True
)
def on_new_annotation(relayout_data):
    global mnist_model
    if "shapes" in relayout_data:
        shapes = relayout_data["shapes"]
        global debug_shapes
        debug_shapes = shapes
        path_data_list = [shape['path'] for shape in shapes]
        pixel_array = path_to_pixel_matrix(path_data_list)
        with open("example.txt", "a") as file:
            file.write('B4 PRED')
        pred = predict_digit(pixel_array,mnist_model)
        with open("example.txt", "a") as file:
            file.write(f"After PRED {pred}")


        return f"You drew a {pred}"
    return "No drawing yet!"


import numpy as np
import re
from PIL import Image, ImageDraw
def path_to_pixel_matrix(path_data_list):
  return np.sum(
            [single_path_to_pixel_matrix(path, 28, 28) for path in path_data_list], axis=0
        )

def single_path_to_pixel_matrix(path_data, width, height):
    """Converts an SVG path data string to a pixel matrix (binary image).

    Args:
        path_data: The SVG path data string.
        width: The width of the output image.
        height: The height of the output image.

    Returns:
        A NumPy array representing the pixel matrix (0 for background, 1 for path).
    """

    # Create a blank image
    img = Image.new("L", (width, height), color=0)  # "L" mode for grayscale
    draw = ImageDraw.Draw(img)

    # Split the path data into commands and coordinates
    commands = re.findall(r"[MLHVCSQTAZ]", path_data)
    coordinates = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", path_data)
    coordinates = [float(coord) for coord in coordinates]

    # Initialize the points array
    points = []
    current_point = (0, 0)

    # Iterate through commands and coordinates to extract points
    coord_index = 0
    for command in commands:
        if command in "ML":  # Moveto or Lineto
            num_coords = 2
            x, y = coordinates[coord_index : coord_index + num_coords]
            points.append((x, y))
            current_point = (x, y)
            coord_index += num_coords
        # Add handling for other commands if needed

    # Draw the path on the image
    if points:
        draw.line(points, fill=255, width=3)  # Draw line with width 2

    # Convert the image to a NumPy array
    pixel_matrix = np.array(img)

    return pixel_matrix

