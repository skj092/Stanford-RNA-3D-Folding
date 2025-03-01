import plotly.graph_objects as go
import numpy as np


config = {
    "seed": 0,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 384,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "mixed_precision": "bf16",
    "model_config_path": "../working/configs/pairwise.yaml",  # Adjust path as needed
    "epochs": 10,
    "cos_epoch": 5,
    "loss_power_scale": 1.0,
    "max_cycles": 1,
    "grad_clip": 0.1,
    "gradient_accumulation_steps": 1,
    "d_clamp": 30,
    "max_len_filter": 9999999,
    "min_len_filter": 10,
    "structural_violation_epoch": 50,
    "balance_weight": False,
    "model_path": "./models/RibonanzaNet.pt"
}


def get_plot(xyz):
    for _ in range(2):  # plot twice because it doesnt show up on first try for some reason
        # Extract columns
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # Create the 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=z,  # Coloring based on z-value
                colorscale='Viridis',  # Choose a colorscale
                opacity=0.8
            )
        )])

        # Customize layout
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            ),
            title="3D Scatter Plot"
        )
    fig.show()
    # fig.write_image("3d_scatter_plot.png")
