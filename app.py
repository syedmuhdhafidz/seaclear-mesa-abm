"""
SeaClear Mesa Visualization
---------------------------

Browser-based visualization for the simplified
SeaClear Observation ROV Agent-Based Model.

Run with:

    solara run app.py
"""

import numpy as np
import solara

from matplotlib.figure import Figure
from matplotlib.patches import Circle

from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
)

from mesa.visualization.utils import (
    update_counter,
)

from seaclear.model import (
    SeaClearModel,
)


# ============================================================
# 1. MODEL PARAMETERS
# ============================================================

model_params = {

    "rng": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },

    "width": Slider(
        label="Area Width (m)",
        value=50.0,
        min=20.0,
        max=100.0,
        step=5.0,
    ),

    "height": Slider(
        label="Area Height (m)",
        value=50.0,
        min=20.0,
        max=100.0,
        step=5.0,
    ),

    "num_litter": Slider(
        label="Number of Litter Agents",
        value=20,
        min=5,
        max=100,
        step=5,
    ),

    "rov_speed": Slider(
        label="ROV Speed (m/s)",
        value=1.0,
        min=0.5,
        max=5.0,
        step=0.5,
    ),

    "track_spacing": Slider(
        label="Track Spacing (m)",
        value=5.0,
        min=2.0,
        max=10.0,
        step=0.5,
    ),

    "detection_radius": Slider(
        label="Detection Radius (m)",
        value=3.0,
        min=1.0,
        max=10.0,
        step=0.5,
    ),

    "max_detection_probability": Slider(
        label="Max Detection Probability",
        value=0.95,
        min=0.10,
        max=1.00,
        step=0.05,
    ),

    "detection_decay": Slider(
        label="Detection Decay",
        value=0.18,
        min=0.01,
        max=1.00,
        step=0.01,
    ),

    # Fixed for now
    "step_size": 0.5,
}


# ============================================================
# 2. INITIAL MODEL
# ============================================================

model = SeaClearModel()


# ============================================================
# 3. CUSTOM SEACLEAR VISUALIZATION
# ============================================================

@solara.component
def SeaClearPlot(model):
    """
    Display the current state of the SeaClear simulation.

    Shows:
        - Search area
        - Planned lawnmower path
        - Travelled ROV trajectory
        - Observation ROV
        - Detection radius
        - Detected litter
        - Undetected litter
        - Mission statistics
    """

    # --------------------------------------------------------
    # IMPORTANT:
    # This tells Solara/Mesa that this component depends on
    # the simulation update counter.
    #
    # Every model step changes the counter and therefore
    # triggers this component to redraw.
    # --------------------------------------------------------

    update_counter.get()

    # --------------------------------------------------------
    # Create Matplotlib figure
    # --------------------------------------------------------

    fig = Figure(
        figsize=(10, 7)
    )

    ax = fig.subplots()

    # Leave space on right side for statistics
    fig.subplots_adjust(
        right=0.75,
        bottom=0.18,
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        model.width,
    )

    ax.set_ylim(
        0,
        model.height,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(
        "X Position (m)"
    )

    ax.set_ylabel(
        "Y Position (m)"
    )

    ax.set_title(
        "SeaClear Observation ROV Search",
        fontsize=14,
        fontweight="bold",
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    # --------------------------------------------------------
    # Planned lawnmower path
    # --------------------------------------------------------

    ax.plot(
        model.rov_path[:, 0],
        model.rov_path[:, 1],
        linestyle="--",
        linewidth=1.0,
        alpha=0.4,
        label="Planned Search Path",
    )

    # --------------------------------------------------------
    # ROV travelled path
    # --------------------------------------------------------

    history = np.array(
        model.rov.history
    )

    if len(history) > 1:

        ax.plot(
            history[:, 0],
            history[:, 1],
            linewidth=2.0,
            alpha=0.8,
            label="ROV Travelled Path",
        )

    # --------------------------------------------------------
    # Litter agents
    # --------------------------------------------------------

    detected_litter = (
        model.get_detected_litter()
    )

    undetected_litter = (
        model.get_undetected_litter()
    )

    # --------------------------------------------------------
    # Undetected litter
    # --------------------------------------------------------

    if len(undetected_litter) > 0:

        undetected_positions = np.array(
            [
                litter.position
                for litter
                in undetected_litter
            ]
        )

        ax.scatter(
            undetected_positions[:, 0],
            undetected_positions[:, 1],
            marker="x",
            s=70,
            linewidths=2,
            label="Undetected Litter",
        )

    # --------------------------------------------------------
    # Detected litter
    # --------------------------------------------------------

    if len(detected_litter) > 0:

        detected_positions = np.array(
            [
                litter.position
                for litter
                in detected_litter
            ]
        )

        ax.scatter(
            detected_positions[:, 0],
            detected_positions[:, 1],
            marker="o",
            s=70,
            label="Detected Litter",
        )

    # --------------------------------------------------------
    # Observation ROV
    # --------------------------------------------------------

    rov_x = model.rov.position[0]
    rov_y = model.rov.position[1]

    ax.scatter(
        [rov_x],
        [rov_y],
        marker="o",
        s=130,
        label="Observation ROV",
        zorder=5,
    )

    # --------------------------------------------------------
    # Sensor detection radius
    # --------------------------------------------------------

    sensor_circle = Circle(
        (
            rov_x,
            rov_y,
        ),
        radius=model.detection_radius,
        fill=False,
        linestyle=":",
        linewidth=1.5,
        alpha=0.9,
        label="Detection Radius",
    )

    ax.add_patch(
        sensor_circle
    )

    # --------------------------------------------------------
    # Mission statistics
    # --------------------------------------------------------

    detected_count = (
        model.get_detected_count()
    )

    undetected_count = (
        model.get_undetected_count()
    )

    detection_rate = (
        model.get_detection_rate()
    )

    progress = (
        model.get_progress()
    )

    status_text = (

        "MISSION STATUS\n"
        "--------------------\n"

        f"ABM Step:     "
        f"{model.step_count}\n"

        f"Progress:     "
        f"{progress:.1f}%\n"

        f"Time:         "
        f"{model.simulation_time:.1f} s\n"

        f"Distance:     "
        f"{model.rov.distance_travelled:.1f} m\n\n"

        "LITTER MAPPING\n"
        "--------------------\n"

        f"Total:        "
        f"{model.num_litter}\n"

        f"Detected:     "
        f"{detected_count}\n"

        f"Undetected:   "
        f"{undetected_count}\n"

        f"Detection:    "
        f"{detection_rate:.1f}%"
    )

    ax.text(
        1.04,
        0.96,
        status_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=9,
        family="monospace",
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.10,
        ),
        ncol=2,
        fontsize=8,
    )

    # --------------------------------------------------------
    # Render in Solara
    # --------------------------------------------------------

    solara.FigureMatplotlib(
        fig,
        format="png",
        bbox_inches="tight",
    )


# ============================================================
# 4. DATA VISUALIZATION COMPONENTS
# ============================================================

DetectionPlot = make_plot_component(
    {
        "Detected Litter": "tab:green",
        "Undetected Litter": "tab:red",
    }
)

DetectionRatePlot = make_plot_component(
    {
        "Detection Rate": "tab:blue",
    }
)


# ============================================================
# 5. MESA SOLARA PAGE
# ============================================================

page = SolaraViz(

    model,

    components=[
        SeaClearPlot,
        DetectionPlot,
        DetectionRatePlot,
    ],

    model_params=model_params,

    name=(
        "SeaClear Agent-Based Model"
    ),
)

page