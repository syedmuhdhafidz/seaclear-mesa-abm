"""
SeaClear Mesa Visualization
---------------------------

Browser-based visualization for the simplified
parallel mapping-and-collection SeaClear ABM.

Run with:

    solara run app.py
"""

import numpy as np
import solara

from matplotlib.figure import Figure

from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
)

from mesa.visualization.utils import (
    update_counter,
)

from seaclear.agents import (
    LitterState,
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
        value=30,
        min=5,
        max=100,
        step=5,
    ),

    "num_marine_life": Slider(
        label="Marine Life Agents",
        value=10,
        min=0,
        max=30,
        step=1,
    ),

    "rov_speed": Slider(
        label="ObsROV Speed (m/s)",
        value=1.0,
        min=0.5,
        max=3.0,
        step=0.1,
    ),

    "collection_rov_speed": Slider(
        label="Collection ROV Speed (m/s)",
        value=1.2,
        min=0.5,
        max=3.0,
        step=0.1,
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

    "large_litter_probability": Slider(
        label="Large Litter Probability",
        value=0.10,
        min=0.00,
        max=0.50,
        step=0.05,
    ),

    "collection_success_probability": Slider(
        label="Collection Success Probability",
        value=0.90,
        min=0.10,
        max=1.00,
        step=0.05,
    ),

    "step_size": 0.5,
    "pickup_radius": 0.75,
    "basket_docking_radius": 1.0,
}


# ============================================================
# 2. INITIAL MODEL
# ============================================================

model = SeaClearModel()


# ============================================================
# 3. CUSTOM VISUALIZATION
# ============================================================

@solara.component
def SeaClearPlot(model):

    update_counter.get()

    fig = Figure(
        figsize=(11, 8)
    )

    ax = fig.subplots()

    fig.subplots_adjust(
        right=0.76,
        bottom=0.18,
    )

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
        (
            "Parallel SeaClear ABM"
            f"  |  t = "
            f"{model.simulation_time / 60.0:.1f} min"
            f"  |  in basket: "
            f"{model.basket.collected_count}"
        ),
        fontsize=14,
        fontweight="bold",
    )

    ax.grid(
        True,
        alpha=0.20,
    )

    # --------------------------------------------------------
    # Planned ObsROV path
    # --------------------------------------------------------

    ax.plot(
        model.rov_path[:, 0],
        model.rov_path[:, 1],
        linestyle="--",
        linewidth=1.0,
        alpha=0.25,
        label="ObsROV Planned Path",
    )

    # --------------------------------------------------------
    # ObsROV travelled path
    # --------------------------------------------------------

    observation_history = np.array(
        model.rov.history
    )

    if len(observation_history) > 1:

        ax.plot(
            observation_history[:, 0],
            observation_history[:, 1],
            linewidth=1.5,
            alpha=0.65,
            label="ObsROV Travelled",
        )

    # --------------------------------------------------------
    # Collection ROV travelled path
    # --------------------------------------------------------

    collection_history = np.array(
        model.collection_rov.history
    )

    if len(collection_history) > 1:

        ax.plot(
            collection_history[:, 0],
            collection_history[:, 1],
            linewidth=1.5,
            alpha=0.55,
            linestyle=":",
            label="ColROV Travelled",
        )

    # --------------------------------------------------------
    # Litter states
    # --------------------------------------------------------

    pending = (
        model.get_pending_litter()
    )

    collected = (
        model.get_collected_litter()
    )

    failed = (
        model.get_failed_litter()
    )

    large = (
        model.get_large_flagged_litter()
    )

    # Undiscovered litter is intentionally hidden.

    if pending:

        positions = np.array(
            [
                litter.position
                for litter
                in pending
            ]
        )

        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            marker="o",
            s=55,
            color="0.30",
            label="Litter: on map / pending",
        )

    if collected:

        positions = np.array(
            [
                litter.position
                for litter
                in collected
            ]
        )

        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            marker="o",
            s=55,
            color="tab:blue",
            label="Litter: collected",
        )

    if failed:

        positions = np.array(
            [
                litter.position
                for litter
                in failed
            ]
        )

        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            marker="x",
            s=65,
            linewidths=2,
            color="tab:orange",
            label="Litter: failed",
        )

    if large:

        positions = np.array(
            [
                litter.position
                for litter
                in large
            ]
        )

        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            marker="s",
            s=65,
            color="#E69F00",
            label="Large litter: flagged",
        )

    # --------------------------------------------------------
    # Marine life
    # --------------------------------------------------------

    if model.marine_life_agents:

        marine_positions = np.array(
            [
                agent.position
                for agent
                in model.marine_life_agents
            ]
        )

        ax.scatter(
            marine_positions[:, 0],
            marine_positions[:, 1],
            marker="^",
            s=50,
            color="#1B9E77",
            label="Marine life",
        )

    # --------------------------------------------------------
    # Basket
    # --------------------------------------------------------

    ax.scatter(
        [model.basket.position[0]],
        [model.basket.position[1]],
        marker="*",
        s=280,
        color="black",
        label="Collection Basket",
        zorder=7,
    )

    # --------------------------------------------------------
    # Observation ROV
    # --------------------------------------------------------

    ax.scatter(
        [model.rov.position[0]],
        [model.rov.position[1]],
        marker="D",
        s=120,
        color="black",
        label="miniTortuga / ObsROV",
        zorder=8,
    )

    # --------------------------------------------------------
    # Collection ROV
    # --------------------------------------------------------

    ax.scatter(
        [model.collection_rov.position[0]],
        [model.collection_rov.position[1]],
        marker="P",
        s=150,
        color="tab:blue",
        edgecolors="black",
        linewidths=1.2,
        label="Tortuga / ColROV",
        zorder=9,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    status_text = (

        "MISSION STATUS\n"
        "--------------------\n"
        f"ABM step:      "
        f"{model.step_count}\n"
        f"Time:          "
        f"{model.simulation_time:.1f} s\n"
        f"Map progress:  "
        f"{model.get_progress():.1f}%\n\n"

        "LITTER STATUS\n"
        "--------------------\n"
        f"Total:         "
        f"{model.num_litter}\n"
        f"Undiscovered:  "
        f"{model.get_undiscovered_count()}\n"
        f"Pending:       "
        f"{model.get_pending_count()}\n"
        f"Large flagged: "
        f"{model.get_large_flagged_count()}\n"
        f"Failed:        "
        f"{model.get_failed_count()}\n"
        f"Collected:     "
        f"{model.get_collected_count()}\n"
        f"In basket:     "
        f"{model.basket.collected_count}\n\n"

        "ROV DISTANCE\n"
        "--------------------\n"
        f"ObsROV:        "
        f"{model.rov.distance_travelled:.1f} m\n"
        f"Collection:    "
        f"{model.collection_rov.distance_travelled:.1f} m\n"
        f"Collector mode:"
        f" {model.collection_rov.mode}"
    )

    ax.text(
        1.04,
        0.96,
        status_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=8.5,
        family="monospace",
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.10,
        ),
        ncol=2,
        fontsize=7.5,
    )

    solara.FigureMatplotlib(
        fig,
        format="png",
        bbox_inches="tight",
    )


# ============================================================
# 4. DATA PLOTS
# ============================================================

LitterStatusPlot = make_plot_component(
    {
        "Mapped Litter": "tab:blue",
        "Collected Litter": "tab:green",
        "Failed Litter": "tab:red",
        "Large Litter": "tab:orange",
    }
)

BasketPlot = make_plot_component(
    {
        "Basket Count": "tab:green",
    }
)


# ============================================================
# 5. SOLARA PAGE
# ============================================================

page = SolaraViz(

    model,

    components=[
        SeaClearPlot,
        LitterStatusPlot,
        BasketPlot,
    ],

    model_params=model_params,

    name=(
        "SeaClear Multi-Agent Model"
    ),
)

page