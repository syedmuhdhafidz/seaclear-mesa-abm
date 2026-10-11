"""
SeaClear Mesa Continuous-Space Model
------------------------------------

This model uses Mesa's ContinuousSpace to represent
a 2D underwater search environment.

Agents:
    - ObservationROVAgent
    - LitterAgent
"""

import math

import numpy as np

from mesa import Model
from mesa.datacollection import DataCollector

from mesa.experimental.continuous_space import (
    ContinuousSpace,
)

from seaclear.agents import (
    LitterAgent,
    ObservationROVAgent,
)


# ============================================================
# 1. PATH HELPER FUNCTIONS
# ============================================================


def interpolate_line(
    start,
    end,
    step_size,
):
    """
    Generate evenly spaced points between two coordinates.
    """

    start = np.array(
        start,
        dtype=float,
    )

    end = np.array(
        end,
        dtype=float,
    )

    distance = np.linalg.norm(
        end - start
    )

    if distance == 0:

        return np.array(
            [start]
        )

    number_of_steps = max(
        1,
        int(
            math.ceil(
                distance
                / step_size
            )
        ),
    )

    return np.linspace(
        start,
        end,
        number_of_steps + 1,
    )


def generate_lawnmower_path(
    width,
    height,
    track_spacing,
    step_size,
):
    """
    Generate a continuous lawnmower search trajectory.
    """

    path_segments = []

    y_positions = np.arange(
        track_spacing / 2.0,
        height,
        track_spacing,
    )

    if len(y_positions) == 0:

        raise ValueError(
            "Track spacing is too large "
            "for the selected environment."
        )

    current_position = np.array(
        [
            0.0,
            y_positions[0],
        ],
        dtype=float,
    )

    path_segments.append(
        np.array(
            [current_position]
        )
    )

    moving_right = True

    for row_index, y in enumerate(
        y_positions
    ):

        if moving_right:

            row_end = np.array(
                [
                    width,
                    y,
                ],
                dtype=float,
            )

        else:

            row_end = np.array(
                [
                    0.0,
                    y,
                ],
                dtype=float,
            )

        horizontal_segment = (
            interpolate_line(
                current_position,
                row_end,
                step_size,
            )
        )

        path_segments.append(
            horizontal_segment[1:]
        )

        current_position = (
            row_end.copy()
        )

        if (
            row_index
            < len(y_positions) - 1
        ):

            next_y = (
                y_positions[
                    row_index + 1
                ]
            )

            vertical_end = np.array(
                [
                    current_position[0],
                    next_y,
                ],
                dtype=float,
            )

            vertical_segment = (
                interpolate_line(
                    current_position,
                    vertical_end,
                    step_size,
                )
            )

            path_segments.append(
                vertical_segment[1:]
            )

            current_position = (
                vertical_end.copy()
            )

        moving_right = (
            not moving_right
        )

    return np.vstack(
        path_segments
    )


# ============================================================
# 2. SEACLEAR MODEL
# ============================================================


class SeaClearModel(Model):
    """
    Mesa ContinuousSpace model for the simplified
    SeaClear Observation ROV simulation.
    """

    def __init__(
        self,
        width=50.0,
        height=50.0,
        num_litter=20,
        rov_speed=1.0,
        track_spacing=5.0,
        step_size=0.5,
        detection_radius=3.0,
        max_detection_probability=0.95,
        detection_decay=0.18,
        rng=42,
    ):

        # ====================================================
        # Mesa model
        # ====================================================

        super().__init__(
            rng=rng
        )

        # ====================================================
        # Parameters
        # ====================================================

        self.width = float(
            width
        )

        self.height = float(
            height
        )

        self.num_litter = int(
            num_litter
        )

        self.rov_speed = float(
            rov_speed
        )

        self.track_spacing = float(
            track_spacing
        )

        self.step_size = float(
            step_size
        )

        self.detection_radius = float(
            detection_radius
        )

        self.max_detection_probability = float(
            max_detection_probability
        )

        self.detection_decay = float(
            detection_decay
        )

        # ====================================================
        # Simulation state
        # ====================================================

        self.simulation_time = 0.0

        self.step_count = 0

        self.running = True

        # ====================================================
        # Mesa ContinuousSpace
        # ====================================================

        self.space = ContinuousSpace(
            dimensions=[
                [
                    0.0,
                    self.width,
                ],
                [
                    0.0,
                    self.height,
                ],
            ],
            torus=False,
            random=self.random,
            n_agents=(
                self.num_litter + 1
            ),
        )

        # ====================================================
        # Generate ROV search path
        # ====================================================

        self.rov_path = (
            generate_lawnmower_path(
                width=(
                    self.width
                ),
                height=(
                    self.height
                ),
                track_spacing=(
                    self.track_spacing
                ),
                step_size=(
                    self.step_size
                ),
            )
        )

        # ====================================================
        # Create litter agents
        # ====================================================

        self.litter_agents = []

        for _ in range(
            self.num_litter
        ):

            x = self.random.uniform(
                0.0,
                self.width,
            )

            y = self.random.uniform(
                0.0,
                self.height,
            )

            litter = LitterAgent(
                model=self,
                space=self.space,
                position=[
                    x,
                    y,
                ],
            )

            self.litter_agents.append(
                litter
            )

        # ====================================================
        # Create Observation ROV
        # ====================================================

        self.rov = ObservationROVAgent(
            model=self,
            space=self.space,
            path=self.rov_path,
            speed=(
                self.rov_speed
            ),
            detection_radius=(
                self.detection_radius
            ),
            max_detection_probability=(
                self.max_detection_probability
            ),
            detection_decay=(
                self.detection_decay
            ),
        )


        # ====================================================
        # Mesa DataCollector
        # ====================================================

        self.datacollector = DataCollector(

            model_reporters={

                "Detected Litter": (
                    lambda model:
                    model.get_detected_count()
                ),

                "Undetected Litter": (
                    lambda model:
                    model.get_undetected_count()
                ),

                "Detection Rate": (
                    lambda model:
                    model.get_detection_rate()
                ),

                "Distance Travelled": (
                    lambda model:
                    model.rov.distance_travelled
                ),

                "Simulation Time": (
                    lambda model:
                    model.simulation_time
                ),

                "Mission Progress": (
                    lambda model:
                    model.get_progress()
                ),
            }
        )

        # Record initial state at Step 0
        self.datacollector.collect(
            self
        )


    # ========================================================
    # MODEL STEP
    # ========================================================

    def step(self):
        """
        Advance simulation by one ABM timestep.

        During every step:
            1. Advance the ROV.
            2. Update the simulation state.
            3. Record statistics with Mesa DataCollector.
            4. Stop when the ROV completes its path.
        """

        if not self.running:
            return

        self.step_count += 1

        self.rov.step()

        # Record model state after this step
        self.datacollector.collect(
            self
        )

        if self.rov.mission_complete:
            self.running = False


    # ========================================================
    # STATISTICS
    # ========================================================

    def get_detected_litter(
        self
    ):

        return [
            litter
            for litter
            in self.litter_agents
            if litter.detected
        ]


    def get_undetected_litter(
        self
    ):

        return [
            litter
            for litter
            in self.litter_agents
            if not litter.detected
        ]


    def get_detected_count(
        self
    ):

        return len(
            self.get_detected_litter()
        )


    def get_undetected_count(
        self
    ):

        return len(
            self.get_undetected_litter()
        )


    def get_detection_rate(
        self
    ):

        if self.num_litter == 0:

            return 0.0

        return (
            self.get_detected_count()
            / self.num_litter
            * 100.0
        )


    def get_progress(
        self
    ):

        total_path_steps = (
            len(self.rov.path) - 1
        )

        if total_path_steps <= 0:

            return 100.0

        return (
            self.rov.path_index
            / total_path_steps
            * 100.0
        )