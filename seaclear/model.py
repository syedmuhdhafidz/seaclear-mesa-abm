"""
SeaClear Mesa Model
-------------------

This file defines the Mesa model that coordinates
the simplified SeaClear Observation ROV simulation.

The model:
    - Creates the 2D seabed environment
    - Generates random litter agents
    - Generates the lawnmower search path
    - Creates the Observation ROV agent
    - Advances the simulation one step at a time
    - Tracks mission statistics
"""

import math
import numpy as np

from mesa import Model

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
    Generate points along a straight line between two
    coordinates.

    Parameters
    ----------
    start:
        Starting coordinate.

    end:
        Ending coordinate.

    step_size:
        Approximate distance between points.

    Returns
    -------
    numpy.ndarray
        Array of XY coordinates.
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
                distance / step_size
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

    The ROV:
        1. Travels horizontally across the seabed
        2. Moves upward
        3. Travels horizontally in the opposite direction
        4. Repeats until the search area is covered
    """

    path_segments = []

    y_positions = np.arange(
        track_spacing / 2.0,
        height,
        track_spacing,
    )

    if len(y_positions) == 0:
        raise ValueError(
            "TRACK_SPACING is too large "
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

        # ----------------------------------------------------
        # Determine horizontal direction
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Generate horizontal section
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Move vertically to next row
        # ----------------------------------------------------

        if row_index < len(y_positions) - 1:

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
    Mesa model for the simplified SeaClear Observation
    ROV litter-mapping simulation.

    Parameters
    ----------
    width : float
        Width of simulated seabed in metres.

    height : float
        Height of simulated seabed in metres.

    num_litter : int
        Number of litter agents.

    rov_speed : float
        ROV speed in metres per second.

    track_spacing : float
        Distance between lawnmower search tracks.

    step_size : float
        Distance between path coordinates.

    detection_radius : float
        Maximum litter sensing distance.

    max_detection_probability : float
        Maximum probability of detecting litter.

    detection_decay : float
        Controls how detection probability decreases
        with distance.

    seed : int
        Random seed.
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
        # Initialise Mesa Model
        # ====================================================

        super().__init__(
            rng=rng
        )

        # ====================================================
        # Store model parameters
        # ====================================================

        self.width = width

        self.height = height

        self.num_litter = num_litter

        self.rov_speed = rov_speed

        self.track_spacing = (
            track_spacing
        )

        self.step_size = step_size

        self.detection_radius = (
            detection_radius
        )

        self.max_detection_probability = (
            max_detection_probability
        )

        self.detection_decay = (
            detection_decay
        )

        # ====================================================
        # Simulation state
        # ====================================================

        self.simulation_time = 0.0

        self.step_count = 0

        self.running = True

        # ====================================================
        # Generate lawnmower trajectory
        # ====================================================

        self.rov_path = (
            generate_lawnmower_path(
                width=self.width,
                height=self.height,
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
                x=x,
                y=y,
            )

            self.litter_agents.append(
                litter
            )

        # ====================================================
        # Create Observation ROV agent
        # ====================================================

        self.rov = ObservationROVAgent(
            model=self,
            path=self.rov_path,
            speed=self.rov_speed,
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


    # ========================================================
    # MODEL STEP
    # ========================================================

    def step(self):
        """
        Advance the model by one ABM timestep.

        In each model step:

            1. Increment the model step counter.
            2. Execute the ROV behaviour.
            3. Check whether the mission has finished.
        """

        if not self.running:
            return

        self.step_count += 1

        # ----------------------------------------------------
        # Activate Observation ROV
        # ----------------------------------------------------

        self.rov.step()

        # ----------------------------------------------------
        # Check termination condition
        # ----------------------------------------------------

        if self.rov.mission_complete:

            self.running = False


    # ========================================================
    # MODEL STATISTICS
    # ========================================================

    def get_detected_litter(
        self
    ):
        """
        Return all detected litter agents.
        """

        return [
            litter
            for litter
            in self.litter_agents
            if litter.detected
        ]


    def get_undetected_litter(
        self
    ):
        """
        Return all undetected litter agents.
        """

        return [
            litter
            for litter
            in self.litter_agents
            if not litter.detected
        ]


    def get_detected_count(
        self
    ):
        """
        Return number of detected litter agents.
        """

        return len(
            self.get_detected_litter()
        )


    def get_undetected_count(
        self
    ):
        """
        Return number of undetected litter agents.
        """

        return len(
            self.get_undetected_litter()
        )


    def get_detection_rate(
        self
    ):
        """
        Return litter detection percentage.
        """

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
        """
        Return mission completion percentage.
        """

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