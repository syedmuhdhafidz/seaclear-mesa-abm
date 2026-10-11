"""
SeaClear Mesa Continuous-Space Agents
-------------------------------------

Agents:
    - LitterAgent
    - ObservationROVAgent

Both agents use Mesa's ContinuousSpaceAgent so that their
positions are managed natively by Mesa ContinuousSpace.
"""

import math

import numpy as np

from mesa.experimental.continuous_space import (
    ContinuousSpaceAgent,
)


# ============================================================
# 1. LITTER AGENT
# ============================================================


class LitterAgent(ContinuousSpaceAgent):
    """
    Represents one litter object on the seabed.

    The litter agent is passive:
        - It remains stationary.
        - It stores its detection state.
        - It records detection time.
    """

    def __init__(
        self,
        model,
        space,
        position,
    ):
        """
        Create one litter agent.

        Parameters
        ----------
        model:
            Parent Mesa model.

        space:
            Mesa ContinuousSpace.

        position:
            Initial [x, y] position.
        """

        super().__init__(
            space,
            model,
        )

        self.position = np.array(
            position,
            dtype=float,
        )

        self.detected = False

        self.detection_time = None

        self.detected_by = None


    def mark_detected(
        self,
        detection_time,
        rov_id,
    ):
        """
        Change litter state from undetected to detected.
        """

        if not self.detected:

            self.detected = True

            self.detection_time = (
                detection_time
            )

            self.detected_by = (
                rov_id
            )


    def step(self):
        """
        Litter is stationary in this simplified model.
        """

        pass


# ============================================================
# 2. OBSERVATION ROV AGENT
# ============================================================


class ObservationROVAgent(
    ContinuousSpaceAgent
):
    """
    Active Observation ROV agent.

    Behaviour:
        1. Follow the lawnmower path.
        2. Query nearby agents from ContinuousSpace.
        3. Attempt probabilistic litter detection.
    """

    def __init__(
        self,
        model,
        space,
        path,
        speed,
        detection_radius,
        max_detection_probability,
        detection_decay,
    ):

        super().__init__(
            space,
            model,
        )

        self.path = np.array(
            path,
            dtype=float,
        )

        self.speed = speed

        self.detection_radius = (
            detection_radius
        )

        self.max_detection_probability = (
            max_detection_probability
        )

        self.detection_decay = (
            detection_decay
        )

        self.path_index = 0

        self.position = (
            self.path[0].copy()
        )

        self.previous_position = (
            self.position.copy()
        )

        self.distance_travelled = 0.0

        self.history = [
            self.position.copy()
        ]

        self.mission_complete = False


    # ========================================================
    # MOVEMENT
    # ========================================================

    def move(self):
        """
        Move one discrete path step along the lawnmower path.
        """

        if self.path_index >= len(self.path) - 1:

            self.mission_complete = True

            return

        self.previous_position = (
            self.position.copy()
        )

        self.path_index += 1

        new_position = (
            self.path[
                self.path_index
            ].copy()
        )

        movement_distance = (
            np.linalg.norm(
                new_position
                - self.previous_position
            )
        )

        # Mesa manages the position internally
        self.position = (
            new_position
        )

        self.distance_travelled += (
            movement_distance
        )

        self.history.append(
            self.position.copy()
        )

        movement_time = (
            movement_distance
            / self.speed
        )

        self.model.simulation_time += (
            movement_time
        )

        if (
            self.path_index
            >= len(self.path) - 1
        ):

            self.mission_complete = True


    # ========================================================
    # SENSOR MODEL
    # ========================================================

    def calculate_detection_probability(
        self,
        distance,
    ):
        """
        Calculate distance-dependent detection probability.

        P(d) = P_max * exp(-k * d)
        """

        if (
            distance
            > self.detection_radius
        ):

            return 0.0

        probability = (
            self.max_detection_probability
            * math.exp(
                -self.detection_decay
                * distance
            )
        )

        return max(
            0.0,
            min(
                1.0,
                probability,
            ),
        )


    # ========================================================
    # DETECTION
    # ========================================================

    def detect_litter(self):
        """
        Use Mesa ContinuousSpace to find nearby agents.

        Only LitterAgent instances are considered.
        """

        nearby_agents, distances = (
            self.get_neighbors_in_radius(
                radius=(
                    self.detection_radius
                )
            )
        )

        for agent, distance in zip(
            nearby_agents,
            distances,
        ):

            if not isinstance(
                agent,
                LitterAgent,
            ):
                continue

            if agent.detected:
                continue

            probability = (
                self.calculate_detection_probability(
                    distance
                )
            )

            random_value = (
                self.model.random.random()
            )

            if (
                random_value
                <= probability
            ):

                agent.mark_detected(
                    detection_time=(
                        self.model.simulation_time
                    ),
                    rov_id=(
                        self.unique_id
                    ),
                )


    # ========================================================
    # AGENT STEP
    # ========================================================

    def step(self):
        """
        Execute one complete Observation ROV step.
        """

        if self.mission_complete:
            return

        self.move()

        self.detect_litter()