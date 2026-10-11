"""
SeaClear Mesa Agents
--------------------

This file defines the agents used in the simplified
SeaClear Observation ROV simulation.

Agents:
    - LitterAgent
    - ObservationROVAgent
"""

import math

import numpy as np

from mesa import Agent


# ============================================================
# 1. LITTER AGENT
# ============================================================


class LitterAgent(Agent):
    """
    Represents one individual litter object on the seabed.

    The litter is a passive agent:
    - It does not move.
    - It stores its own position.
    - It stores whether it has been detected.
    - It records when it was detected.
    """

    def __init__(
        self,
        model,
        x,
        y,
    ):
        """
        Create one litter agent.

        Parameters
        ----------
        model:
            The Mesa model that owns this agent.

        x : float
            X-coordinate of the litter.

        y : float
            Y-coordinate of the litter.
        """

        super().__init__(model)

        self.position = np.array(
            [x, y],
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
        Change the litter state from undetected to detected.
        """

        if not self.detected:

            self.detected = True

            self.detection_time = detection_time

            self.detected_by = rov_id


    def step(self):
        """
        Litter is passive in this simplified model.

        Therefore, no action is performed during its step.
        """

        pass


# ============================================================
# 2. OBSERVATION ROV AGENT
# ============================================================


class ObservationROVAgent(Agent):
    """
    Represents the active Observation ROV agent.

    Behaviour:
        1. Follow the lawnmower search trajectory.
        2. Sense nearby litter agents.
        3. Calculate detection probability.
        4. Detect litter probabilistically.
    """

    def __init__(
        self,
        model,
        path,
        speed,
        detection_radius,
        max_detection_probability,
        detection_decay,
    ):
        """
        Create the Observation ROV agent.

        Parameters
        ----------
        model:
            Mesa model that owns this agent.

        path:
            Sequence of (x, y) coordinates defining
            the lawnmower trajectory.

        speed : float
            ROV speed in metres per second.

        detection_radius : float
            Maximum sensor detection distance.

        max_detection_probability : float
            Maximum probability of detecting litter.

        detection_decay : float
            Controls how detection probability decreases
            with distance.
        """

        super().__init__(model)

        self.path = np.array(
            path,
            dtype=float,
        )

        self.speed = speed

        self.detection_radius = detection_radius

        self.max_detection_probability = (
            max_detection_probability
        )

        self.detection_decay = detection_decay

        self.path_index = 0

        self.position = self.path[0].copy()

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
        Move one step along the planned lawnmower trajectory.
        """

        if self.path_index >= len(self.path) - 1:

            self.mission_complete = True

            return

        self.previous_position = (
            self.position.copy()
        )

        self.path_index += 1

        self.position = (
            self.path[self.path_index].copy()
        )

        movement_distance = np.linalg.norm(
            self.position
            - self.previous_position
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

        if self.path_index >= len(self.path) - 1:

            self.mission_complete = True


    # ========================================================
    # SENSOR MODEL
    # ========================================================

    def calculate_detection_probability(
        self,
        distance,
    ):
        """
        Calculate probability of detecting litter.

        Formula:

            P(d) = P_max * exp(-k * d)

        where:

            d     = distance to litter
            P_max = maximum detection probability
            k     = detection decay coefficient
        """

        if distance > self.detection_radius:

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
    # DETECTION BEHAVIOUR
    # ========================================================

    def detect_litter(self):
        """
        Inspect all litter agents.

        If a litter agent is within the sensor radius,
        attempt detection using the probabilistic
        sensor model.
        """

        for litter in self.model.litter_agents:

            if litter.detected:
                continue

            distance = np.linalg.norm(
                litter.position
                - self.position
            )

            if distance <= self.detection_radius:

                probability = (
                    self.calculate_detection_probability(
                        distance
                    )
                )

                random_value = (
                    self.model.random.random()
                )

                if random_value <= probability:

                    litter.mark_detected(
                        detection_time=(
                            self.model.simulation_time
                        ),
                        rov_id=self.unique_id,
                    )


    # ========================================================
    # AGENT STEP
    # ========================================================

    def step(self):
        """
        Execute one complete ROV agent step.

        Order:
            1. Move
            2. Sense/detect litter
        """

        if self.mission_complete:
            return

        self.move()

        self.detect_litter()