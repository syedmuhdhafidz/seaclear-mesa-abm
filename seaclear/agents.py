"""
SeaClear Mesa Continuous-Space Agents
-------------------------------------

Agents:
    - LitterAgent
    - ObservationROVAgent
    - CollectionROVAgent
    - BasketAgent
    - MarineLifeAgent

All spatial agents use Mesa's ContinuousSpaceAgent.
"""

import math
from enum import Enum

import numpy as np

from mesa.experimental.continuous_space import ContinuousSpaceAgent


# ============================================================
# 1. LITTER STATE
# ============================================================


class LitterState(Enum):
    """
    Lifecycle states for a litter object.
    """

    UNDISCOVERED = "undiscovered"
    ON_MAP = "on_map"
    PICKED_UP = "picked_up"
    COLLECTED = "collected"
    FAILED = "failed"


# ============================================================
# 2. LITTER AGENT
# ============================================================


class LitterAgent(ContinuousSpaceAgent):
    """
    Represents one litter object on the seabed.

    Litter is stationary, but its state changes through
    interaction with the Observation and Collection ROVs.
    """

    def __init__(
        self,
        model,
        space,
        position,
        is_large=False,
    ):
        super().__init__(
            space,
            model,
        )

        self.position = np.array(
            position,
            dtype=float,
        )

        self.state = LitterState.UNDISCOVERED

        # Large litter can be mapped but is not collected
        # by the Collection ROV in this simplified model.
        self.is_large = bool(is_large)

        self.detection_time = None
        self.detected_by = None

        self.collection_time = None
        self.collected_by = None

        self.collection_attempts = 0


    def mark_detected(
        self,
        detection_time,
        rov_id,
    ):
        """
        Add litter to the litter map.
        """

        if self.state == LitterState.UNDISCOVERED:

            self.state = LitterState.ON_MAP

            self.detection_time = detection_time

            self.detected_by = rov_id


    def mark_picked_up(
        self,
        rov_id,
    ):
        """
        Mark a successful pickup.
        """

        if self.state == LitterState.ON_MAP:

            self.state = LitterState.PICKED_UP

            self.collected_by = rov_id


    def mark_collected(
        self,
        collection_time,
    ):
        """
        Mark litter as deposited in the basket.
        """

        self.state = LitterState.COLLECTED

        self.collection_time = collection_time


    def mark_failed(self):
        """
        Mark a failed collection attempt.
        """

        self.state = LitterState.FAILED


    def step(self):
        """
        Litter has no autonomous movement.
        """

        pass


# ============================================================
# 3. BASKET AGENT
# ============================================================


class BasketAgent(ContinuousSpaceAgent):
    """
    Represents the stationary collection basket.
    """

    def __init__(
        self,
        model,
        space,
        position,
    ):
        super().__init__(
            space,
            model,
        )

        self.position = np.array(
            position,
            dtype=float,
        )

        self.collected_count = 0


    def deposit(self):
        """
        Record one deposited litter object.
        """

        self.collected_count += 1


    def step(self):
        pass


# ============================================================
# 4. MARINE LIFE AGENT
# ============================================================


class MarineLifeAgent(ContinuousSpaceAgent):
    """
    Represents marine life within the search environment.

    In the current simplified version, marine-life agents
    perform a slow random walk. They do not yet affect
    robot navigation.
    """

    def __init__(
        self,
        model,
        space,
        position,
        speed=0.10,
    ):
        super().__init__(
            space,
            model,
        )

        self.position = np.array(
            position,
            dtype=float,
        )

        self.speed = float(speed)


    def step(self):
        """
        Perform a bounded random walk.
        """

        angle = self.model.random.uniform(
            0.0,
            2.0 * math.pi,
        )

        displacement = np.array(
            [
                math.cos(angle),
                math.sin(angle),
            ]
        ) * self.speed

        new_position = (
            self.position
            + displacement
        )

        # Keep marine life inside the environment.
        new_position[0] = np.clip(
            new_position[0],
            0.0,
            self.model.width,
        )

        new_position[1] = np.clip(
            new_position[1],
            0.0,
            self.model.height,
        )

        self.position = new_position


# ============================================================
# 5. OBSERVATION ROV AGENT
# ============================================================


class ObservationROVAgent(
    ContinuousSpaceAgent
):
    """
    miniTortuga / Observation ROV.

    Behaviour:
        1. Follow lawnmower trajectory.
        2. Find nearby litter.
        3. Perform probabilistic detection.
        4. Add successfully detected litter to the map.
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

        self.speed = float(speed)

        self.detection_radius = float(
            detection_radius
        )

        self.max_detection_probability = float(
            max_detection_probability
        )

        self.detection_decay = float(
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


    def move(self):
        """
        Move one waypoint increment along the lawnmower path.
        """

        if (
            self.path_index
            >= len(self.path) - 1
        ):
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

        movement_distance = np.linalg.norm(
            new_position
            - self.previous_position
        )

        self.position = new_position

        self.distance_travelled += (
            movement_distance
        )

        self.history.append(
            self.position.copy()
        )

        if (
            self.path_index
            >= len(self.path) - 1
        ):
            self.mission_complete = True


    def calculate_detection_probability(
        self,
        distance,
    ):
        """
        P(d) = Pmax * exp(-k * d)
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


    def detect_litter(self):
        """
        Detect previously undiscovered litter within
        the sensor radius.
        """

        nearby_agents, distances = (
            self.get_neighbors_in_radius(
                radius=self.detection_radius
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

            if (
                agent.state
                != LitterState.UNDISCOVERED
            ):
                continue

            probability = (
                self.calculate_detection_probability(
                    distance
                )
            )

            if (
                self.model.random.random()
                <= probability
            ):

                agent.mark_detected(
                    detection_time=(
                        self.model.simulation_time
                    ),
                    rov_id=self.unique_id,
                )


    def step(self):
        """
        One Observation ROV action cycle.
        """

        if self.mission_complete:
            return

        self.move()
        self.detect_litter()


# ============================================================
# 6. COLLECTION ROV AGENT
# ============================================================


class CollectionROVAgent(
    ContinuousSpaceAgent
):
    """
    Tortuga / Collection ROV.

    Simplified behaviour:
        IDLE
          -> select nearest mapped litter
        TO_TARGET
          -> navigate to litter
          -> attempt pickup
        RETURN_TO_BASKET
          -> navigate to basket
          -> deposit litter
          -> IDLE
    """

    IDLE = "idle"
    TO_TARGET = "to_target"
    RETURN_TO_BASKET = "return_to_basket"


    def __init__(
        self,
        model,
        space,
        position,
        speed,
        pickup_radius,
        collection_success_probability,
    ):
        super().__init__(
            space,
            model,
        )

        self.position = np.array(
            position,
            dtype=float,
        )

        self.speed = float(speed)

        self.pickup_radius = float(
            pickup_radius
        )

        self.collection_success_probability = float(
            collection_success_probability
        )

        self.mode = self.IDLE

        self.target = None

        self.carrying_litter = None

        self.distance_travelled = 0.0

        self.history = [
            self.position.copy()
        ]


    def get_available_targets(self):
        """
        Return mapped, normal-sized litter that can
        still be collected.
        """

        return [
            litter
            for litter
            in self.model.litter_agents
            if (
                litter.state
                == LitterState.ON_MAP
                and not litter.is_large
            )
        ]


    def select_target(self):
        """
        Select nearest available litter.
        """

        targets = (
            self.get_available_targets()
        )

        if not targets:

            self.target = None
            return

        self.target = min(
            targets,
            key=lambda litter:
                np.linalg.norm(
                    litter.position
                    - self.position
                ),
        )

        self.mode = self.TO_TARGET


    def move_towards(
        self,
        target_position,
    ):
        """
        Move toward a target using straight-line motion.
        """

        vector = (
            np.array(
                target_position,
                dtype=float,
            )
            - self.position
        )

        distance = np.linalg.norm(
            vector
        )

        if distance == 0:
            return 0.0

        maximum_step = (
            self.speed
            * self.model.time_step
        )

        movement_distance = min(
            maximum_step,
            distance,
        )

        direction = (
            vector
            / distance
        )

        self.position = (
            self.position
            + direction
            * movement_distance
        )

        self.distance_travelled += (
            movement_distance
        )

        self.history.append(
            self.position.copy()
        )

        return distance


    def attempt_collection(self):
        """
        Attempt to collect the selected litter.
        """

        if self.target is None:
            self.mode = self.IDLE
            return

        self.target.collection_attempts += 1

        successful = (
            self.model.random.random()
            <= self.collection_success_probability
        )

        if successful:

            self.target.mark_picked_up(
                rov_id=self.unique_id
            )

            self.carrying_litter = (
                self.target
            )

            self.target = None

            self.mode = (
                self.RETURN_TO_BASKET
            )

        else:

            self.target.mark_failed()

            self.target = None

            self.mode = self.IDLE


    def deposit_litter(self):
        """
        Deposit carried litter into the basket.
        """

        if self.carrying_litter is None:
            self.mode = self.IDLE
            return

        self.carrying_litter.mark_collected(
            collection_time=(
                self.model.simulation_time
            )
        )

        self.model.basket.deposit()

        self.carrying_litter = None

        self.mode = self.IDLE


    def step(self):
        """
        One Collection ROV action cycle.
        """

        if self.mode == self.IDLE:

            self.select_target()

            return

        if (
            self.mode
            == self.TO_TARGET
        ):

            if self.target is None:
                self.mode = self.IDLE
                return

            distance = np.linalg.norm(
                self.target.position
                - self.position
            )

            if (
                distance
                <= self.pickup_radius
            ):
                self.attempt_collection()
                return

            self.move_towards(
                self.target.position
            )

            return

        if (
            self.mode
            == self.RETURN_TO_BASKET
        ):

            basket_distance = np.linalg.norm(
                self.model.basket.position
                - self.position
            )

            if (
                basket_distance
                <= self.model.basket_docking_radius
            ):
                self.deposit_litter()
                return

            self.move_towards(
                self.model.basket.position
            )