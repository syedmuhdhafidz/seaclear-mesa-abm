"""
SeaClear Mesa Multi-Agent Continuous-Space Model
------------------------------------------------

Current agents:
    - ObservationROVAgent (miniTortuga)
    - CollectionROVAgent (Tortuga)
    - LitterAgent
    - BasketAgent
    - MarineLifeAgent
"""

import math

import numpy as np

from mesa import Model
from mesa.datacollection import DataCollector

from mesa.experimental.continuous_space import (
    ContinuousSpace,
)

from seaclear.agents import (
    BasketAgent,
    CollectionROVAgent,
    LitterAgent,
    LitterState,
    MarineLifeAgent,
    ObservationROVAgent,
)


# ============================================================
# 1. PATH HELPERS
# ============================================================


def interpolate_line(
    start,
    end,
    step_size,
):
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

    def __init__(
        self,
        width=50.0,
        height=50.0,
        num_litter=30,
        num_marine_life=10,
        rov_speed=1.0,
        collection_rov_speed=1.2,
        track_spacing=5.0,
        step_size=0.5,
        detection_radius=3.0,
        max_detection_probability=0.95,
        detection_decay=0.18,
        large_litter_probability=0.10,
        pickup_radius=0.75,
        collection_success_probability=0.90,
        basket_docking_radius=1.0,
        rng=42,
    ):

        super().__init__(
            rng=rng
        )

        # ====================================================
        # Parameters
        # ====================================================

        self.width = float(width)
        self.height = float(height)

        self.num_litter = int(
            num_litter
        )

        self.num_marine_life = int(
            num_marine_life
        )

        self.rov_speed = float(
            rov_speed
        )

        self.collection_rov_speed = float(
            collection_rov_speed
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

        self.large_litter_probability = float(
            large_litter_probability
        )

        self.pickup_radius = float(
            pickup_radius
        )

        self.collection_success_probability = float(
            collection_success_probability
        )

        self.basket_docking_radius = float(
            basket_docking_radius
        )

        # One model step corresponds to the time required
        # for the Observation ROV to travel STEP_SIZE.
        self.time_step = (
            self.step_size
            / self.rov_speed
        )

        # ====================================================
        # Simulation state
        # ====================================================

        self.simulation_time = 0.0

        self.step_count = 0

        self.running = True

        # ====================================================
        # Continuous space
        # ====================================================

        expected_agents = (
            self.num_litter
            + self.num_marine_life
            + 3
        )

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
            n_agents=expected_agents,
        )

        # ====================================================
        # Search path
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
        # Basket
        # ====================================================

        basket_position = np.array(
            [
                self.width / 2.0,
                self.height / 2.0,
            ],
            dtype=float,
        )

        self.basket = BasketAgent(
            model=self,
            space=self.space,
            position=basket_position,
        )

        # ====================================================
        # Litter agents
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

            is_large = (
                self.random.random()
                < self.large_litter_probability
            )

            litter = LitterAgent(
                model=self,
                space=self.space,
                position=[
                    x,
                    y,
                ],
                is_large=is_large,
            )

            self.litter_agents.append(
                litter
            )

        # ====================================================
        # Marine-life agents
        # ====================================================

        self.marine_life_agents = []

        for _ in range(
            self.num_marine_life
        ):

            x = self.random.uniform(
                0.0,
                self.width,
            )

            y = self.random.uniform(
                0.0,
                self.height,
            )

            marine_life = (
                MarineLifeAgent(
                    model=self,
                    space=self.space,
                    position=[
                        x,
                        y,
                    ],
                    speed=0.10,
                )
            )

            self.marine_life_agents.append(
                marine_life
            )

        # ====================================================
        # Observation ROV
        # ====================================================

        self.rov = ObservationROVAgent(
            model=self,
            space=self.space,
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

        # ====================================================
        # Collection ROV
        # ====================================================

        self.collection_rov = (
            CollectionROVAgent(
                model=self,
                space=self.space,
                position=(
                    basket_position.copy()
                ),
                speed=(
                    self.collection_rov_speed
                ),
                pickup_radius=(
                    self.pickup_radius
                ),
                collection_success_probability=(
                    self.collection_success_probability
                ),
            )
        )

        # ====================================================
        # DataCollector
        # ====================================================

        self.datacollector = DataCollector(

            model_reporters={

                "Mapped Litter": (
                    lambda model:
                    model.get_mapped_count()
                ),

                "Collected Litter": (
                    lambda model:
                    model.get_collected_count()
                ),

                "Failed Litter": (
                    lambda model:
                    model.get_failed_count()
                ),

                "Large Litter": (
                    lambda model:
                    model.get_large_flagged_count()
                ),

                "Pending Litter": (
                    lambda model:
                    model.get_pending_count()
                ),

                "Undiscovered Litter": (
                    lambda model:
                    model.get_undiscovered_count()
                ),

                "Basket Count": (
                    lambda model:
                    model.basket.collected_count
                ),

                "Observation Distance": (
                    lambda model:
                    model.rov.distance_travelled
                ),

                "Collection Distance": (
                    lambda model:
                    model.collection_rov.distance_travelled
                ),

                "Simulation Time": (
                    lambda model:
                    model.simulation_time
                ),
            }
        )

        self.datacollector.collect(
            self
        )


    # ========================================================
    # MODEL STEP
    # ========================================================

    def step(self):

        if not self.running:
            return

        self.step_count += 1

        # Mapping
        self.rov.step()

        # Collection occurs in parallel.
        self.collection_rov.step()

        # Environmental agents
        for marine_life in (
            self.marine_life_agents
        ):
            marine_life.step()

        # Advance global model time only once.
        self.simulation_time += (
            self.time_step
        )

        self.datacollector.collect(
            self
        )

        if self.is_mission_complete():
            self.running = False


    # ========================================================
    # MISSION TERMINATION
    # ========================================================

    def is_mission_complete(self):

        if not self.rov.mission_complete:
            return False

        normal_pending = [
            litter
            for litter
            in self.litter_agents
            if (
                litter.state
                == LitterState.ON_MAP
                and not litter.is_large
            )
        ]

        collector_busy = (
            self.collection_rov.mode
            != CollectionROVAgent.IDLE
            or
            self.collection_rov.carrying_litter
            is not None
        )

        return (
            len(normal_pending) == 0
            and not collector_busy
        )


    # ========================================================
    # STATE QUERIES
    # ========================================================

    def get_undiscovered_litter(self):

        return [
            litter
            for litter
            in self.litter_agents
            if (
                litter.state
                == LitterState.UNDISCOVERED
            )
        ]


    def get_pending_litter(self):

        return [
            litter
            for litter
            in self.litter_agents
            if (
                litter.state
                == LitterState.ON_MAP
                and not litter.is_large
            )
        ]


    def get_large_flagged_litter(self):

        return [
            litter
            for litter
            in self.litter_agents
            if (
                litter.state
                == LitterState.ON_MAP
                and litter.is_large
            )
        ]


    def get_collected_litter(self):

        return [
            litter
            for litter
            in self.litter_agents
            if (
                litter.state
                == LitterState.COLLECTED
            )
        ]


    def get_failed_litter(self):

        return [
            litter
            for litter
            in self.litter_agents
            if (
                litter.state
                == LitterState.FAILED
            )
        ]


    def get_picked_up_litter(self):

        return [
            litter
            for litter
            in self.litter_agents
            if (
                litter.state
                == LitterState.PICKED_UP
            )
        ]


    # ========================================================
    # COUNTERS
    # ========================================================

    def get_undiscovered_count(self):

        return len(
            self.get_undiscovered_litter()
        )


    def get_pending_count(self):

        return len(
            self.get_pending_litter()
        )


    def get_large_flagged_count(self):

        return len(
            self.get_large_flagged_litter()
        )


    def get_collected_count(self):

        return len(
            self.get_collected_litter()
        )


    def get_failed_count(self):

        return len(
            self.get_failed_litter()
        )


    def get_mapped_count(self):
        """
        Count every litter item that is no longer undiscovered.
        """

        return (
            self.num_litter
            - self.get_undiscovered_count()
        )


    def get_mapping_rate(self):

        if self.num_litter == 0:
            return 0.0

        return (
            self.get_mapped_count()
            / self.num_litter
            * 100.0
        )


    def get_collection_rate(self):
        """
        Percentage of all litter deposited in basket.
        """

        if self.num_litter == 0:
            return 0.0

        return (
            self.get_collected_count()
            / self.num_litter
            * 100.0
        )


    def get_progress(self):

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