'''
Class definition of Systolic Array, GEMMWorkloads
'''
from __future__ import annotations
from system.utils.GEMMWorkload import GEMMWorkload



CORE_SIZE_STEP   = 16
NUM_CORES        = 4
CORE_SIZES       = [64,96,128,192]

BUFFER_SIZE_STEP = 8            # kiB
MAX_BUFFER_SIZE  = 64             # kiB
MIN_BUFFER_SIZE  = max(
    BUFFER_SIZE_STEP,
    MAX_BUFFER_SIZE - NUM_CORES * BUFFER_SIZE_STEP,
)
BUFFER_CONFIG = [
        min(MAX_BUFFER_SIZE, MIN_BUFFER_SIZE + BUFFER_SIZE_STEP * i)
        for i in range(NUM_CORES)
]

SIZE_BUFFER_DICT = {}
for i in range(len(CORE_SIZES)):
    SIZE_BUFFER_DICT[CORE_SIZES[i]] = BUFFER_CONFIG[i]

'''
Systolic Array core class. 
This class is only for core specs, no interconnect info is added.
'''
class SystolicArray:
    def __init__(self, width, height, buffer_size, 
                 id, power, area, 
                 sram_energy_scale, dram_energy_scale, 
                 bandwidth = -1, data_flow='ws', 
                 node = 7, frequency = 10**9):
        assert id >= 0, 'Systolic Array Core ID must be non-negative'
        assert data_flow in ['ws', 'os', 'is'], f'Unsupported Dataflow {data_flow}'

        self.width = width
        self.height = height
        self.area = area # mm^2
        self.compute_power = height * width  # MACs/cycle
        self.buffer_size = buffer_size
        self.workloads = []
        self.tile_capacity = 0
        self.location = "chiplet" # base, top, chiplet by default
        self.dram_bandwidth = bandwidth  # words/cycle
        self.sram_energy_scale = sram_energy_scale # pj per bit
        self.dram_energy_scale = dram_energy_scale # pj per bit
        self.data_flow = data_flow
        self.power= power
        self.id = id
        self.saturation = 0
        self.node = int(node) # tech node
        self.total_cycle = 0 # simulated cycle count
        self.cycle_per_layer = []
        self.frequency = frequency # in GHz
        self.simulation_reporter = None

    @classmethod
    def init_from_other_instance(cls, other: 'SystolicArray'):
        assert isinstance(other, SystolicArray), \
            "[ERROR] SystolicArray copy constructor can only accept an instance of SystolicArray"

        # Create a new object with core constructor fields
        new_instance = cls(
            width=other.width,
            height=other.height,
            buffer_size=other.buffer_size,
            id=other.id,
            power=other.power,
            area = other.area,
            dram_energy_scale = other.dram_energy_scale,
            sram_energy_scale = other.sram_energy_scale,
            bandwidth=other.dram_bandwidth,
            data_flow=other.data_flow,
            node=other.node
        )

        return new_instance

    # define systolic array nature order
    def __lt__(self, other):
        return self.compute_power < other.compute_power
    
    def __eq__(self, other):

        return isinstance(other, SystolicArray) and \
        self.id == other.id and \
        self.compute_power == other.compute_power

    def __hash__(self):
        return hash((self.id, self.compute_power))

    def __str__(self) -> str:
        to_return = f"Systolic Array[{self.id}]:{self.width}x{self.height} {self.buffer_size}kiB " \
                f"{len(self.workloads)} workloads with {sum(GEMMWorkload.iter_macs(self.workloads)) / 1e9:.3f} GOPs assigned."
        return to_return
    @staticmethod
    def iter_sa(sa_list):
        for sa in sa_list:
            yield sa

    @staticmethod
    def iter_compute_power(sa_list: list[SystolicArray]):
        for sa in sa_list:
            yield sa.compute_power
    
    def iter_get_workloads(self):
        for wl in self.workloads:
            yield wl

    @staticmethod
    def iter_get_width(sa_list: list[SystolicArray]):
        for core in sa_list:
            yield core.width 

    @staticmethod
    def iter_get_height(sa_list: list[SystolicArray]):
        for core in sa_list:
            yield core.height

    def print_assigned_workloads(self):
        to_return = f"--- SA[{self.id}] {self.width} x {self.height} ---\n"
        for i in range(len(self.workloads)):
            wl = self.workloads[i]
            dim = f"wl[{i}]:{wl.m} x {wl.n} x {wl.k}\n"
            to_return += dim
        print(to_return)
        return to_return
    

    def increase_tile_capacity(self, capacity):
        self.tile_capacity += capacity

    def set_tile_capacity(self, tile_capacity):
        self.tile_capacity = tile_capacity
    
    def assign_workload(self, workload: GEMMWorkload):
        self.workloads.append(workload)
        self.cycle_per_layer.append(0)

    def is_full(self):
        return len(self.workloads) >= self.tile_capacity


    