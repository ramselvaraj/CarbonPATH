'''
GEMM workload class abstraction
'''

class GEMMWorkload:
    def __init__(self, m, k, n, m_offset = 0, n_offset = 0, k_offset = 0):
        self.m = m
        self.k = k
        self.n = n
        # offset is the starting point
        self.m_offset = m_offset  
        self.k_offset = k_offset
        self.n_offset = n_offset

        self.macs = m * k * n
        self.assigned = False
        self.assigned_SA = None

    def __lt__(self, other):
        return self.macs < other.macs

    def __eq__(self, other):
        # Consider equal if they represent the same macs (ranking equality) and same volume
        return self.macs == other.macs
    
    def shape(self):
        return (self.m, self.k, self.n)

    def __hash__(self):
        # Uniquely hash based on spatial coordinates of the output and input ranges
        return hash((
            self.m_offset, self.m_offset + self.m,
            self.k_offset, self.k_offset + self.k,
            self.n_offset, self.n_offset + self.n
        ))

    def __repr__(self):
        return f"GEMMWorkload({self.m}, {self.k}, {self.n})"

    def __str__(self):
        return f"({self.m}, {self.k}, {self.n})"

    @staticmethod
    def iter_macs(workload_list):
        for w in workload_list:
            yield w.macs


    
