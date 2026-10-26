import unittest

from system.utils.Rule4MlInputBuffer import (
    BRAM_BLOCK_BITS,
    input_buffer,
    minimum_blocks,
    precision_bits,
)
from system.utils.UnsupportedEvaluation import (
    ResourceExceeded,
    UnsupportedEvaluation,
)


class InputBufferTests(unittest.TestCase):
    def test_size_is_elements_times_bits(self):
        self.assertEqual(input_buffer(1024, 8).bits, 8192)

    def test_default_uses_the_fewest_blocks(self):
        buffer = input_buffer(1024, 8)
        self.assertEqual(buffer.blocks, 1)
        self.assertEqual(buffer.read_cycles, 228)  # ceil(8192 / 36)

    def test_more_blocks_read_faster(self):
        cycles = [input_buffer(1024, 8, blocks).read_cycles for blocks in (1, 4, 16, 228)]
        self.assertEqual(cycles, [228, 57, 15, 1])

    def test_capacity_sets_the_minimum_blocks(self):
        self.assertEqual(minimum_blocks(BRAM_BLOCK_BITS), 1)
        self.assertEqual(minimum_blocks(BRAM_BLOCK_BITS + 1), 2)

    def test_blocks_below_capacity_are_rejected_as_a_resource_failure(self):
        with self.assertRaises(ResourceExceeded):
            input_buffer(4096, 8, blocks=1)  # 32768 bits need two blocks

    def test_invalid_blocks_are_unsupported(self):
        for blocks in (0, -1, 1.5, True):
            with self.assertRaises(UnsupportedEvaluation):
                input_buffer(16, 8, blocks)

    def test_precision_width(self):
        self.assertEqual(precision_bits("ap_fixed<8, 3>"), 8)

    def test_unreadable_precision_is_unsupported(self):
        for precision in ("fp8", "int8", None):
            with self.assertRaises(UnsupportedEvaluation):
                precision_bits(precision)


if __name__ == "__main__":
    unittest.main()
