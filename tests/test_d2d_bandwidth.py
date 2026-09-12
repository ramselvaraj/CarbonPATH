import unittest

from chiplet.n_utils import d2d_bw_calc


class D2DBandwidthTests(unittest.TestCase):
    def test_protocol_rate_is_not_replaced_by_last_configured_protocol(self):
        area = 0.292383

        self.assertAlmostEqual(
            d2d_bw_calc("2.5d_active", "ucie_adv", area, False, node=7),
            138.4254755744043,
        )
        self.assertAlmostEqual(
            d2d_bw_calc("2.5d_active", "ucie_std", area, False, node=7),
            69.21273778720215,
        )
        self.assertAlmostEqual(
            d2d_bw_calc("2.5d_active", "ucie_3d", area, False, node=7),
            8.651592223400268,
        )


if __name__ == "__main__":
    unittest.main()
