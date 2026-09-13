import unittest

from script.run_baseline_campaign import (
    DEFAULT_WORKLOADS,
    SCHEDULE,
    planned_move_count,
    task_list,
)


class BaselineCampaignTests(unittest.TestCase):
    def test_requested_schedule_has_expected_move_budget(self):
        self.assertEqual(planned_move_count(SCHEDULE), 75650)

    def test_tasks_are_round_robin_by_run(self):
        self.assertEqual(
            task_list(DEFAULT_WORKLOADS, 2),
            [(7, 1), (9, 1), (10, 1), (7, 2), (9, 2), (10, 2)],
        )


if __name__ == "__main__":
    unittest.main()
