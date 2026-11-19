"""Local checks of test aggregation; real MPI transport is tested on NERSC."""
import unittest
import numpy as np
from local_overdensity.run_environment_mpi_test import compare_statistics


class MpiSmokeTests(unittest.TestCase):
    def test_round_robin_covers_each_task_once(self):
        for count in (1, 3, 147):
            assigned = [i for rank in range(4) for i in range(rank, count, 4)]
            self.assertEqual(sorted(assigned), list(range(count)))

    def test_comparison_accepts_roundoff_rejects_missing_contribution(self):
        reference = np.ones((6, 3, 5))
        self.assertLess(compare_statistics(reference + 1e-12, reference), 1e-10)
        with self.assertRaises(AssertionError):
            compare_statistics(reference * 2, reference)


if __name__ == '__main__':
    unittest.main()
