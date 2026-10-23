import hashlib
import unittest
from collections import OrderedDict


class LearnedDepthCacheTests(unittest.TestCase):
    def test_cached_frame_does_not_require_model_inference(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("vision extras are not installed")
        from visual_nav.learned_depth import LearnedMetricDepthEstimator

        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        expected = np.ones((2, 3), dtype=np.float32)
        key = hashlib.blake2b(memoryview(np.ascontiguousarray(frame)), digest_size=16).digest()
        estimator = LearnedMetricDepthEstimator.__new__(LearnedMetricDepthEstimator)
        estimator._cache = OrderedDict([(key, expected)])
        estimator.cache_hits = 0
        estimator.cache_misses = 0
        result = estimator.estimate(frame)
        self.assertIs(result, expected)
        self.assertEqual(estimator.cache_hits, 1)
        self.assertEqual(estimator.cache_misses, 0)


if __name__ == "__main__":
    unittest.main()
