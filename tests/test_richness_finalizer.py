import unittest
from unittest.mock import patch
import numpy as np
from richness_relation.finalize_saved_richness_mpi import predictive_one, summarize_predictive


class FinalizerTests(unittest.TestCase):
    def test_summary(self):
        result=summarize_predictive([({'x':1.},{'x':2.}),({'x':3.},{'x':2.})])
        self.assertEqual(result['x']['observed_median'],2.)
        self.assertEqual(result['x']['ppc_probability_rep_ge_obs'],.5)

    def test_rank_independent_draws(self):
        data=dict(lambda_rm=np.arange(5.),lambda_spec=np.arange(5.),z=np.ones(5))
        with patch('richness_relation.finalize_saved_richness_mpi.selected_rvs',
                   side_effect=lambda t,s,z,c,rng:rng.normal(size=len(s))), \
             patch('richness_relation.finalize_saved_richness_mpi.discrepancies',
                   side_effect=lambda rm,*args:dict(mean=float(np.mean(rm)))):
            serial=[predictive_one(None,data,None,42,i) for i in range(6)]
            parallel=sorted((i,predictive_one(None,data,None,42,i)) for rank in range(3) for i in range(rank,6,3))
        self.assertEqual(serial,[v for i,v in parallel])
