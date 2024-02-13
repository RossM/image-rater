import unittest
import torch
import cProfile

from scripts.modules.mtree import *

class MTreeUnitTest(unittest.TestCase):
    
    def test_all(self):

        with cProfile.Profile() as pr:
            data = torch.randn((5000, 128))
            data.div_(data.norm(dim=1,keepdim=True))
        
            mtree = MTree()
        
            for i in range(data.shape[0]):
                dist, val = mtree.get_nearest(data[i])
                mtree.add_point(data[i])
               
        pr.print_stats('time')
        
        test_data = torch.randn((20, 128))
        test_data.div_(test_data.norm(dim=1,keepdim=True))
        
        for i in range(test_data.shape[0]):
            truth_dist, truth_idx = (data - test_data[i]).norm(dim=1).min(dim=0)
            truth_val = data[truth_idx]

            dist, val = mtree.get_nearest(test_data[i])
            self.assertAlmostEqual(dist, truth_dist.item())
            self.assertTrue(torch.allclose(val, truth_val))
            
if __name__ == '__main__':
    unittest.main()