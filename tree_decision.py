from data_loader import load_normalized_data


class Node:
    def __init__(self):
        self.feature_index     = None
        self.threshold         = None
        self.left_subtree      = None
        self.right_subtree     = None
        self.prediction_value  = None   # None as long as this is not a leaf
 
    def is_leaf(self):
        return self.prediction_value is not None
    

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.root              = None