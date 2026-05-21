from data_loader import load_normalized_data


class Node:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_subtree = None
        self.right_subtree = None
        self.prediction_value = None   
 
    def is_leaf(self):
        return self.prediction_value is not None
    

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, Y):
        X = [list(row) for row in X]
        Y = list(Y)
 
        self.root = self._build_tree(X, Y, current_depth=0)
    

    def predict(self, sample):
        if self.root is None:
            raise Exception("The tree is not trained yet. Call fit() first.")
 
        return self._traverse_tree(sample, self.root)
    

    def evaluate(self, X, Y):
        X = [list(row) for row in X]
        Y = list(Y)
 
        unique_classes = list(set(Y))
 
        all_predictions = [self.predict(sample) for sample in X]
 
        f1_scores_per_class = []
 
        for current_class in unique_classes:
            true_positives  = 0   
            false_positives = 0   
            false_negatives = 0   
 
            for predicted_class, true_class in zip(all_predictions, Y):
                if predicted_class == current_class and true_class == current_class:
                    true_positives += 1
                elif predicted_class == current_class and true_class != current_class:
                    false_positives += 1
                elif predicted_class != current_class and true_class == current_class:
                    false_negatives += 1
 
            if true_positives + false_positives == 0:
                precision = 0.0
            else:
                precision = true_positives / (true_positives + false_positives)
 
            if true_positives + false_negatives == 0:
                recall = 0.0
            else:
                recall = true_positives / (true_positives + false_negatives)
 
            if precision + recall == 0:
                f1_current_class = 0.0
            else:
                f1_current_class = 2 * (precision * recall) / (precision + recall)
 
            f1_scores_per_class.append(f1_current_class)
 
        f1_macro = sum(f1_scores_per_class) / len(f1_scores_per_class)
        return f1_macro