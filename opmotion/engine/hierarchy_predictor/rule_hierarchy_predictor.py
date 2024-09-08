import numpy as np
import copy

class RuleHierarchyPredictor:
    def __init__(self):
        self.name = "rule_hierarchy_predictor"

    def predict(self, catboxes):
        catboxes = copy.deepcopy(catboxes)
        # Pick the base part with the largest volume as the root if there are more than 1 base part
        base = None
        max_volume = None
        for part_id, catbox in catboxes.items():
            if catbox.cat == "base":
                volume = np.product(catbox.box.dim)
                if base is None:
                    base = part_id
                    max_volume = volume
                else:
                    if volume > max_volume:
                        base = part_id
                        max_volume = volume
        catboxes[base].parent = -1

        # Set the rest of the parts as the children of the base
        for catbox in catboxes.values():
            if catbox.parent is None:
                catbox.parent = base

        return catboxes
