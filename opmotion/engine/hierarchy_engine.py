from .engine import Engine

class HierarchyEngine(Engine):
    def check(self, catboxes, predictor_name=None):
        num_not_predicted = 0
        for catbox in catboxes.values():
            if catbox.parent is None:
                num_not_predicted += 1
        
        if num_not_predicted > 0:
            if predictor_name is None:
                print(f"Hierarchy Engine Warning: Current rules cannot cover all catboxes, still {num_not_predicted} catboxes not predicted")
            else:
                print(f"Hierarchy Engine Warning: After {predictor_name}, still {num_not_predicted} catboxes not predicted")
            return False
        if predictor_name is not None and self.INFOLEVEL == "WARNING":
            print(f"{predictor_name} Hierarchy Engine: All catboxes are predicted")
        return True

