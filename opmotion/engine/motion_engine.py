from .engine import Engine

class MotionEngine(Engine):
    def check(self, catboxes, predictor_name=None):
        num_not_predicted = 0
        for catbox in catboxes.values():
            if catbox.motionType is None:
                num_not_predicted += 1

        if num_not_predicted > 0:
            if predictor_name is None:
                print(f"Motion Engine Warning: Current rules cannot cover all catboxes, still {num_not_predicted} catboxes not predicted")
            else:
                print(f"Motion Engine Warning: After {predictor_name}, still {num_not_predicted} catboxes not predicted")
            return False
        if predictor_name is not None and self.INFOLEVEL == "WARNING":
            print(f"{predictor_name} Motion Engine: All catboxes are predicted")
        return True