class Engine:
    def __init__(self, predictors):
        # The predictors should be a list of functions
        # The order in the list will be the priority of each predictor
        self.predictors = predictors
        self.all_done = False
        # self.INFOLEVEL = "WARNING"
        self.INFOLEVEL = "NONE"
    
    def process(self, catboxes):
        output_catboxes = {}
        for predictor in self.predictors:
            output_catboxes[predictor.name] = predictor.predict(catboxes)
            if self.check(output_catboxes[predictor.name], predictor.name):
                break
        
        # TODO: implement the voting methods to vote for the final prediction
        final_catboxes = list(output_catboxes.values())[0]
        
        self.all_done = self.check(final_catboxes)
        return final_catboxes
    
    def check(self, catboxes, predictor_name=None):
        NotImplemented