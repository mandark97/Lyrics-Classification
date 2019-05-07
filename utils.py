from datetime import datetime
import csv


class ResultLogger(object):
    PIPELINE = "pipeline"

    def __init__(self, log_type, obj):
        self.log_type = log_type
        self.obj = obj

    def score(self, X_test, y_test):
        if self.log_type == self.PIPELINE:
            score = self.obj.score(X_test, y_test)
            print(score)

            return {
                "steps": self.obj.steps,
                "score": score,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
