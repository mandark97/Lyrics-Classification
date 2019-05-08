from datetime import datetime
import csv


class ResultLogger(object):
    PIPELINE = "pipeline"

    def __init__(self, log_type, obj):
        self.log_type = log_type
        self.obj = obj

    def score(self, X_test, y_test, save_to_csv=True):
        if self.log_type == self.PIPELINE:
            score = self.obj.score(X_test, y_test)
            print(score)

            data = {
                "steps": self.obj.steps,
                "score": score,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            if save_to_csv:
                self.__to_csv(data)
            else:
                return data

    def __to_csv(self, data):
        with open("pipeline_results.csv", "a") as f:
            dict_writer = csv.DictWriter(f, fieldnames=['timestamp', 'score', 'steps'])
            dict_writer.writerow(data)
