import json

from generate_code import Relationship


class SimilaryMetric:

    def __init__(self, filePath=None):
        self.generated_classes = []
        self.generated_relationships = []
        self.ground_truth_classes = []
        self.ground_truth_relationships = []

        # calculated statistic data
        self.class_cnt_percentage = 0
        self.relationships_evalution_score = 0
        self.ocr_evalutaion_score = 0

        # load groundTurth data
        if filePath:
            self.load_from_json(filePath)

    def load_from_json(self, file_path):
        with open(file_path) as data_file:
            data = json.load(data_file)
            self.ground_truth_classes = data["classes"]
            self.parse_relationship_data(data["relationships"])

    def set_generated_classes(self, data):
        self.generated_classes = data

    def parse_relationship_data(self, data):
        for d in data:
            r = Relationship(type_name=d["type_name"])
            self.ground_truth_relationships.append(r)

    def set_generated_relationships(self, data):
        self.generated_relationships = data



    def calculate_similarity(self):
        self.calculate_class_similarity()
        self.calculate_relationships_similarity()
        self.evaluate_OCR()

    # @TODO: implementirati da daje vise detalja, da gleda vise stvari
    def calculate_class_similarity(self):
        self.class_cnt_percentage = float(len(self.generated_classes)) / len(self.ground_truth_classes)


    def calculate_relationships_similarity(self):
        found_similar = 0.0
        for groud_truth_rel in self.ground_truth_relationships:
            for generated_rel in self.generated_relationships:

                # dodati jos logige ono da li istu klasu povezuju
                if groud_truth_rel.type_name == generated_rel.type_name:
                    found_similar += 1

        self.relationships_evalution_score = found_similar / len(self.ground_truth_relationships)

    # @TODO: implementirati
    def evaluate_OCR(self):
        pass

    def show_statistic(self):
        self.calculate_similarity()
        print("class cnt percentage: ", self.class_cnt_percentage)
        print("relation ship evalution score: ", self.relationships_evalution_score)
        print("OCR evalution score: ", self.ocr_evalutaion_score)


def init_evaluation_data(file_path=None):
    return SimilaryMetric(file_path)


if __name__ == '__main__':
    similarityMetric = init_evaluation_data("dataset/test/groudTruth/ground_truth_d11.txt")
    similarityMetric.show_statistic()
