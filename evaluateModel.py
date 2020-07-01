import json

import numpy as np

from utill import get_iou
from generate_code import Relationship, Class


class SimilaryMetric:

    def __init__(self, filePath=None):
        self.generated_classes = []
        self.ground_truth_classes = []

        # bind ground_truth_class and generated_class
        self.class_mapping = {}

        # calculated statistic data
        self.class_cnt_percentage = 0
        self.over_all_atribue_recognition_score = 0
        self.type_and_access_correct_atribute_score = 0
        self.relationships_evalution_score = 0

        self.ocr_attr_name_evalutaion_score = 0
        self.ocr_method_name_evalutaion_score = 0
        self.ocr_class_name_evalutation_score = 0

        # load groundTurth data
        if filePath:
            self.load_from_json(filePath)

    def load_from_json(self, file_path):
        with open(file_path) as data_file:
            data = json.load(data_file)
            self.parse_class_data(data["classes"])

    def set_generated_classes(self, data):
        self.generated_classes = data

    def parse_class_data(self, data):
        for d in data:
            c = Class(d["text_array"])
            c.set_coordinates(d["region"])
            self.parse_relationship_data(c, d["relationships"])
            self.ground_truth_classes.append(c)

    def parse_relationship_data(self, grt_class, data):
        for d in data:
            r = Relationship(type_name=d["type_name"], rel_type=d["type"], class_a=grt_class)
            grt_class.add_relationship(r)

    def calculate_similarity(self):
        self.do_class_mapping()
        self.calculate_class_similarity()
        self.calculate_relationships_similarity()
        self.evaluate_OCR()

    def do_class_mapping(self):
        for ground_truth_class in self.ground_truth_classes:
            for generated_class in self.generated_classes:
                iou = get_iou(ground_truth_class.coordinates, generated_class.coordinates)
                if iou > 0.8:
                    self.class_mapping[ground_truth_class] = generated_class

    def calculate_atribute_similarity(self):
        grt_attr_cnt = 0
        gen_attr_cnt = 0
        valid_type_acces_attr = 0
        for grt_class, generated_classs in self.class_mapping.items():
            grt_attr_cnt += len(grt_class.attributes)
            gen_attr_cnt += len(generated_classs.attributes)
            # trazimo atribute kod kojih je prepoznao dobro modifkator pristupa tip atriubta
            for grt_att in grt_class.attributes:
                for gen_attr in generated_classs.attributes:
                    if grt_att.private == gen_attr.private and grt_att.type == gen_attr.type:
                        valid_type_acces_attr += 1
                        break

        self.over_all_atribue_recognition_score = float(gen_attr_cnt) / grt_attr_cnt if grt_attr_cnt != 0 else 0
        self.type_and_access_correct_atribute_score = float(
            valid_type_acces_attr) / grt_attr_cnt if grt_attr_cnt != 0 else 0

        self.over_all_atribue_recognition_score = 1 if \
            self.over_all_atribue_recognition_score > 1 else self.over_all_atribue_recognition_score

    def calculate_class_similarity(self):
        self.class_cnt_percentage = float(len(self.class_mapping.keys())) / len(self.ground_truth_classes)
        self.calculate_atribute_similarity()

    """
        moraju da imaju isti tip isti kardinalitet i istu klasu koja je dodeljena vezi
    """

    def calculate_relationships_similarity(self):
        found_similar = 0.0
        total_rel = 0.0
        for grt_class, generated_classs in self.class_mapping.items():

            total_rel += len(grt_class.relationships)

            for groud_truth_rel in grt_class.relationships:
                for generated_rel in generated_classs.relationships:

                    if groud_truth_rel.type == generated_rel.type \
                            and groud_truth_rel.type_name in generated_rel.type_name:
                        found_similar += 1

        self.relationships_evalution_score = found_similar / total_rel if total_rel > 0 else 0

        if total_rel == 0:
            self.relationships_evalution_score = None


    def calculate_attribute_and_function_names_similarity(self):
        total_attr_cnt = 0
        total_method_cnt = 0
        total_classes_name = len(self.class_mapping.keys())

        valid_attr_name_cnt = 0.0
        valid_method_name_cnt = 0.0
        valid_classes_name = 0.0

        for grt_class, generated_classs in self.class_mapping.items():
            total_attr_cnt += len(grt_class.attributes)
            total_method_cnt += len(grt_class.methods)

            if grt_class.name == generated_classs.name:
                valid_classes_name += 1

            for grt_method in grt_class.methods:
                for gen_method in generated_classs.methods:
                    if grt_method.name == gen_method.name:
                        valid_method_name_cnt += 1
                        break

            for grt_attr in grt_class.attributes:
                for gen_attr in generated_classs.attributes:
                    if grt_attr.name == gen_attr.name:
                        valid_attr_name_cnt += 1
                        break

        self.ocr_class_name_evalutation_score = valid_classes_name / total_classes_name if total_classes_name != 0 else 0
        self.ocr_attr_name_evalutaion_score = valid_attr_name_cnt / total_attr_cnt if total_attr_cnt != 0 else 0
        self.ocr_method_name_evalutaion_score = valid_method_name_cnt / total_method_cnt if total_method_cnt != 0 else 0

        if total_method_cnt == 0:
            self.ocr_method_name_evalutaion_score = None
        if total_attr_cnt == 0:
            self.ocr_attr_name_evalutaion_score = None


    def evaluate_OCR(self):
        self.calculate_attribute_and_function_names_similarity()

    def show_statistic(self):
        self.calculate_similarity()
        print("class cnt percentage: ", self.class_cnt_percentage)
        print("relationship evalution score: ", self.relationships_evalution_score)
        print("total found attributes / total ground truth attributes", self.over_all_atribue_recognition_score)
        print("valid type and access modifier attribute score", self.type_and_access_correct_atribute_score)
        print("OCR classes name evalution score: ", self.ocr_class_name_evalutation_score)
        print("OCR attributes name evalution score: ", self.ocr_attr_name_evalutaion_score)
        print("OCR methods name evalution score: ", self.ocr_method_name_evalutaion_score)




def init_evaluation_data(file_path=None):
    return SimilaryMetric(file_path)


def calculate_average_metrics(sim_metrics):
    avg_class_percentage = []
    avg_relationship_score = []
    avg_over_all_attribute_recognition = []
    avg_type_and_access_correctnes = []
    avg_class_name_percentage = []
    avg_attr_name_percetange = []
    avg_method_name_percentage = []

    for sim_metric in sim_metrics:
        if sim_metric.class_cnt_percentage is not None:
            avg_class_percentage.append(sim_metric.class_cnt_percentage)

        if sim_metric.relationships_evalution_score is not None:
            avg_relationship_score.append(sim_metric.relationships_evalution_score)

        if sim_metric.over_all_atribue_recognition_score is not None:
            avg_over_all_attribute_recognition.append(sim_metric.over_all_atribue_recognition_score)

        if sim_metric.type_and_access_correct_atribute_score is not None:
            avg_type_and_access_correctnes.append(sim_metric.type_and_access_correct_atribute_score)

        if sim_metric.ocr_class_name_evalutation_score is not None:
            avg_class_name_percentage.append(sim_metric.ocr_class_name_evalutation_score)

        if sim_metric.ocr_attr_name_evalutaion_score is not None:
            avg_attr_name_percetange.append(sim_metric.ocr_attr_name_evalutaion_score)

        if sim_metric.ocr_method_name_evalutaion_score is not None:
            avg_method_name_percentage.append(sim_metric.ocr_method_name_evalutaion_score)

    avg_class_percentage = np.average(avg_class_percentage)
    avg_relationship_score = np.average(avg_relationship_score)
    avg_over_all_attribute_recognition = np.average(avg_over_all_attribute_recognition)
    avg_type_and_access_correctnes = np.average(avg_type_and_access_correctnes)
    avg_class_name_percentage = np.average(avg_class_name_percentage)
    avg_attr_name_percetange = np.average(avg_attr_name_percetange)
    avg_method_name_percentage = np.average(avg_method_name_percentage)

    print("avg class cnt percentage: ", avg_class_percentage)
    print("avg relationship evalution score: ", avg_relationship_score)
    print("avg total found attributes / total ground truth attributes", avg_over_all_attribute_recognition)
    print("avg valid type and access modifier attribute score", avg_type_and_access_correctnes)
    print("avg OCR classes name evalution score: ", avg_class_name_percentage)
    print("avg OCR attributes name evalution score: ", avg_attr_name_percetange)
    print("avg OCR methods name evalution score: ", avg_method_name_percentage)


if __name__ == '__main__':
    similarityMetric = init_evaluation_data("dataset/test/groudTruth/ground_truth_d11.txt")
    similarityMetric.show_statistic()
