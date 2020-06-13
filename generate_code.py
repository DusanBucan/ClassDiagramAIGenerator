import os
import sys


class Class:
    def __init__(self, name, img=None):
        self.img = img
        self.coordinates = []
        self.name = name
        self.attributes = []
        self.methods = []
        self.relationships = []  # ovo je da se na osnovu veza naprave npr liste atributa (ako je ovo klasa auto)
        # da se kreira atribut tockovi koji ce biti lista ciji su elementi tockovi

    def add_relationship(self, relationship):
        self.relationships.append(relationship)


class Relationship:
    def __init__(self, rel_type, class_a):
        self.type = rel_type
        self.class_a = class_a


def init_project(name, path):
    if not os.path.isdir(path + '/' + name):
        os.mkdir(path + '/' + name)


def add_relationship(relationship, class_a, class_b):
    if relationship == "asocijacija":
        class_a.add_relationship(Relationship("jedan", class_b))
        class_b.add_relationship(Relationship("jedan", class_a))
    elif relationship == "agregacija_desno":
        class_b.add_relationship(Relationship("vise", class_a))
    elif relationship == "agregacija_levo":
        class_a.add_relationship(Relationship("vise", class_b))
    elif relationship == "generalizacija_desno":
        class_a.add_relationship(Relationship("abstaraktna", class_b))
    elif relationship == "generalizacija_levo":
        class_b.add_relationship(Relationship("abstaraktna", class_a))
    elif relationship == "kompozicija_desno":
        class_a.add_relationship(Relationship("jedan", class_b))
        class_b.add_relationship(Relationship("vise", class_a))
    elif relationship == "kompozicija_levo":
        class_b.add_relationship(Relationship("jedan", class_a))
        class_a.add_relationship(Relationship("vise", class_b))
    elif relationship == "realizacija_desno":
        class_a.add_relationship(Relationship("interfejs", class_b))
    elif relationship == "realizacija_levo":
        class_b.add_relationship(Relationship("interfejs", class_a))
    elif relationship == "zavisnost_desno":
        class_a.add_relationship(Relationship("jedan", class_b))
    elif relationship == "zavisnost_levo":
        class_b.add_relationship(Relationship("jedan", class_a))

# ovde da se na osnovu rezultata OCR-a upisu naziv klase i osnovni atributi
def generateClassObjectCode(class_data, class_name, all_relationships):
    newClass = Class(class_name)
    newClass.attributes = class_data['attributes']
    newClass.methods = class_data['methods']

    # treba proci kroz sve veze i videti od kojih treba praviti atribute tipa liste, set-a..

    return newClass


# TODO: ovde na osnovu instance tipa Class treba da se pise u fajl ---> da se taj objekat prevede u java kod..
def write_class_object_to_file(class_data: Class, class_path):
    f = open(class_path, "w")

    f.write("package model;\n\n")
    for rs in class_data.relationships:
        f.write("import model." + rs.class_a.name + ";\n")
    f.write("\n")
    f.write("public class " + class_data.name)

    num_int = 0
    for rs in class_data.relationships:
        if rs.type == "abstaraktna":
            f.write(" extends " + rs.class_a.name)
        elif rs.type == "interfejs" and num_int == 0:
            f.write("implements " + rs.class_a.name)
        elif rs.type == "interfejs" and num_int != 0:
            f.write(", " + rs.class_a.name)

    f.write(" {\n\n")

    # todo: dodati atribute

    constructor_params = []
    for rs in class_data.relationships:
        if rs.type == "jedan":
            f.write("\tprivate " + rs.class_a.name + " " + rs.class_a.name.lower() + ";\n")
            constructor_params.append(rs.class_a.name + " " + rs.class_a.name.lower())
        elif rs.type == "vise":
            f.write("\tprivate Collection<" + rs.class_a.name + "> " + rs.class_a.name.lower() + "Collection;\n")
            constructor_params.append("Collection<" + rs.class_a.name + "> " + rs.class_a.name.lower() + "Collection")

    f.write("\tpublic " + class_data.name + " () { }\n\n")

    if len(constructor_params) > 0 or len(class_data.attributes) > 0:
        # todo: dodati za atribute
        f.write("\tpublic " + class_data.name + " (")
        for idx, param in enumerate(constructor_params):
            f.write(param)
            if idx != len(constructor_params)-1:
                f.write(", ")
        f.write(") {\n")
        for idx, param in enumerate(constructor_params):
            param_name = param.split(" ")[1]
            f.write("\t\tthis." + param_name + " = " + param_name + ";\n")
        f.write("\t}\n\n")

    # todo: dodati metode

    f.write("\n}\n")
    f.close()
    pass


def make_project(path, name, classes_data):

    init_project(name, path)

    for class_data in classes_data:
        class_name = class_data.name
        class_path = path + '/' + name + '/' + class_name + '.java'

        write_class_object_to_file(class_data, class_path)


if __name__ == '__main__':
    prName = sys.argv[1]
    prPath = sys.argv[2]

    # ovako nesto treba da vrati ML deo da bi GeneretaCode radilo, implement fije do kraja..
    ml_returned_classData = [
        {"class_name": "KlasaA",
         "coordinates": [],
         "atributes": [
            {"scope": "public", "name": "a", "type": "int"},
            {"scope": "public", "name": "b", "type": "int"},
         ],
         "methods": [
             {"scope": "public", "name": "sum", "retVal_type": "int", "args": []},
         ]
        },
        {"class_name": "KlasaB",
         "coordinates": [],
         "atributes": [
             {"scope": "public", "name": "a", "type": "int"},
             {"scope": "public", "name": "b", "type": "int"},
         ],
         "methods": [
             {"scope": "public", "name": "sum", "retVal_type": "int", "args": []},
         ]
         },
        {"class_name": "KlasaC",
         "coordinates": [],
         "atributes": [
             {"scope": "public", "name": "a", "type": "int"},
             {"scope": "public", "name": "b", "type": "int"},
         ],
         "methods": [
             {"scope": "public", "name": "sum", "retVal_type": "int", "args": []},
         ]
         }
    ]

    ml_returned_relationshipData = [
        {"coordinates": [], "relationship_type": "realizacija_levo"},
        {"coordinates": [], "relationship_type": "kompozicija_desno"}
    ]

    make_project(prPath, prName, ml_returned_classData, ml_returned_relationshipData)
