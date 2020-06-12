import os
import sys


class Class:
    def __init__(self, name):
        self.coordinates = []
        self.name = name
        self.atributes = []
        self.methods = []
        self.relationships = []  # ovo je da se na osnovu veza naprave npr liste atributa (ako je ovo klasa auto)
        # da se kreira atribut tockovi koji ce biti lista ciji su elementi tockovi


class Relationship:
    def __init__(self, rel_type):
        self.coordinates = []
        self.type = rel_type
        self.classA = None
        self.classB = None


def init_project(projectName, projectPath):
    os.mkdir(projectPath + '/' + projectName)


# TODO: na osnovu pozicija klasa i veza da se generisu objeti tipa Relationship..
def generateRelationshipObjects(relationships, class_regions):
    retVal = []
    for relationship in relationships:
        a = Relationship(relationship["relationship_type"])

        #TODO: na osnovu koordinata veze i iz klass objekta pokupiti nazive klasa i staviti za Atribute classA i classB
        retVal.append(a)
    return retVal

# ovde da se na osnovu rezultata OCR-a upisu naziv klase i osnovni atributi
def generateClassObjectCode(class_data, class_name, all_relationships):
    newClass = Class(class_name)
    newClass.atributes = class_data['atributes']
    newClass.methods = class_data['methods']

    # treba proci kroz sve veze i videti od kojih treba praviti atribute tipa liste, set-a..

    return newClass


# TODO: ovde na osnovu instance tipa Class treba da se pise u fajl ---> da se taj objekat prevede u java kod..
def writeClassObjectToFile(classObject: Class, classPath):
    f = open(classPath, "x")
    pass


def make_project(projectPath, projectName, classes_data, relationshipData):

    init_project(projectName, projectPath)
    relationShips = generateRelationshipObjects(relationshipData, classes_data)

    for indx, class_data in enumerate(classes_data):
        class_name = class_data['class_name'] if 'class_name' in class_data else str(indx)
        classPath = projectPath + '/' + projectName + '/' + class_name + '.java'

        gen_class: Class = generateClassObjectCode(class_data, class_name, relationShips)
        writeClassObjectToFile(gen_class, classPath)


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
