import os
import string
from collections import Counter


alpha = list(string.ascii_letters)
chars = ['+', "-", "(", ")", ":"]
types = ["string", "int", "boolean", "long", "double", "float", "void"]


class Class:

    def __init__(self, text_array, img=None):
        self.img = img
        self.name = self.set_name(text_array[0])
        self.attributes, self.methods = self.add_atributtes_and_methods(text_array)
        self.relationships = []

    def set_name(self, name):
        return ''.join([name[i] for i in range(0, len(name)) if name[i] in alpha])

    def add_relationship(self, relationship):
        self.relationships.append(relationship)

    def add_atributtes_and_methods(self, text_array):
        attributes = []
        methods = []
        for i in range(1, len(text_array)):
            text = text_array[i]
            text = ''.join(
                [text[j] for j in range(0, len(text)) if text[j] in alpha or text[j] in chars or text[j] == " "])
            if text == "":
                continue
            aom = AoM()
            if "+" in text:
                aom.private = False

            if "(" in text or ")" in text:
                if ")" in text:
                    index = text.index(")")
                else:
                    index = text.index("(")

                name = text[:index]
                aom.name = ''.join([name[j] for j in range(0, len(name)) if name[j] in alpha])
                type = text[index + 1:]
                type = ''.join([type[j] for j in range(0, len(type)) if type[j] in alpha])
                if type != "":
                    word_count = dict()
                    for word in types:
                        word_count[word] = shared_chars(word, type)

                    word_count = {k: v for k, v in sorted(word_count.items(), key=lambda item: item[1])}
                    # todo: ovde se mogu dodati jos neke provere
                    if list(word_count.items())[-1][0] == "int" and len(type) > 5:
                        aom.type = list(word_count.items())[-2][0]
                    else:
                        aom.type = list(word_count.items())[-1][0]
                methods.append(aom)
            else:
                text = ''.join([text[j] for j in range(0, len(text)) if text[j] in alpha])
                for word in types:
                    if word in text:
                        aom.type = word
                        index = text.index(word)
                        text = text[:index]
                        break
                aom.name = text
                attributes.append(aom)

        return attributes, methods


def shared_chars(s1, s2):
    return sum((Counter(s1) & Counter(s2)).values())


class Relationship:
    def __init__(self, rel_type, class_a):
        self.type = rel_type
        self.class_a = class_a
        self.private = True


class AoM:
    def __init__(self):
        self.type = "int"
        self.name = "name"
        self.private = True


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
        class_a.add_relationship(Relationship("kreira", class_b))
    elif relationship == "zavisnost_levo":
        class_b.add_relationship(Relationship("kreira", class_a))


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

    constructor_params = []
    for at in class_data.attributes:
        f.write("\t")
        if at.private:
            f.write("private ")
        else:
            f.write("public ")
        f.write(at.type + " " + at.name + ";\n")
        constructor_params.append(at.type + " " + at.name)

    for rs in class_data.relationships:
        if rs.type == "jedan":
            f.write("\tprivate " + rs.class_a.name + " " + rs.class_a.name.lower() + ";\n")
            constructor_params.append(rs.class_a.name + " " + rs.class_a.name.lower())
        elif rs.type == "vise":
            f.write("\tprivate Collection<" + rs.class_a.name + "> " + rs.class_a.name.lower() + "Collection;\n")
            constructor_params.append("Collection<" + rs.class_a.name + "> " + rs.class_a.name.lower() + "Collection")

    f.write("\n\tpublic " + class_data.name + " () { }\n\n")

    if len(constructor_params) > 0 or len(class_data.attributes) > 0:
        f.write("\tpublic " + class_data.name + " (")
        for idx, param in enumerate(constructor_params):
            f.write(param)
            if idx != len(constructor_params) - 1:
                f.write(", ")
        f.write(") {\n")
        for idx, param in enumerate(constructor_params):
            param_name = param.split(" ")[1]
            f.write("\t\tthis." + param_name + " = " + param_name + ";\n")
        f.write("\t}\n\n")

    for rs in class_data.relationships:
        if rs.type == "kreira":
            f.write("\n\tpublic " + rs.class_a.name + " create" + rs.class_a.name + " ( ) {\n\t\treturn null;\n\t}\n")


    for m in class_data.methods:
        f.write("\n\t")
        if m.private:
            f.write("private ")
        else:
            f.write("public ")
        f.write(m.type + " " + m.name + " ( ) {\n\t\treturn null;\n\t}\n")

    f.write("\n}\n")
    f.close()


def make_project(path, name, classes_data):

    init_project(name, path)

    for class_data in classes_data:
        class_name = class_data.name
        class_path = path + '/' + name + '/' + class_name + '.java'

        write_class_object_to_file(class_data, class_path)
