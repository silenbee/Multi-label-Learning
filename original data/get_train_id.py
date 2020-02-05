import os
import xml.etree.ElementTree as ET

#  xml_path = './VOCdevkit/VOC2007/Annotations/'
xml_path = './VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/'
def get_xml_id(folder_name, xml_list):
    for _,dir,files in os.walk(folder_name):
        for file in files:
            if os.path.splitext(file)[-1] == '.xml':
                xml_list.append(os.path.splitext(file)[-2])


def write_map(folder_name, file_name):
    # for _,dir,files in os.walk(folder_name):
    #     for file in files:
    #         with open(os.path(file), 'r') as f:
    listdir = os.listdir(folder_name)
    for file in listdir:
        nameset = set()
        if file.endswith('xml'):
            file_id = os.path.splitext(file)[-2]
            file = os.path.join(folder_name, file)
            tree = ET.parse(file)
            root = tree.getroot()
            for xml_object in root.findall('object'):
                for name in xml_object.findall('name'):
                    nameset.add(name.text)
            namelist = list(nameset)
            with open(os.path.join('./', file_name), 'a+') as f:
                f.write(file_id+'\t')
                if len(nameset) > 1: 
                    for name in namelist[:-1]:
                        f.write(name+',')
                    f.write(namelist[-1])
                else:
                    f.write(namelist[0])
                f.write('\n')

                
# xml_list = []
# get_xml_id(xml_path, xml_list)
# print(len(xml_list))
# print(xml_list[:100])

write_map(xml_path,'2012map.txt')