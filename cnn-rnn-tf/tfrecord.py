import tensorflow as tf
from lxml import etree
import os
from PIL import Image
 
#定义解析单个lxml文件
def parse_xml(xml_path,name_label_map):
    tree = etree.parse(xml_path)
    dict = {}
    for x in tree.xpath('//filename'):
        if len(x):
            print("error")
        else:
            dict["image_"+x.tag] = x.text
    for x in tree.xpath('//size'):
        for x1 in x.getchildren():
            dict["image_"+x1.tag] = x1.text
    object_numbers = 0
    #可能存在多个object节点，即多目标检测
    for obj in tree.xpath('//object'):
        #获取当前object节点的子节点
        for x in obj.getchildren():
            #判断节点x是否有子节点
            if len(x):
                if x.tag == 'bndbox':
                    for bbox in x.getchildren():
                        dict['object_'+str(object_numbers)+'/bndbbox/'+bbox.tag] = bbox.text
                else:
                    pass
            elif x.tag == 'name':
                #将name与id均保存到字典中
                dict["object_"+str(object_numbers)+"/"+x.tag] = x.text
                dict["object_" + str(object_numbers) + "/id" ] = name_label_map[x.text]
            else:
                pass
        object_numbers += 1
    dict['object_number'] = object_numbers
    return dict
#将name与label的映射map文件解析为字典格式
# name<---->id
def parse_map_file(map_file):
    name_label_map = {}
    with open(map_file) as f:
        id = 0
        for line in f.readlines():
            if len(line) > 1:
                if line.find('id') != -1:
                    line = line.strip('\\n')
                    line = line.strip(' ')
                    colon = line.index(':')
                    colon = colon + 1
                    line_id = line[colon:].strip(' ')
                    id = int(line_id)
                elif line.find('name') != -1:
                    line = line.strip('\\n').strip(' ')
                    first = line.index("'")
                    last = line.rindex("'")
                    line_name = line[first+1:last]
                    name_label_map[line_name]=id
                    id = 0
                else:
                    pass
            else:
                #print("empty line")
                pass
    return name_label_map
 
MAP_FILE = r"D:\models-master\research\object_detection\data\pascal_label_map.pbtxt"
BASE_PATH= r"E:\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations"
BASE_JPEG_PATH = r"E:\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages"
name_label_map = parse_map_file(MAP_FILE)
xml_file_list = os.listdir(BASE_PATH)
train_list = []
test_list = []
j = 0
for i in range(len(xml_file_list)):
    if j % 6 == 0:
        test_list.append(xml_file_list[i])
    else:
        train_list.append(xml_file_list[i])
    j = j + 1
with tf.python_io.TFRecordWriter(path=r"E:\VOCtrainval_11-May-2012\train.tfrecords") as tf_writer:
    for i in range(len(train_list)):
        file_path = os.path.join(BASE_PATH,train_list[i])
        if os.path.isfile(file_path):
            #解析xml为字典格式数据
            xml_dict = parse_xml(file_path,name_label_map)
            image = Image.open(os.path.join(BASE_JPEG_PATH,xml_dict['image_filename']))
            image_bytes = image.tobytes()
            features = {}
            features["image"] = tf.train.Feature(bytes_list=tf.train.BytesList(value = [image_bytes]))
            features['image_width'] = tf.train.Feature(int64_list=tf.train.Int64List(value = [int(xml_dict['image_width'])]))
            features['image_height'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(xml_dict['image_height'])]))
            features['image_depth'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(xml_dict['image_depth'])]))
            features['image/object_number'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[int(xml_dict['object_number'])]))
            xmin = []
            xmax = []
            ymin = []
            ymax = []
            object_id = []
            object_name = []
            object_number = xml_dict['object_number']
            for j in range(object_number):
                object_i = 'object_'+str(j)
                #print(xml_dict[object_i+'/name'])
                #print(type(xml_dict[object_i+'/name']))
                object_name.append(bytes(xml_dict[object_i+'/name'],'utf-8'))
                object_id.append(int(xml_dict[object_i+'/id']))
                xmin.append(float(xml_dict[object_i+'/bndbbox/xmin']))
                xmax.append(float(xml_dict[object_i + '/bndbbox/xmax']))
                ymin.append(float(xml_dict[object_i + '/bndbbox/ymin']))
                ymax.append(float(xml_dict[object_i + '/bndbbox/ymax']))
            #变长数据以list形式存储
            features["image/object/names"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=object_name))
            features['image/object/id'] = tf.train.Feature(int64_list=tf.train.Int64List(value=object_id))
            features['image/object/xmin'] = tf.train.Feature(float_list=tf.train.FloatList(value=xmin))
            features['image/object/xmax'] = tf.train.Feature(float_list=tf.train.FloatList(value=xmax))
            features['image/object/ymin'] = tf.train.Feature(float_list=tf.train.FloatList(value=ymin))
            features['image/object/ymax'] = tf.train.Feature(float_list=tf.train.FloatList(value=ymax))
            tf_features = tf.train.Features(feature=features)
            tf_example = tf.train.Example(features=tf_features)
            tf_serialized = tf_example.SerializeToString()
            tf_writer.write(tf_serialized)