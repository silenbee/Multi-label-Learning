import os
from collections import Counter

name = ""
tags = []
txtdir = "./"
gcount = Counter()

def get_counter():
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    with open('./img_tag_2012.txt','r') as file:
        for line in file:
            # id,tokens=json.loads(line)
            tokens = line.split()[1].split(',')
            counter.update(tokens)
    # for word, cnt in counter.items():
    #     print(word,':',cnt)
    global gcount
    gcount = counter


def takefreq(elem):
    global gcount
    return gcount[elem]


def write_txt(folder_name, old_name, new_name):
    with open(os.path.join(folder_name, new_name), 'a+') as fw:
        with open(os.path.join(folder_name, old_name)) as f:
            lines = f.readlines()
            for line in lines:
                name = line.split()[0]
                tags = line.split()[1].split(',')
                tags.sort(key=takefreq,reverse=True)
                fw.write(name+'\t')
                if len(tags) > 1: 
                    for tag in tags[:-1]:
                        fw.write(tag+',')
                    fw.write(tags[-1])
                else:
                    fw.write(tags[0])
                fw.write('\n')

            
get_counter()
write_txt(txtdir, "img_tag_2012.txt", "re_img_tag.txt")