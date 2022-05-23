from os.path import join
from codecs import open


def build_corpus(split, make_vocab=True, data_dir="./ResumeNER"):
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            s = line.strip('\n').split()
            if s != []:
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []


    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_corpus_(split, make_vocab=True, data_dir="./data_test"):
    """读取数据"""
    assert split in ['train_rand_extract', 'dev_rand_extract', 'test_rand_extract']

    data_x = []
    data_y = []
    with open(join(data_dir, split+".txt"), 'r', encoding='utf-8') as file:
        text_data_list = file.readlines() # 读取所有的数据成列表，每一行内容为列表一个元素，包含换行符，换行符全部转化为'\n'
        k = 0 # 此为data_x, data_y 的第一层元素定位
        for i in range(len(text_data_list)): # 遍历所有的元素，将其分成两个列表
            if text_data_list[i] != '\n' and '-seq' in text_data_list[i]:
                text_data_cell = text_data_list[i].rstrip('\n').split('-seq-')
                # print(len(text_data_cell[0].split(' ')) == len(text_data_cell[-1].split(' ')))
                data_x.append(text_data_cell[0].split(' '))
                data_y.append(text_data_cell[-1].split(' '))
    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(data_x)
        tag2id = build_map(data_y)
        return data_x, data_y, word2id, tag2id
    else:
        return data_x, data_y


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps

if __name__=='__main__':
    # build_corpus_("train_rand_extract")
    build_corpus("train")