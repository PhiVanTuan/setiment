import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_text(array):
    len_arr = (len(array))
    array_train = []
    array_test = []
    array_valid = []
    if (len_arr > 0):
        x_train = (round)(len_arr * 0.7)
        array_train = array[:x_train]
        x_test = (round)((len_arr - x_train) * 0.67)
        if (x_test > 0):
            array_test = array[x_train:x_train + x_test]

            x_valid = len_arr - x_train - x_test
            if (x_valid > 0):
                array_valid = array[-x_valid:]

    return [array_train, array_test, array_valid]


def write_file(file_train, file_test, file_valid, array):
    text = get_text(array)
    text_train = text[0]
    text_test = text[1]
    text_valid = text[2]
    for train in text_train:
        one = " ".join(train.splitlines())
        file_train.write(one + '\n')
    for test in text_test:
        one = " ".join(test.splitlines())
        file_test.write(one + '\n')
    for valid in text_valid:
        one = " ".join(valid.splitlines())
        file_valid.write(one + '\n')


path = '/home/phivantuan/Documents/craw2/'
path2 = '/home/phivantuan/Documents/data_train/test/neg/'
path3 = '/home/phivantuan/Documents/data_train/test/pos/'
path4 = '/home/phivantuan/Documents/tiki/review_train.txt'
files = []
files2 = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))


# for r, d, f in os.walk(path3):
#     for file in f:
#         if '.txt' in file:
#             files2.append(os.path.join(r, file))
# sum=0
# with open(path4) as review_file:
#     for line in review_file:
#         sum+=len(line.split())
# print(sum/70000)
def get_review():
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0
    # label=""
    with open('oneStar_train.txt', 'w') as oneStar_train:
        with open('oneStar_test.txt', 'w') as oneStar_test:
            with open('oneStar_valid.txt', 'w') as oneStar_valid:
                with open('twoStar_train.txt', 'w') as twoStar_train:
                    with open('twoStar_test.txt', 'w') as twoStar_test:
                        with open('twoStar_valid.txt', 'w') as twoStar_valid:
                            with open('threeStar_train.txt', 'w') as threeStar_train:
                                with open('threeStar_test.txt', 'w') as threeStar_test:
                                    with open('threeStar_valid.txt', 'w') as threeStar_valid:
                                        with open('fourStar_train.txt', 'w') as fourStar_train:
                                            with open('fourStar_valid.txt', 'w') as fourStar_valid:
                                                with open('fourStar_test.txt', 'w') as fourStar_test:
                                                    with open('fiveStar_train.txt', 'w') as fiveStar_train:
                                                        with open('fiveStar_valid.txt', 'w') as fiveStar_valid:
                                                            with open('fiveStar_test.txt', 'w') as fiveStar_test:
                                                                for f in files:
                                                                    try:
                                                                        # print(f)
                                                                        data = json.loads(open(f).read())
                                                                        oneStar = data['1']
                                                                        write_file(oneStar_train, oneStar_test,oneStar_valid,oneStar)
                                                                        twoStar = data['2']
                                                                        write_file(twoStar_train, twoStar_test,
                                                                                   twoStar_valid, twoStar)
                                                                        threeStar = data['3']
                                                                        write_file(threeStar_train, threeStar_test,
                                                                               threeStar_test, threeStar)
                                                                        fourStar=data['4']
                                                                        write_file(fourStar_train, fourStar_test,
                                                                               fourStar_valid, fourStar)
                                                                        fiveStar = data['5']

                                                                        write_file(fiveStar_train, fiveStar_test,
                                                                                   fiveStar_valid,
                                                                                   fiveStar)
                                                                        one+=len(oneStar)
                                                                        two+=len(twoStar)
                                                                        three+=len(threeStar)
                                                                        four+=len(fourStar)
                                                                        five+=len(fiveStar)
                                                                        print(str(one)+" : "+str(two)+" : "+str(three)+" : "+str(four)+" : "+str(five))
                                                                        # for one in negative:
                                                                        #     one = " ".join(one.splitlines())
                                                                        #     ne_file.write(one + '\n')
                                                                        #
                                                                        # for n in neu:
                                                                        #     n = " ".join(n.splitlines())
                                                                        #     nt_file.write(n + '\n')
                                                                        #
                                                                        # for p in positive:
                                                                        #     p = " ".join(p.splitlines())
                                                                        #     po_file.write(p + '\n')


                                                                    except:
                                                                        print("error " + f)


def get_word2vec_file():
    oneStar = 0
    with open('review_train.txt', 'w') as review:
        for f in files:
            try:
                # print(f)
                data = json.loads(open(f).read())
                negative = data['1']
                negative.extend(data['2'])
                negative.extend(data['3'])
                neu = data['4']
                positive = data['5']
                oneStar += len(negative)

                print(oneStar)
                for one in negative:
                    one = " ".join(one.splitlines())
                    review.write(one + '\n')
                for n in neu:
                    n = " ".join(n.splitlines())
                    review.write(n + '\n')

                for p in positive:
                    p = " ".join(p.splitlines())
                    review.write(p + '\n')

                # print("oneStar :  " + str(oneStar) + "  twoStar :  " + str(twoStar) + "  threeStar :  " + str(
                #     threeStar) + "  fourStar :  " + str(fourStar) + "  fiveStar :  " + str(fiveStar))
            except:
                print("error " + f)


def word2vec_file():
    with open('review_tiki.txt', 'w') as out_file:
        with open('review_train.txt') as in_file:
            for text in in_file:
                text = text.lower()
                text = re.sub(r'[^\w\s]', ' <punct> ', text)
                text = re.sub(r'\d+', ' <number>', text)
                text = re.sub(r'\n', ' ', text)
                text = re.sub('\s+', ' ', text)
                out_file.write(text)

def write_label_file(file,label_file,review_file,label):
    for index,line in enumerate(file):
        if(index<5000):
            review_file.write(line)
            label_file.write(label+"\n")
        else:break
def get_label():
    with open('label_test.txt', 'w') as label_train:
        with open('review_test.txt', 'w') as review_train:
            with open('oneStar_test.txt') as oneStar:
                with open('twoStar_test.txt') as twoStar:
                    with open('threeStar_test.txt') as threeStar:
                        with open('fourStar_test.txt') as fourStar:
                            with open('fiveStar_test.txt') as fiveStar:
                                write_label_file(oneStar,label_train,review_train,"0")
                                write_label_file(twoStar,label_train,review_train,"1")
                                write_label_file(threeStar,label_train,review_train,"2")
                                write_label_file(fourStar,label_train,review_train,"3")
                                write_label_file(fiveStar,label_train,review_train,"4")
                                # for index,line in enumerate()
                                # for line in ne_file:
                                #     review_train.write(line)
                                #     label_train.write("0\n")
                                # for line in po_file:
                                #     review_train.write(line)
                                #     label_train.write("1\n")
get_label()

def get_vlsp():
    with open('review_train.txt', 'w') as review_train:
        with open('label_train.txt', 'w') as label_train:
            # with open('review_valid.txt','w') as review_valid:
            #     with open('label_valid.txt','w') as label_valid:
            #         with open('review_test.txt','w') as review_test:
            #             with open('label_test.txt','w') as label_test:
            with open('/home/phivantuan/Documents/SA2016-training_data/SA2016-training_data/negative.txt') as ne_file:
                with open(
                        '/home/phivantuan/Documents/SA2016-training_data/SA2016-training_data/neutral.txt') as neu_file:
                    with open(
                            '/home/phivantuan/Documents/SA2016-training_data/SA2016-training_data/positive.txt') as po_file:
                        for f in ne_file:
                            if f.strip():
                                review_train.write(f)
                                label_train.write("0\n")
                        for f in neu_file:
                            if f.strip():
                                review_train.write(f)
                                label_train.write("1\n")
                        for f in po_file:
                            if f.strip():
                                review_train.write(f)
                                label_train.write("2\n")


def get_length():
    sum = 0
    with open('/home/phivantuan/Documents/vn_use/train_sens.txt') as in_file:
        reviews_len = [len(x.split()) for x in in_file]

        pd.Series(reviews_len).hist()
        plt.show()
        print(pd.Series(reviews_len).describe())
        # for index,line in enumerate(in_file):
        #     sum+=len(line.split())
        #     print(sum/(index+1))


def vlsp_valid():
    with open('review_valid.txt', 'w') as re_file:
        with open('label_valid.txt', 'w') as la_file:
            with open('review_train.txt') as train_file:
                with open('review_train.txt', 'w') as re_train:
                    with open('label_train.txt', 'w') as la_train:
                        for index, line in enumerate(train_file):
                            if index < 170:
                                re_file.write(line)
                                la_file.write('0\n')
                            elif index < 1700:
                                re_train.write(line)
                                la_train.write('0\n')
                            elif index < 1870:
                                re_file.write(line)
                                la_file.write('1\n')
                            elif index < 3400:
                                re_train.write(line)
                                la_train.write('1\n')
                            elif index < 3570:
                                re_file.write(line)
                                la_file.write('2\n')
                            else:
                                re_train.write(line)
                                la_train.write('2\n')


def vlsp_test():
    with open('review_test.txt', 'w') as re_file:
        with open('label_test.txt', 'w') as la_file:
            with open('/home/phivantuan/Documents/test_raw_ANS.txt') as file:
                for index, line in enumerate(file):
                    if line.strip() == 'POS':
                        la_file.write('2\n')
                    elif line.strip() == 'NEU':
                        la_file.write('1\n')
                    elif line.strip() == 'NEG':
                        la_file.write('0\n')
                    else:
                        re_file.write(line)

# get_review()

# word2vec_file()
# # open('label_train.txt','w').write(label)
# print("FINAL :   oneStar :  " + str(oneStar) + "  fourStar :  " + str(fourStar) + "  fiveStar :  " + str(fiveStar))
# file=open('negative.txt')
# with open('review_train.txt', 'w') as review_file:
#     with open('review_train.txt', 'w') as review_test_file:
#         with open('negative.txt') as ne_file:
#             with open('positive.txt') as po_file:
#                 with open('neutral.txt') as net_file:
#                     with open('label_train.txt', 'w') as lb_train:
#                         with open('label_train.txt', 'w') as lb_test:
#                             for index, line in enumerate(ne_file):
#                                 if (index < 100000):
#                                     review_file.write(line)
#                                     lb_train.write("0\n")
#                                 elif index < 135000:
#                                     review_test_file.write(line)
#                                     lb_test.write("0\n")
#                             for index, line in enumerate(po_file):
#                                 if (index < 100000):
#                                     review_file.write(line)
#                                     lb_train.write("2\n")
#                                 elif index < 135000:
#                                     review_test_file.write(line)
#                                     lb_test.write("2\n")
#                             for index, line in enumerate(net_file):
#                                 if (index < 100000):
#                                     review_file.write(line)
#                                     lb_train.write("1\n")
#                                 elif index < 135000:
#                                     review_test_file.write(line)
#                                     lb_test.write("1\n")

# sum_line = 0;
# with open('review_train.txt') as infile:
#     for text in infile:
#         text = text.lower()
#         text = re.sub(r'[^\w\s]', ' ', text)
#         text = re.sub(r'\d+', ' <number>', text)
#         text = re.sub(r'\n', ' ', text)
#         text = re.sub('\s+', ' ', text)
#         sum_line += len(text.split())
# print(sum_line / 300000)


# print(get_text([1, 2, 3, 4, 5]))
