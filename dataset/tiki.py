import os
import json
import re


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

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))
oneStar = 0
twoStar = 0
threeStar = 0
fourStar = 0
fiveStar = 0
# label=""
with open('negative_train.txt', 'w') as negative_train:
    with open('negative_test.txt', 'w') as negative_test:
        with open('negative_valid.txt', 'w') as negative_valid:
            with open('positive_train.txt', 'w') as positive_train:
                with open('positive_test.txt', 'w') as positive_test:
                    with open('positive_valid.txt', 'w') as positive_valid:
                        with open('neutral_train.txt', 'w') as neutral_train:
                            with open('neutral_test.txt', 'w') as neutral_test:
                                with open('neutral_valid.txt', 'w') as neutral_valid:
                                    for f in files:
                                        try:
                                            # print(f)
                                            data = json.loads(open(f).read())
                                            negative = data['1']
                                            negative.extend(data['2'])
                                            write_file(negative_train,negative_test,negative_valid,negative)
                                            neu = data['4']
                                            neu.extend(data['3'])
                                            write_file(neutral_train, neutral_test, neutral_valid, neu)
                                            positive = data['5']
                                            write_file(positive_train, positive_test, positive_valid, positive)
                                            oneStar += len(negative)
                                            fourStar += len(neu)
                                            fiveStar += len(positive)
                                            print(oneStar)
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
# # with open('review_test.txt','w') as review:
# #         for f in files:
# #             try:
# #                 # print(f)
# #                 data = json.loads(open(f).read())
# #                 negative = data['1']
# #                 negative.extend(data['2'])
# #                 negative.extend(data['3'])
# #                 neu = data['4']
# #                 positive = data['5']
# #                 oneStar += len(negative)
# #                 fourStar += len(neu)
# #                 fiveStar += len(positive)
# #                 print(oneStar)
# #                 for one in negative:
# #                     one=" ".join(one.splitlines())
# #                     review.write(one +'\n')
# #                     label+="0\n"
# #                 if (fourStar < 135000):
# #                     for n in neu:
# #                         n=" ".join(n.splitlines())
# #                         review.write(n + '\n')
# #                         label += "1\n"
# #                 if fiveStar < 135000:
# #                     for p in positive:
# #                         p=" ".join(p.splitlines())
# #                         review.write(p + '\n')
# #                         label += "2\n"
# #                 # print("oneStar :  " + str(oneStar) + "  twoStar :  " + str(twoStar) + "  threeStar :  " + str(
# #                 #     threeStar) + "  fourStar :  " + str(fourStar) + "  fiveStar :  " + str(fiveStar))
# #             except:
# #                 print("error " + f)
#
# # open('label_test.txt','w').write(label)
# print("FINAL :   oneStar :  " + str(oneStar) + "  fourStar :  " + str(fourStar) + "  fiveStar :  " + str(fiveStar))
# file=open('negative.txt')
# with open('review_train.txt', 'w') as review_file:
#     with open('review_test.txt', 'w') as review_test_file:
#         with open('negative.txt') as ne_file:
#             with open('positive.txt') as po_file:
#                 with open('neutral.txt') as net_file:
#                     with open('label_train.txt', 'w') as lb_train:
#                         with open('label_test.txt', 'w') as lb_test:
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
