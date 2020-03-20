import numpy as np
from numpy.linalg import norm

# num_lines = sum(1 for line in open('/home/phivantuan/Documents/vn_use/train_vecs.txt'))
# print(num_lines)
# array = [1, 2, 3, 4, 5]
# b = [1, 2, 3, 4, 5]
# x = np.dot(array, b)
# y = norm(array) * norm(b)
# print(x / y)
# index = 0
# array = []
# result = []


# num_lines = sum(1 for line in file_object)
# print(num_lines)
# with open("/home/phivantuan/Documents/vn_use/train_vecs.txt") as file:
#     for index,line in enumerate(file):
#         path='/home/phivantuan/Documents/translate/'+str(index)+".txt"
#         with open(path,'w') as out_file:
#             out_file.write(line)
#         print(index)
#         if index==1172716:break
        # result.append([])
        # x = np.array(line).astype(np.float)
        # array.append(x)
        # for i in range(len(array)):
        #     value = array[i]
        #     if (i + 1 == len(array)):
        #         data = 1.0
        #     else:
        #         data = np.dot(x, value) / (norm(value) * norm(x))
        #         result[len(result) - 1].append(data)
        #     result[i].append(data)
        # index += 1
        # if index % 40 == 0:
        #     text_file='output'+str(int (index/40))+'.txt'
        #     np.savetxt(text_file, np.array(result))
        #     array = []
        #     result = []
        #     print(text_file)
        #     break