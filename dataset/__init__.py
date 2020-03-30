# import numpy as np
# from numpy.linalg import norm
#
# def matrix_(file):
#     result = []
#     array = []
#     for line in file:
#         result.append([])
#         x = np.array(line).astype(np.float)
#         array.append(x)
#         for i in range(len(array)):
#             value = array[i]
#             if (i + 1 == len(array)):
#                 data = 1.0
#             else:
#                 data = np.dot(x, value) / (norm(value) * norm(x))
#                 result[len(result) - 1].append(data)
#             result[i].append(data)
#     return np.array(result)
# label=np.arange(start=0, stop=40, step=1)
# print(label)
# array=[]
# # for index in label:
# #     path = '/home/phivantuan/Documents/translate/' + str(index) + ".txt"
# #     with open(path) as label_file:
# #         vec_line=np.array(label_file.read().split()).astype(float)
# #         array.append(vec_line)
# # result=matrix_(array)
# # np.savetxt("result.txt",result)
# result= np.loadtxt('result.txt')
# vec1=np.array(open('/home/phivantuan/Documents/translate/13.txt').read().split()).astype(float)
# vec2=np.array(open('/home/phivantuan/Documents/translate/29.txt').read().split()).astype(float)
# test= np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
# test1=result[29][13]
# print(test-test1)