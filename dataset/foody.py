import csv
path='/home/phivantuan/Documents/tiki/'
with open('review_train.txt','w') as review_file:
  with open('label_train.txt','w') as label_file:
    with open(path+'negative_data.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            review_file.write(row[1]+"\n")
            label_file.write('0\n')
    with open(path + 'neural_data.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            review_file.write(row[1] + "\n")
            label_file.write('1\n')
    with open(path+'positive_data.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            review_file.write(row[1]+"\n")
            label_file.write('2\n')
