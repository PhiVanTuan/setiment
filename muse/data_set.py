with open("/home/phivantuan/Documents/vn_use/dev_vecs.txt") as file:
    for index,line in enumerate(file):
        path='/home/phivantuan/Documents/testtranslate/'+str(index)+".txt"
        with open(path,'w') as out_file:
            out_file.write(line)
        print(index)
