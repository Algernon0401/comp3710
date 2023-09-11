import os

# IO Paths
DATA_PATH = './data/keras_png_slices_data/'
TRAIN_INPUT_PATH = DATA_PATH + 'keras_png_slices_train'
VALID_INPUT_PATH = DATA_PATH + 'keras_png_slices_validate'
TEST_INPUT_PATH = DATA_PATH + 'keras_png_slices_test'
VALID_TARGET_PATH = DATA_PATH + 'keras_png_slices_seg_validate'
TRAIN_TARGET_PATH = DATA_PATH + 'keras_png_slices_seg_train'
TEST_TARGET_PATH = DATA_PATH + 'keras_png_slices_seg_test'
TRAIN_TXT = './oasis_train.txt'
VALID_TXT = './oasis_valid.txt'
TEST_TXT = './oasis_test.txt'
SEG_PREFIX = 'seg'

path = VALID_INPUT_PATH
txt_path = DATA_PATH + VALID_TXT

file_list = []
filelist = os.listdir(path) #该文件夹下所有的文件（包括文件夹）
count=0
write_file = open(txt_path, "w") #以只写方式打开write_file_name文件

for file in os.listdir(path):   #遍历所有文件
    filename=os.path.splitext(file)[0]  #文件名
    filename_no_pre = os.path.splitext(file)[0][4:]
    filetype = os.path.splitext(file)[1]   #文件扩展名
    new_pair = os.path.join(filename + filetype + ' ' + SEG_PREFIX + filename_no_pre + filetype)
    file_list.append(new_pair)
    count+=1

number_of_lines = len(file_list)#列表中元素个数
print('file_list1:',file_list)
file_list.sort(key=lambda item:len(str(item)), reverse=False)#排序
print('file_list:',file_list)
print(type(file_list))
for current_line in range(number_of_lines):
    write_file.write(file_list[current_line] + '\n')  # 关闭文件