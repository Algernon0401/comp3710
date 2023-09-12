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

path = TRAIN_INPUT_PATH
txt_path = DATA_PATH + TRAIN_TXT
# path = VALID_INPUT_PATH
# txt_path = DATA_PATH + VALID_TXT
# path = TEST_INPUT_PATH
# txt_path = DATA_PATH + TEST_TXT

file_list = []
filelist = os.listdir(path) # all files (and folders) in this dir
count=0
write_file = open(txt_path, "w") # open file (readonly)

for file in os.listdir(path):   # iterate through all files
    filename=os.path.splitext(file)[0]  # filename (without .png)
    # filename_no_pre = os.path.splitext(file)[0][4:]
    idx = os.path.splitext(file)[0][15:len(os.path.splitext(file)[0])-4]
    filetype = os.path.splitext(file)[1]   # .png
    # new_pair = os.path.join(filename + filetype + ' ' + SEG_PREFIX + filename_no_pre + filetype)
    new_pair = os.path.join(filename + filetype + ' ' + idx)
    file_list.append(new_pair)
    count+=1

number_of_lines = len(file_list) # number of elements in list
print('file_list1:',file_list)
file_list.sort(key=lambda item:len(str(item)), reverse=False) # sorting
print('file_list:',file_list)
for current_line in range(number_of_lines):
    write_file.write(file_list[current_line] + '\n')  # close file