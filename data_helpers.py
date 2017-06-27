#那个打分程序报错的话，补上这个缺的函数

def load_data(lable_data_file):
    lable_examples = list(open(lable_data_file, "r", -1, "utf-8").readlines())
    lable_examples = [s.strip() for s in lable_examples]
    x_text = lable_examples
    x_text2 = [clean_str(sent) for sent in x_text]
    x_text=[sent for sent in x_text]
    return[x_text2,x_text]

