from pyparsing import line

def get_pos_test_data():
    text = []
    with open('./test/test_pos-K.txt') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            if i%2==1:
                text.append(' '.join(list(lines[i].split()[0])))
    with open('./test/test_pos-P.txt') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            if i%2==1:
                text.append(' '.join(list(lines[i].split()[0])))
    with open('./test/test_pos-R.txt') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            if i%2==1:
                text.append(' '.join(list(lines[i].split()[0])))
    with open('./test/test_pos-T.txt') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            if i%2==1:
                text.append(' '.join(list(lines[i].split()[0])))
    return text

def get_neg_test_data():
    text = []
    with open('./test/test_neg-K.txt') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            if i%2==1:
                text.append(' '.join(list(lines[i].split()[0])))

    with open('./test/test_neg-P.txt') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            if i%2==1:
                text.append(' '.join(list(lines[i].split()[0])))
    
    with open('./test/test_neg-R.txt') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            if i%2==1:
                text.append(' '.join(list(lines[i].split()[0])))
    
    with open('./test/test_neg-T.txt') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            if i%2==1:
                text.append(' '.join(list(lines[i].split()[0])))
    
    return text


def get_pos_train_data():
    text = []
    with open('./train/train_pos-K.txt') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            if i%2==1:
                text.append(' '.join(list(lines[i].split()[0])))
    with open('./train/train_pos-P.txt') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            if i%2==1:
                text.append(' '.join(list(lines[i].split()[0])))
    with open('./train/train_pos-R.txt') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            if i%2==1:
                text.append(' '.join(list(lines[i].split()[0])))
    with open('./train/train_pos-T.txt') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            if i%2==1:
                text.append(' '.join(list(lines[i].split()[0])))
    return text

def get_neg_train_data():
    text = []
    with open('./train/train_neg-K.txt') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            if i%2==1:
                text.append(' '.join(list(lines[i].split()[0])))
    with open('./train/train_neg-P.txt') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            if i%2==1:
                text.append(' '.join(list(lines[i].split()[0])))
    with open('./train/train_neg-R.txt') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            if i%2==1:
                text.append(' '.join(list(lines[i].split()[0])))
    with open('./train/train_neg-T.txt') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            if i%2==1:
                text.append(' '.join(list(lines[i].split()[0])))
    return text

def get_train_data():
    return get_pos_train_data()+get_neg_train_data()

def get_test_data():
    return get_pos_test_data()+get_neg_test_data()

def get_data():
    return get_train_data()+get_test_data()