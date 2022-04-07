from pyparsing import line

def get_neg_train_data():
    text = []
    with open('./uniprot-reviewed_yes.txt') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            if lines[i][0]=='>':
                continue
            else:
                text.append(' '.join(list(lines[i].split()[0])))
    return text
