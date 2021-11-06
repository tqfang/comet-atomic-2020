import pandas as pd

cskb = pd.read_csv('kg\\atomic2020_data-feb2021\\train.tsv', sep='\t', header=None)

head = list(cskb[0])
tail = list(cskb[2]) # check na value: np.array(range(len(tail)))[tail.isna()] 
relation = cskb[1]
inverted_relation = list(relation.apply(lambda x:x+' Inversed'))
relation = list(relation)

pd.DataFrame(data={
        "header":head+tail,
        "relation":relation+inverted_relation,
        "tail":tail+head
    }).to_csv('kg\\atomic2020_data-feb2021\\train_withInversedSample.tsv', header=False, index=False, sep='\t')

    