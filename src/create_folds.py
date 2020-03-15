import pandas as pd
import glob
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")
    print(df.head())

    df.loc[:,"kfold"] = -1

    df = df.sample(frac=1).reset_index(drop = True)

    X = df.image_id.values
    Y = df[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]].values

    mskf = MultilabelStratifiedKFold( n_splits = 5 )

    for fold, (trn_, val_) in enumerate(mskf.split(X,Y)):
        print( "TRAIN: ", trn_, "VALIDATION: ", val_ )
        df.loc[val_, "kfold"] = fold

    print( df.kfold.value_counts() )
    df.to_csv("../input/train_folds.csv", index = False)