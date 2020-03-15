import pandas as pd
import numpy as np
import joblib
import glob
from tqdm import tqdm
import os
SHUTDOWN = True

if __name__ == "__main__":
    files = glob.glob("../input/train_*.parquet")
    for _,f in enumerate(files):
        # if _ == 3:
        df = pd.read_parquet(f)
        image_ids = df.image_id.values
        df = df.drop(["image_id"],axis = 1)

        image_arry = df.values
        for j, image_id in tqdm(enumerate(image_ids), total= len(image_ids)):
            joblib.dump(image_arry[j], f"../input/image_pickles/{image_id}.pkl")
    