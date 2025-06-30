from pathlib import Path
from glob import glob
import shutil
import os

DIR_PATH = Path(__file__).parent.resolve()

def perform_split_by_moving(data_path, filter, split=(7,2,1)):
    print("Splitting dataset by moving files to subfolders.")

    train_path = Path(f"{data_path}/train")
    train_path.mkdir(parents=True, exist_ok=True)

    val_path = Path(f"{data_path}/val")
    val_path.mkdir(parents=True, exist_ok=True)

    test_path = Path(f"{data_path}/test")
    test_path.mkdir(parents=True, exist_ok=True)

    files = sorted(glob(f"{data_path}/{filter}"))
    for i, path in enumerate(files):
        i = i % sum(split)
        filename = os.path.basename(path)

        if i < split[0]:
            shutil.move(path, f"{train_path}/{filename}")
        elif i < split[0] + split[1]:
            shutil.move(path, f"{val_path}/{filename}")
        else:
            shutil.move(path, f"{test_path}/{filename}")

    print("Done.")

        
if __name__=="__main__":
    data_path = f"{DIR_PATH}/data/language_table/robotics/language_table/captions"
    perform_split_by_moving(data_path=data_path, filter="*.tfrecord*", split=(7,2,1))

    # data_path = f"{DIR_PATH}/data/furniture_bench/low/lamp"
    # perform_split_by_moving(data_path=data_path, filter="*.pkl", split=(7,2,1))
    print("Done preprocessing.")
                