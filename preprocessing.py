import argparse
import glob
from torch.utils.data import random_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, required=False, default='')
    parser.add_argument('-r', '--train_ratio', type=float, required=False, default=0.7)
    args = parser.parse_args()

    total_data_list = glob.glob(args.data_dir + "/*.jpg", recursive=True)

    train_size = int(args.train_ratio * len(total_data_list))
    test_size = len(total_data_list) - train_size
    train_set, test_set = random_split(total_data_list, [train_size, test_size])
    print(test_set)
    with open('test_set_images.ctrl', 'w') as f:
        for img_name in test_set:
            print(img_name)
            f.write(img_name + '\n')

    with open('train_set_images.ctrl', 'w') as f:
        for img_name in train_set:
            print(img_name)
            f.write(img_name + '\n')
