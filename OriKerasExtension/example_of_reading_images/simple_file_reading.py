
from OriKerasExtension.ThesisHelper import readCompleteMatFile, ExtractDataVer4
import numpy as np
from keras.utils.np_utils import to_categorical


def downsample_data(data, number_of_original_samples, down_samples_param):
    new_number_of_time_stamps = number_of_original_samples / down_samples_param

    temp_data_for_eval = np.zeros((data.shape[0], new_number_of_time_stamps, data.shape[2]))

    for new_i, i in enumerate(range(0, number_of_original_samples, down_samples_param)):
        temp_data_for_eval[:, new_i, :] = np.mean(data[:, range(i, (i + down_samples_param)), :], axis=1)
    return temp_data_for_eval


def create_train_data(data_from_mat, down_samples_param, indexes):
    all_positive_train = []
    all_negative_train = []

    last_time_stamp = 800
    fist_time_stamp = -200

    data_for_eval = ExtractDataVer4(data_from_mat['all_relevant_channels'], data_from_mat['marker_positions'],
                                    data_from_mat['target'], fist_time_stamp, last_time_stamp)

    temp_data_for_eval = downsample_data(data_for_eval[0],data_for_eval[0].shape[1], down_samples_param)

    positive_train_data_gcd = temp_data_for_eval[
        np.all([indexes, data_from_mat['target'] == 1], axis=0)]
    negative_train_data_gcd = temp_data_for_eval[
        np.all([indexes, data_from_mat['target'] == 0], axis=0)]
    all_positive_train.append(positive_train_data_gcd)
    all_negative_train.append(negative_train_data_gcd)

    positive_train_data_gcd = np.vstack(all_positive_train)
    negative_train_data_gcd = np.vstack(all_negative_train)

    all_data = np.vstack([positive_train_data_gcd, negative_train_data_gcd])

    all_tags = np.vstack(
        [np.ones((positive_train_data_gcd.shape[0], 1)), np.zeros((negative_train_data_gcd.shape[0], 1))])
    categorical_tags = to_categorical(all_tags)

    shuffeled_samples, suffule_tags = (all_data, categorical_tags)
    return shuffeled_samples, suffule_tags



if __name__ == "__main__":
    file_name = r'C:\Users\ORI\Documents\Thesis\dataset_all\RSVP_Color116msVPfat.mat'

    # read the raw data
    raw_data = readCompleteMatFile(file_name)
    train_data, train_tag = create_train_data(raw_data, 8, raw_data['train_mode'] == 1)
    test_data, test_tag = create_train_data(raw_data, 8, raw_data['train_mode'] != 1)

    # now, extract epoch

    print "done"
