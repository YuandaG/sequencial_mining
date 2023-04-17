# !/usr/bin/env python3
"""rosbag解析"""

from collections import defaultdict
import tqdm
import numpy as np
import rosbag

def get_zip_msg(bag, args):
    """
    按topic解析message
    """
    return [bag.read_messages(topics=[arg]) for arg in args]


def parse_bag(bag_path):
    """
    解析rosbag文件，主要处理函数
    """
    bag = rosbag.Bag(bag_path)
    print(' - filename          : ', bag.filename)
    print(' - get_message_count : ', bag.get_message_count())
    print(' - start_time        : ', bag.get_start_time())
    print(' - end_time          : ', bag.get_end_time())
    print(' - size              : ', bag.size)
    print(' - mode              : ', bag.mode)
    print('\n\n')
    topics = []
    print(' - get_message_count : ', bag.get_message_count())
    with tqdm.tqdm_notebook(total=bag.get_message_count()) as pbar:
        for idx, (topic, msg, t) in enumerate(bag.read_messages()):
            pbar.update(1)
            topics.append(topic)
    topics = np.array(topics)
    print(' - topics : ', np.unique(topics))
    result_dict = defaultdict(list)
    # for imu_data, vcan_data, gps_data in zip(*get_zip_msg(bag, ['/imu/data_raw', '/vcan', '/novatel/oem7/inspva'])): 
    #     result_dict['imu'].append(imu_data)
    #     result_dict['vcan'].append(vcan_data)
    #     result_dict['gps'].append(gps_data)
    for vcan_data, pic in zip(*get_zip_msg(bag, ['/vcan'])): 
        result_dict['vcan'].append(vcan_data)
        # result_dict['pic'].append(pic)
    return bag.get_start_time(), bag.get_end_time(), result_dict 
