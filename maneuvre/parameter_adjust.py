from read_bag import *
from threshold import *
from scipy.ndimage import gaussian_filter
import json, os, glob, zipfile, itertools
import matplotlib.pyplot as plt

class SelfCarStateEstimation:
    """
    调用类
    """
    def __init__(self, rosbag_path, self_car_state_dict_json, self_car_velo_acce_json):
        self.rosbag_path = glob.glob(rosbag_path)
        self.self_car_state_dict_json = self_car_state_dict_json
        self.self_car_velo_acce_json = self_car_velo_acce_json
        self.self_car_state_dict = []
        self.self_car_state_dict_single = []
        self.self_car_ang_mul_steer_dict = []
        self.distance_dic = []
        self.steer_angle = []
        self.v_record = []
        self.timestamp_dic = []

    def run(self):
        """
        调用类的运行函数
        """
        for idx, file in enumerate(self.rosbag_path):
            s_time, e_time, data_dict = parse_bag(file)
            self.s_time = s_time
            self.e_time = e_time
            # parse_list = data_dict['pic']
            parse_list2 = data_dict['vcan']
            self.interval = (e_time - s_time)/len(parse_list2)
            self.fps = round(1/self.interval, 2)
            print(f'{self.fps}-------------')
            velo_list, acc_list = self.self_car_vcan_based_velo_acce(parse_list2)
            self.self_car_state_dict_single, self_car_ang_mul_steer, distance_dic, steer_angle = self.self_car_vcan_based_state_velo_acc(parse_list2, acc_list)
            self.self_car_state_dict.append(self.self_car_state_dict_single)
            self.self_car_ang_mul_steer_dict.append(self_car_ang_mul_steer)
            self.result_save(file)
            self.distance_dic.append(distance_dic)
            self.steer_angle.append(steer_angle)
            assert 1 == 1
        # with open(self.self_car_state_dict_json, 'w') as f: 
            # json.dump(self.self_car_state_dict, f)

    def self_car_vcan_based_velo_acce(self, vcan_datas):
        """
        基于自车vcan数据输出速度、加速度
        """
        acc_dic = []
        velo_dic = []
        for idx, vcan_data in enumerate(vcan_datas):
            vcan_data = vcan_data[0] 
            if idx == 0:
                prev_v = vcan_data.message.VehicleSpeed
                velo = vcan_data.message.VehicleSpeed
            else:
                prev_v = vcan_datas[idx - 1][0].message.VehicleSpeed
                velo = vcan_data.message.VehicleSpeed
            acce = (velo - prev_v) / self.interval
            velo_dic.append(velo)
            acc_dic.append(acce)

        return velo_dic, acc_dic

    def self_car_vcan_based_state_velo_acc(self, steer_data, acc_dic):
        """
        基于自车imu数据估计状态
        """
        self_car_state_dict = []
        # turning = ['leftTurn', 'rightTurn']
        # changing = ['leftChange', 'rightChange']
        # uturning = ['leftUturn', 'rightUturn']
        # status = ['driveStart', 'driveSlow', 'driveNormal', 'driveStop']
        maneuvre = turning + changing + uturning

        #准备steering、速度数据
        self_car_steer_angle = []
        self_car_vel = []
        accross_distance = 0
        distance_dic = []
        timestamp_dic = []
        for idx, vcan in enumerate(steer_data):
            vcan = vcan[0]
            steeringwheelangle = (vcan.message.SteeringWheelAngle - steer_bias)
            velocity = vcan.message.VehicleSpeed
            accross_distance += velocity * 1 / self.fps # 偏移距离，用来排除低速时打方向盘，但未发生变道行为

            # 重要调整参数steeringwheelangle，开始计算变道、转向、掉头的阈值
            if np.abs(steeringwheelangle) <= accross_distance_threshold:
                accross_distance = 0
            self_car_steer_angle.append(steeringwheelangle)
            self_car_vel.append(velocity)
            distance_dic.append(accross_distance)

            # ßßtimestamp = float(f'{vcan.timestamp.secs}.{vcan.timestamp.nsecs}')
            timestamp = float(f"{str(vcan.timestamp)[:10]}.{str(vcan.timestamp)[10:]}")
            timestamp_dic.append(timestamp)

        self.timestamp_dic = timestamp_dic
        self_car_steer_angle = np.around(self_car_steer_angle, 0)
        self_car_vel = np.around(self_car_vel, 0)
        distance_dic = np.around(distance_dic, 0)
        # high_passed_steer = gaussian_filter(self_car_steer_angle, sigma=sigma_2)
        high_passed_steer = self_car_steer_angle

        counter = 0

        dic_speed = []
        self_car_ang_mul_steer_dict = []
        for pos in range(len(high_passed_steer)):
            angle = high_passed_steer[pos]
            vel = self_car_vel[pos]
            dis = distance_dic[pos]
            acc = acc_dic[pos]
            # 判断起步和速度状态
            if pos == 1:
                if vel == 0:
                    dic_speed.append('driveStop')
                elif vel > 0 and vel < slow_threshold:
                    dic_speed.append('driveSlow')
                elif vel >= slow_threshold:
                    dic_speed.append('driveNormal')
            else:
                vel_his = self_car_vel[pos-1]
                if vel == 0:
                    dic_speed.append('driveStop')
                elif vel > 0 and vel < slow_threshold:
                    if vel_his == 0 or (counter > 0 and counter < start_duration * self.fps):
                        if acc > 0:
                            dic_speed.append('driveStart')
                            counter += 1
                        else:
                            dic_speed.append('driveNormal')
                            counter = 0
                    else:
                        dic_speed.append('driveSlow')

                elif vel >= slow_threshold:
                    counter = 0
                    dic_speed.append('driveNormal')
            # 判断是否转向
            ang_mul_steer = abs(angle * dis)
            if ang_mul_steer > threshold_change and ang_mul_steer < threshold_turn:
                if angle > 0:
                    dic_speed[pos] = 'leftChange'
                elif angle < 0:
                    dic_speed[pos] = 'rightChange'
            if ang_mul_steer > threshold_turn and ang_mul_steer < threshold_uturn:
                if angle > 0:
                    dic_speed[pos] = 'leftTurn'
                elif angle < 0:
                    dic_speed[pos] = 'rightTurn'
            if ang_mul_steer > threshold_uturn:
                if angle > 0:
                    dic_speed[pos] = 'leftUturn'
                elif angle < 0:
                    dic_speed[pos] = 'rightUturn'
        start_time = 0
        end_time = 0

        for idx in range(len(dic_speed)):
            if idx < 1:
                pass
            else:
                # 合并转向掉头情景
                if dic_speed[idx-1] in status and dic_speed[idx] in maneuvre:
                    start_time = idx
                elif dic_speed[idx-1] in maneuvre and dic_speed[idx] in status:
                    end_time = idx
                if idx == (len(dic_speed) - 1):
                    end_time = idx
                if start_time and end_time:
                    # for item_uturn in uturning:
                    #     if item_uturn in dic_speed[start_time: end_time] and max(abs(distance_dic[start_time: end_time])) < threshold_long_straight_bend:
                    #         dic_speed[start_time: end_time] = [item_uturn] * (end_time - start_time)
                    #     elif item_uturn in dic_speed[start_time: end_time] and max(abs(distance_dic[start_time: end_time])) > threshold_long_straight_bend:
                    #         dic_speed[start_time: end_time] = [dic_speed[start_time - 1]] * (end_time - start_time)
                    for item_turning in turning:
                        # threshold_long_straight_bend用于排除长弯道直行，被错误识别为转弯的情形
                        if item_turning in dic_speed[start_time: end_time] and max(abs(distance_dic[start_time: end_time])) < threshold_long_straight_bend:
                            dic_speed[start_time: end_time] = [item_turning] * (end_time - start_time)
                            # for item_uturn in uturning:
                            #     if item_uturn in dic_speed[start_time: end_time]:
                            #         dic_speed[start_time: end_time] = [item_uturn] * (end_time - start_time)
                            #     elif item_uturn in dic_speed[start_time: end_time]:
                            #         dic_speed[start_time: end_time] = [dic_speed[start_time - 1]] * (end_time - start_time)
                        elif item_turning in dic_speed[start_time: end_time] and max(abs(distance_dic[start_time: end_time])) > threshold_long_straight_bend:
                            dic_speed[start_time: end_time] = [dic_speed[start_time - 1]] * (end_time - start_time)
                    start_time, end_time = 0, 0

        for idx in range(len(dic_speed)):
            if idx < 1:
                pass
            else:
                # 排除少量误标
                if dic_speed[idx-1] in status and dic_speed[idx] in maneuvre:
                    start_time = idx
                elif dic_speed[idx-1] in maneuvre and dic_speed[idx] in status:
                    end_time = idx
                if end_time - start_time < 5:
                    dic_speed[start_time: end_time] = [dic_speed[start_time - 1]] * (end_time - start_time)

        # for i in range(len(dic_speed)):
        #     if dic_speed[i] in ['leftUturn', 'rightUturn']:
        #         dic_speed[i] = 'Uturn'
        #     if dic_speed[i] in ['driveSlow', 'driveNormal']:
        #         dic_speed[i] = 'straight'

        self_car_state_dict = [dic_speed]
        self_car_ang_mul_steer_dict = [distance_dic*high_passed_steer]

        return self_car_state_dict, self_car_ang_mul_steer_dict, distance_dic, high_passed_steer

    # 将字典中的行为状态变成数字，方便画图，直观分析
    def map_dic_number(self):
        char_dic = {x: 0 for x in status_update}
        char_dic.update({y: 1 for y in changing})
        char_dic.update({z: 2 for z in turning}) 
        char_dic.update({n: 3 for n in uturning_update})
        for row in self.self_car_state_dict:
            for i in row:
                for idx in range(len(i)):
                    i[idx] = char_dic[i[idx]]
        print(char_dic)

    #记录每种行为的ang_mul_steer_dic最大值（abs），用来参考调整阈值（需要调整）
    def evaluation(self):
        state_dic = list(itertools.chain(*test.self_car_state_dict))
        ang_mul_steer_dic = list(itertools.chain(*test.self_car_ang_mul_steer_dict))
        assert len(state_dic) == len(ang_mul_steer_dic)
        # value_dict = {[],
        #     {"max" : [],
        #     "min" : []
        #     }
        #     }
        for row in range(len(state_dic)):
            unique_values = np.unique(state_dic[row], return_inverse=True) 
            # 使用字典存储每个目标值的最大和最小值
            value_dict = {}
            for value in unique_values[0]:
                value_positions = [i for i in range(len(state_dic[row])) if state_dic[row][i] == value]
                if len(value_positions) > 0:
                    value_dict[value] = {"max":max([ang_mul_steer_dic[row][i] for i in value_positions]), "min": min([ang_mul_steer_dic[row][i] for i in value_positions])}
                    # value_dict[value] = (max([ang_mul_steer_dic[row][i] for i in value_positions]))
                    # value_dict[value]['max'].append(min([ang_mul_steer_dic[row][i] for i in value_positions]))
        
        return value_dict

    def result_save(self, filepath):
        filename = os.path.splitext(os.path.basename(filepath))[0]
        result = [np.array(self.self_car_state_dict_single).flatten().tolist(), self.timestamp_dic]
        save_path = os.path.join(file_path, filename + '.json')
        with open(save_path, 'w') as f:
            json.dump(result, f)

def zip_result():
    zip_file_name = '%s.zip'%state_mode
    json_files = [f for f in os.listdir(file_path) if f.endswith('.json')]

    # Create a new ZIP archive file
    with zipfile.ZipFile(zip_file_name, 'w') as my_zip_file:
        # Add each JSON file to the ZIP archive
        for json_file in json_files:
            json_file_path = os.path.join(file_path, json_file)
            my_zip_file.write(json_file_path)


def plot_result(dict):
    # Create a figure with two subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(8, 4))

    # Plot the first subplot
    for i in range(len(dict.self_car_ang_mul_steer_dict)):
        ax1.plot(dict.self_car_ang_mul_steer_dict[i][0], label=i)
    ax1.set_title('Steer_mul_distance')

    # Plot the second subplot
    for y in range(len(dict.self_car_state_dict)):
        ax2.plot(dict.self_car_state_dict[y][0])
    ax2.set_title('Label')

    # Plot the third subplot
    for z in range(len(dict.distance_dic)):
        ax3.plot(dict.distance_dic[z])
    ax3.set_title('distance_dic')

    for k in range(len(dict.steer_angle)):
        ax4.plot(dict.steer_angle[k])
    ax4.set_title('steer_angle')
    # plt.legend()
    plt.show()

state_mode = 'change'
# file_path = os.path.join('./download/data_%s'%state_mode ,'/') 
file_path = './download/data_%s/'%state_mode       
# file_path = './download/data_turn/'
bag_path = os.path.join(file_path, '*.bag')
# bag_path = './download/data_turn/*.bag'
save_path = '/'

if __name__ == "__main__":
    turning = ['leftTurn', 'rightTurn']
    changing = ['leftChange', 'rightChange']
    uturning = ['leftUturn', 'rightUturn']
    status = ['driveStart', 'driveSlow', 'driveNormal', 'driveStop']
    status_update = ['straight'] + status
    uturning_update = ['Uturn'] + uturning
    test = SelfCarStateEstimation(bag_path, save_path, save_path)
    test.run()
    # test.map_dic_number()
    # evaluation_dic = test.evaluation()
    # print(evaluation_dic)
    zip_result()
    plot_result(test)
