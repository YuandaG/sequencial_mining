from parameter_adjust import *




bag_path = './dataset/*.bag'
save_path = '/'

if __name__ == "__main__":
    turning = ['leftTurn', 'rightTurn']
    changing = ['leftChange', 'rightChange']
    uturning = ['leftUturn', 'rightUturn']
    status = ['driveStart', 'driveSlow', 'driveNormal', 'driveStop']
    status_update = ['straight']
    uturning_update = ['Uturn']
    test = SelfCarStateEstimation(bag_path, save_path, save_path)
    test.run()
    test.map_dic_number()
    evaluation_dic = test.evaluation()
    print(evaluation_dic)
    # test.plot_result()