# !/usr/bin/env python3
"""检测阈值设定"""
# 检测阈值设定，以下参数均无效
# 传感器bias
steer_bias = 0
linear_acc_x_bias = 0.14
# 弯道直行阈值设定
Straight_left_right = 0.01
steer_straight = 15
# 转弯阈值设定
Turn_left_right = 0.5
steer_left_right = 80
# 变道角速度阈值
self_turn_v = 0.5
#掉头阈值
uturn_left_right = 0.9
steer_u_turn = 400
# 当前车速的标量
# velocity = 1
# 慢速行驶阈值
slow_move = 0.5
# 检测帧数，目前暂定5s为一次判断（50帧有效帧，慢速行驶不算在内）
time_step = 50
# 慢速阈值
slow_threshold = 5
# 偏移距离阈值：排除速度过小，但是方向盘转向很大的情况
dis_threshold = 26

# Gausian filter设定，无效
sigma_1 = 2
sigma_2 = 25

# fps，无效
fps_vcan = 100

# 基于fps的自定义参数
start_duration = 2 # 车辆起步状态显示最长秒数
change_lane_threshold = 1 #秒，变道时，方向盘在不同方向切换的中间时间。因为在检测中这段时间会被标记为正常行驶，所以当这段时间小于阈值，才会被判定为变道

# 新版本阈值
# threshold_change = 800
# threshold_turn = 4000
# threshold_uturn = 9000

accross_distance_threshold = 7
threshold_change = 800
threshold_turn = 4000
threshold_uturn = 15000

threshold_long_straight_bend = 230