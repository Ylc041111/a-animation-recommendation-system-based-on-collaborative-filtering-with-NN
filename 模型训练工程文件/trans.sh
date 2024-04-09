#!/bin/bash

# 本地文件路径
local_file="/C/Users/86131/Desktop/lovechoose-win32-x64/resources/userData.json"

# Conda 环境名称
conda_env="torch2.0.0_py3.8"

# 服务器相关信息
server_username="ad_1"
server_ip="10.109.118.52"
server_remote_directory="/home/ad_1/ylc/tjjm"

# 传输文件到服务器
scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$local_file" "$server_username"@"$server_ip":"$server_remote_directory/userData.json"

# 在服务器上运行py 脚本并激活 Conda 环境后台运行，并将 "ok" 输出到临时文件中
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$server_username"@"$server_ip" "source /home/ad_1/anaconda3/bin/activate $conda_env && cd $server_remote_directory && python name_prd.py > task_complete.txt &"

# 等待 "ok" 信号的出现
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$server_username"@"$server_ip" "while ! grep -q 'ok' $server_remote_directory/task_complete.txt; do sleep 1; done"

# 从服务器获取文件
local_result_file="/C/Users/86131/Desktop/lovechoose-win32-x64/resources/resultData.json"
scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$server_username"@"$server_ip":"$server_remote_directory/resultData.json" "$local_result_file"

# 清理服务器上的临时文件（如果需要）
# 注意：通常不建议通过 SSH 直接删除文件，因为它可能会因为各种原因失败。更好的做法是让服务器上的脚本或程序自己管理临时文件。
# ssh "$server_username"@"$server_ip" "rm $server_remote_directory/userData.json"

# 清理本地临时文件（如果需要）
# rm "$local_file"



