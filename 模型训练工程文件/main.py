import subprocess

# Git Bash中bash.exe的实际路径，你需要替换成你自己的路径
git_bash_path = r"C:\Program Files\Git\bin\bash.exe" 

# 要在Git Bash中运行的命令列表
commands = [
    "cd /c/Users/86131/Desktop",  # 切换到目标目录，使用Git Bash的路径格式
    "bash trans.sh", 
]

# 使用subprocess启动bash.exe并运行命令
with subprocess.Popen(git_bash_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                      universal_newlines=True, shell=False) as bash:
    # 将命令转换为bash可以理解的格式，并发送给它
    bash_commands = "; ".join(commands) + "; exit\n"  # 使用分号来分隔命令，并在最后加上exit
    bash.stdin.write(bash_commands)  # 发送命令
    bash.stdin.flush()  # 确保所有命令都已发送

    # 读取bash的输出和错误
    stdout, stderr = bash.communicate()

    # 打印输出和错误
    print("Git Bash Output:")
    print(stdout)
    print("Git Bash Error:")
    print(stderr)