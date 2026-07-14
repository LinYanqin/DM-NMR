import nmrglue as ng
import scipy.io as sio
import os
#

def check_duplicate_paths(file_path):
    """
    检测指定文本文件第二列的重复路径
    参数：
        file_path: 要检测的txt文件路径
    返回：
        包含重复路径及其出现位置的字典
    """
    path_records = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, start=1):
            # 清理数据并分割列
            line = line.strip()
            if not line:
                continue  # 跳过空行

            columns = line.split()
            if len(columns) < 2:
                print(f"警告：第 {line_num} 行格式异常")
                continue

            _, path = columns[0], ' '.join(columns[1:])  # 处理含空格的路径

            # 记录路径出现位置
            if path in path_records:
                path_records[path].append(line_num)
            else:
                path_records[path] = [line_num]

    # 筛选重复项
    duplicates = {k: v for k, v in path_records.items() if len(v) > 1}

    # 输出结果
    if duplicates:
        print("\n发现重复路径：")
        for path, lines in duplicates.items():
            print(f"路径 '{path}' 重复出现于行：{lines}")
    else:
        print("\n未发现重复路径")

    return path_records


# 使用示例
if __name__ == "__main__":
    path_records = check_duplicate_paths('/data4/cwh/DPSdata/BMRB2D/readme.txt')
    for k, v in path_records.items():
        dic, data = ng.pipe.read(os.path.join(k,'test.ft21'))
        assert data.ndim == 2, " data dim is error"
        assert data.shape[1] == 128, " data shape is error"
        sio.savemat(os.path.join(f'/data4/cwh/DPSdata/BMRB2D/BMRB{v[0]}.mat'), {'data': data})

    print("[*] Job finished!")