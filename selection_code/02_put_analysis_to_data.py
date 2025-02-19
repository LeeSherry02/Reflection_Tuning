import json
import numpy as np
import argparse
from tqdm import tqdm
import os

"""
put_analysis_to_data.py 脚本的主要功能是将分析结果（ppl 和 loss 相关的指标）合并到原始的 JSON 数据中，并保存为一个新的 JSON 文件。通过这种方式，可以将分析结果与原始数据结合，便于后续的数据分析和处理

python selection_code/02_put_analysis_to_data.py \
    --pt_data_path data/huatuo_26m_lite/analysis/huatuo_26m_lite_score_5_alpaca.jsonl \
    --json_data_path data/huatuo_26m_lite/huatuo_26m_lite_score_5_alpaca.json \
    --json_save_path data/huatuo_26m_lite/result/huatuo_26m_lite_score_5_alpaca.json 
    
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_data_path", type=str, default='')
    parser.add_argument("--json_data_path", type=str, default='')
    parser.add_argument("--json_save_path", type=str, default='')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    print(args)
    
    # 设置工作目录为当前脚本文件的父目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    os.chdir(parent_dir)
    print(f"Working directory set to: {parent_dir}")

    # 1、读取分析结果数据
    pt_data = []
    if args.pt_data_path[-6:] == '.jsonl':
        with open(args.pt_data_path, 'r') as file:
            for line in file:
                pt_data.append(json.loads(line.strip()))
    else:
        with open(args.pt_data_path, "r") as file:
            pt_data = json.load(file)

    with open(args.json_data_path, "r") as f:
        json_data = json.load(f)

    assert len(json_data) == len(pt_data) # 确保分析结果数据和原始数据的长度一致，以避免索引错误

    # 2、合并分析结果到原始数据
    new_data = []
    for i in tqdm(range(len(pt_data))):

        json_data_i = json_data[i]

        pt_data_i = pt_data[i]
        if pt_data_i == {}:
            ppl_Q_direct, ppl_A_direct, ppl_Q_condition, ppl_A_condition = np.nan, np.nan, np.nan, np.nan
            loss_Q_direct, loss_A_direct, loss_Q_condition, loss_A_condition = np.nan, np.nan, np.nan, np.nan
        else:
            ppl_Q_direct, ppl_A_direct, ppl_Q_condition, ppl_A_condition = \
                pt_data_i['ppl'][0], pt_data_i['ppl'][1], pt_data_i['ppl'][2], pt_data_i['ppl'][3]
            loss_Q_direct, loss_A_direct, loss_Q_condition, loss_A_condition = \
                pt_data_i['loss'][0], pt_data_i['loss'][1], pt_data_i['loss'][2], pt_data_i['loss'][3]

        json_data_i['ppl_Q_direct'] = ppl_Q_direct
        json_data_i['ppl_A_direct'] = ppl_A_direct
        json_data_i['ppl_Q_condition'] = ppl_Q_condition
        json_data_i['ppl_A_condition'] = ppl_A_condition
        try:
            json_data_i['ifd_ppl'] = ppl_A_condition/ppl_A_direct
            json_data_i['rifd_ppl'] = ppl_Q_condition/ppl_Q_direct
        except ZeroDivisionError: # 处理除以零的情况，将结果设置为0
            json_data_i['ifd_ppl'] = 0
            json_data_i['rifd_ppl'] = 0

        json_data_i['loss_Q_direct'] = loss_Q_direct
        json_data_i['loss_A_direct'] = loss_A_direct
        json_data_i['loss_Q_condition'] = loss_Q_condition
        json_data_i['loss_A_condition'] = loss_A_condition
        try:
            json_data_i['ifd_loss'] = loss_A_condition/loss_A_direct
            json_data_i['rifd_loss'] = loss_Q_condition/loss_Q_direct
        except ZeroDivisionError:
            json_data_i['ifd_loss'] = 0
            json_data_i['rifd_loss'] = 0

        new_data.append(json_data_i)

    print('New data len \n',len(new_data))
    with open(args.json_save_path, "w") as fw:
        json.dump(new_data, fw, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()