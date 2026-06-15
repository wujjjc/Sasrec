import argparse
import os
import sys
from functools import partial

# ---- 在 import torch 之前选好 GPU 并设置 CUDA_VISIBLE_DEVICES ----
# torch.cuda.is_available() 会初始化 CUDA context，必须在此之前限制可见设备
# def _pre_select_gpu():
#     device_arg = 'auto'
#     for i, arg in enumerate(sys.argv):
#         if arg == '--device' and i + 1 < len(sys.argv):
#             device_arg = sys.argv[i + 1]
#             break
#         elif arg.startswith('--device='):
#             device_arg = arg.split('=', 1)[1]
#             break
#     if device_arg == 'cpu':
#         return
#     if device_arg == 'auto':
#         import subprocess
#         result = subprocess.run(
#             ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
#             capture_output=True, text=True,
#         )
#         if result.returncode != 0 or not result.stdout.strip():
#             print('nvidia-smi 查询失败，默认使用 GPU 0')
#             os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#             return
#         best_gpu, min_usage = 0, 1.0
#         print('GPU 显存状态：')
#         for line in result.stdout.strip().split('\n'):
#             fields = [x.strip() for x in line.split(',')]
#             idx, used, total = int(fields[0]), float(fields[1]), float(fields[2])
#             usage = used / total
#             print(f'  GPU {idx}: {used:.0f}MB / {total:.0f}MB ({usage:.1%})')
#             if usage < min_usage:
#                 min_usage = usage
#                 best_gpu = idx
#         os.environ['CUDA_VISIBLE_DEVICES'] = str(best_gpu)
#         print(f'自动选择 GPU {best_gpu}（显存占用率 {min_usage:.1%}）')
#     elif device_arg.startswith('cuda'):
#         gpu_id = device_arg.split(':')[-1] if ':' in device_arg else '0'
#         os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

# _pre_select_gpu()
# ---- GPU 选择完毕，现在可以安全 import torch ----

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import *
from Net import *
from SasrecData import *


def build_parser():
    parser = argparse.ArgumentParser(description='SASRec on MovieLens 1M')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--eval-every', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--topk', type=int, default=10, help='Top-K for evaluation')
    parser.add_argument('--max-len', type=int, default=200, help='Maximum sequence length')
    parser.add_argument('--neg-sample-num', type=int, default=10, help='Number of negative samples in training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--embedding-size', type=int, default=32, help='Item embedding size')
    parser.add_argument('--model-path', type=str, default='sasrec_model.pth', help='Path to save/load model')
    parser.add_argument('--log-file', type=str, default='sasrec_eval.log', help='Path to evaluation log file')
    parser.add_argument('--loss-log', type=str, default='sasrec_loss.log', help='Path to loss log file')
    parser.add_argument('--mode', choices=['eval', 'train'], default='train', help='Run mode')
    parser.add_argument('--device', type=str, default='auto', help="Device: 'auto' (选占用率最低的GPU), 'cpu', 'cuda', 'cuda:N'")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_message(message, log_file):
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def evaluate(model, trn_click_dict, item_index_2_rawid, max_len, device, all_item, topk=10):
    """在验证集上计算 Recall@topk 和 NDCG@topk"""
    model.eval()
    prev_mode = model.mode
    model.mode = 'user'
    item_emb_np = model.embedding.embedding.weight.detach().cpu().numpy()
    total, hit, ndcg_sum = 0, 0, 0.0
    with torch.no_grad():
        for d in trn_click_dict:
            total += 1
            item_seq = d['item_seq']
            gender = d['gender']
            age = d['age']
            occupation = d['occupation']
            pad_len = max_len - len(item_seq)
            padded_seq = np.concatenate([np.zeros(pad_len, dtype=np.int64), item_seq])
            mask = [1] + [0] * pad_len + [1] * (max_len - pad_len)
            item_seq_tensor = torch.tensor(padded_seq, dtype=torch.long, device=device).unsqueeze(0)
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            age = torch.tensor([age], dtype=torch.long, device=device)
            gender = torch.tensor([gender], dtype=torch.long, device=device)
            occupation = torch.tensor([occupation], dtype=torch.long, device=device)
            output = model(item_seq_tensor, mask_tensor, gender, age, occupation).squeeze(0).detach().cpu().numpy()
            target_id = item_index_2_rawid[d['pos_item']]
            recommended = recommend(
                item_emb_np,
                output,
                d['user_id'],
                d['item_seq'],
                item_index_2_rawid,
                all_item,
                target_id,
                topk=topk,
            )
            for rank, (movie_id, _) in enumerate(recommended):
                if target_id == movie_id:
                    hit += 1
                    ndcg_sum += 1 / np.log2(rank + 2)
                    break

    model.mode = prev_mode
    model.train()
    recall = hit / max(total, 1)
    ndcg = ndcg_sum / max(total, 1)
    return recall, ndcg


def train(model, dataloader, optimizer, criterion, trn_click_dict, item_index_2_rawid, max_len, device, all_item, args):
    print('开始训练')
    best_recall = 0.0
    for i in range(args.epoch):
        model.train()
        total_loss = 0.0
        for d in tqdm(dataloader, desc=f'Epoch {i + 1}/{args.epoch}'):
            item_seq = d['item_seq'].to(device)
            pos_item = d['pos_item'].to(device)
            neg_items = d['neg_items'].to(device)
            mask = d['mask'].to(device)
            labels = d['labels'].to(device)
            gender = d['gender'].to(device)
            age = d['age'].to(device)
            occupation = d['occupation'].to(device)
            

            optimizer.zero_grad()
            # logits = model(item_seq, mask, neg_items, pos_item.unsqueeze(1)) * torch.exp(model.temperature)
            logits = model(item_seq, mask, gender, age, occupation, neg_items, pos_item.unsqueeze(1))
            loss = criterion(logits, labels).mean()
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        avg_loss = total_loss / max(len(dataloader), 1)
        log_message(f'Epoch {i + 1}/{args.epoch}, Loss: {avg_loss:.4f}', args.loss_log)

        if (i + 1) % args.eval_every == 0:
            recall, ndcg = evaluate(model, trn_click_dict, item_index_2_rawid, max_len, device, all_item, topk=args.topk)
            log_message(f'Epoch {i + 1}: Recall@{args.topk}: {recall:.4f}  NDCG@{args.topk}: {ndcg:.4f}', args.log_file)
            if recall > best_recall:
                log_message(f'Epoch {i + 1}: 模型已保存 (Recall@{args.topk} {best_recall:.4f} -> {recall:.4f})', args.log_file)
                best_recall = recall
                torch.save(model.state_dict(), args.model_path)
            else:
                log_message(f'Epoch {i + 1}: 跳过保存 (Recall@{args.topk} 当前 {recall:.4f} <= 最优 {best_recall:.4f})', args.log_file)



def main():
    args = build_parser().parse_args()
    # CUDA_VISIBLE_DEVICES 已在 import torch 前由 _pre_select_gpu() 设置，PyTorch 只看到 1 张卡
    gpu_id = 7
    torch.cuda.set_device(gpu_id)         # 设置默认 GPU
    device = torch.device('cuda')  
    set_seed(args.seed)

    user_info, movie_info, rating_info, max_age, max_occupation = read_data()
    user_item_time_dict = get_user_item_time_dict(rating_info)
    user_item_dict = get_user_item_dict(user_item_time_dict)
    samples, item_num, item_index_2_rawid, trn_click_dict, all_item = gen_input(
        user_item_dict,
        movie_info,
        max_len=args.max_len,
        neg_sample_num=args.neg_sample_num,
        user_info=user_info,
    )
    dataloader = DataLoader(
        SASRecDataset(samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, max_len=args.max_len),
    )

    model = Sasrec_Net(
        embedding_size=args.embedding_size,
        embedding_num=item_num,
        max_len=args.max_len,
        dropout=args.dropout,
        max_age=max_age,
        max_occupation=max_occupation
    ).to(device)

    # if os.path.exists(args.model_path):
    #     print('加载模型')
    #     model.load_state_dict(torch.load(args.model_path, map_location=device))
    #     print('模型加载完成')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    all_item = torch.tensor(all_item, dtype=torch.long, device=device)

    if args.mode == 'train':
        train(model, dataloader, optimizer, criterion, trn_click_dict, item_index_2_rawid, args.max_len, device, all_item, args)
    else:
        print('开始评估')
        recall, ndcg = evaluate(model, trn_click_dict, item_index_2_rawid, args.max_len, device, all_item, topk=args.topk)
        log_message(f'Final Recall@{args.topk}: {recall:.4f}  NDCG@{args.topk}: {ndcg:.4f}', args.log_file)


if __name__ == '__main__':
    main()
