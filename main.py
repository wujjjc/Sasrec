import argparse
import os
from functools import partial

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
    parser.add_argument('--mode', choices=['eval', 'train'], default='eval', help='Run mode')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device selection')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser


def resolve_device(device_name):
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_name == 'cuda' and not torch.cuda.is_available():
        print('CUDA 不可用，已自动切换到 CPU')
        return torch.device('cpu')
    return torch.device(device_name)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_message(message, log_file):
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def evaluate(model, trn_click_dict, item_index_2_rawid, max_len, device, topk=10):
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
            pad_len = max_len - len(item_seq)
            padded_seq = np.concatenate([np.zeros(pad_len, dtype=np.int64), item_seq])
            mask = padded_seq != 0
            item_seq_tensor = torch.tensor(padded_seq, dtype=torch.long, device=device).unsqueeze(0)
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            output = model(item_seq_tensor, mask_tensor).squeeze(0).detach().cpu().numpy()
            target_id = item_index_2_rawid[d['pos_item']]
            recommended = recommend(
                item_emb_np,
                output,
                d['user_id'],
                d['item_seq'],
                item_index_2_rawid,
                d['neg_items'],
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


def train(model, dataloader, optimizer, criterion, trn_click_dict, item_index_2_rawid, max_len, device, args):
    print('开始训练')
    for i in range(args.epoch):
        model.train()
        total_loss = 0.0
        for d in tqdm(dataloader, desc=f'Epoch {i + 1}/{args.epoch}'):
            item_seq = d['item_seq'].to(device)
            pos_item = d['pos_item'].to(device)
            neg_items = d['neg_items'].to(device)
            mask = d['mask'].to(device)
            labels = d['labels'].to(device)

            optimizer.zero_grad()
            logits = model(item_seq, mask, neg_items, pos_item.unsqueeze(1))
            loss = criterion(logits, labels).mean()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / max(len(dataloader), 1)
        log_message(f'Epoch {i + 1}/{args.epoch}, Loss: {avg_loss:.4f}', args.log_file)

        if (i + 1) % args.eval_every == 0:
            torch.save(model.state_dict(), args.model_path)
            recall, ndcg = evaluate(model, trn_click_dict, item_index_2_rawid, max_len, device, topk=args.topk)
            log_message(f'Epoch:{i + 1}: [Eval] Recall@{args.topk}: {recall:.4f}  NDCG@{args.topk}: {ndcg:.4f}', args.log_file)

    torch.save(model.state_dict(), args.model_path)


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    user_info, movie_info, rating_info = read_data()
    user_item_time_dict = get_user_item_time_dict(rating_info)
    user_item_dict = get_user_item_dict(user_item_time_dict)
    samples, item_num, item_index_2_rawid, trn_click_dict = gen_input(
        user_item_dict,
        movie_info,
        max_len=args.max_len,
        neg_sample_num=args.neg_sample_num,
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
    ).to(device)

    if os.path.exists(args.model_path):
        print('加载模型')
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    if args.mode == 'train':
        train(model, dataloader, optimizer, criterion, trn_click_dict, item_index_2_rawid, args.max_len, device, args)
    else:
        torch.save(model.state_dict(), args.model_path)
        recall, ndcg = evaluate(model, trn_click_dict, item_index_2_rawid, args.max_len, device, topk=args.topk)
        log_message(f'Final Recall@{args.topk}: {recall:.4f}  NDCG@{args.topk}: {ndcg:.4f}', args.log_file)


if __name__ == '__main__':
    main()
