import numpy as np
from data import *
import torch
from torch.utils.data import DataLoader
from SasrecData import *
from Net import *
import os
from functools import partial


LOG_FILE = 'sasrec_eval.log'


def log_message(message, log_file=LOG_FILE):
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

dropout = 0.2
epoch = 100
lr = 1e-3
eval_every = 5       # 每隔多少个 epoch 评估一次
topk = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
neg_sample_num = 10


user_info, movie_info, rating_info = read_data()
user_item_time_dict = get_user_item_time_dict(rating_info)
user_item_dict = get_user_item_dict(user_item_time_dict)
max_len = 200
samples, item_num, item_index_2_rawid, trn_click_dict = gen_input(user_item_dict, movie_info, max_len=max_len, neg_sample_num=neg_sample_num)
dataloader = DataLoader(SASRecDataset(samples), batch_size=32, shuffle=True, collate_fn=partial(collate_fn, max_len=max_len))
model = Sasrec_Net(embedding_size=32, embedding_num=item_num, max_len=max_len, dropout=dropout).to(device)
if os.path.exists('sasrec_model.pth'):
    print("加载模型")
    model.load_state_dict(torch.load('sasrec_model.pth', map_location=device))
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')


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
            mask = (padded_seq != 0)
            item_seq_tensor = torch.tensor(padded_seq, dtype=torch.long, device=device).unsqueeze(0)
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            output = model(item_seq_tensor, mask_tensor).squeeze(0).detach().cpu().numpy()
            target_id = item_index_2_rawid[d['pos_item']]
            recommended = recommend(item_emb_np, output, d['user_id'], d['item_seq'], item_index_2_rawid, d['neg_items'], target_id, topk=topk)
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

# pos_weight = torch.ones([1, 1 + neg_sample_num], device=device)
# pos_weight[0][0] *= neg_sample_num
# print("开始训练")
# for i in range(epoch):
#     model.train()
#     total_loss = 0
#     for d in tqdm(dataloader, desc=f"Epoch {i+1}/{epoch}"):
#         item_seq = d['item_seq'].to(device)
#         pos_item = d['pos_item'].to(device)
#         neg_items = d['neg_items'].to(device)
#         mask = d['mask'].to(device)
#         labels = d['labels'].to(device)
#         optimizer.zero_grad()
#         y = model(item_seq, mask, neg_items, pos_item.unsqueeze(1))
#         import torch.nn.functional as F
#
#         pos_logits = y[:, :1]
#         neg_logits = y[:, 1:]
#
#         pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
#         neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits)).mean()
#         loss = 0.5 * (pos_loss + neg_loss)
#         total_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     avg_loss = total_loss / max(len(dataloader), 1)
#     log_message(f"Epoch {i+1}/{epoch}, Loss: {avg_loss:.4f}")
#
#     if (i + 1) % eval_every == 0:
#         torch.save(model.state_dict(), 'sasrec_model.pth')
#         recall, ndcg = evaluate(model, trn_click_dict, item_index_2_rawid, max_len, device, topk=topk)
#         log_message(f"Epoch:{i + 1}: [Eval] Recall@{topk}: {recall:.4f}  NDCG@{topk}: {ndcg:.4f}")
#         if recall >= 0.79 or ndcg >= 0.50:
#             break

torch.save(model.state_dict(), 'sasrec_model.pth')
recall, ndcg = evaluate(model, trn_click_dict, item_index_2_rawid, max_len, device, topk=topk)
log_message(f"Final Recall@{topk}: {recall:.4f}  NDCG@{topk}: {ndcg:.4f}")
