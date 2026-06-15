import torch
from torch.utils.data import Dataset
"""
生成训练所用的迭代器
"""


class SASRecDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, max_len):
    """
    :param batch: List[dict], 每个 dict 包含:
        'item_seq': np.ndarray ()               # 已填充/截断的序列
        'seq_len': int                          # 真实长度
        'pos_item': int                         # 正样本物品ID
        'neg_items': np.ndarray (neg_sample_num,)  # 负样本ID
        'gender': int                           # 用户性别
        'age': int                              # 用户年龄
        'occupation': int                        # 用户职业编码
    :return: dict, 批量张量:
        'item_seq': torch.LongTensor (B, L)     # padding 后的序列
        'pos_item': torch.LongTensor (B,)       # 正样本物品ID
        'neg_items': torch.LongTensor (B, N)    # 负样本物品ID
        'mask': torch.BoolTensor (B, L)         # 序列有效位置 mask
        'seq_len': torch.LongTensor (B,)        # 真实长度
        'labels': torch.LongTensor (B, 1 + n)         # 标签，第一列为正样本，后面 n 列为负样本
        'gender': torch.LongTensor (B,)             # 用户性别
        'age': torch.LongTensor (B,)                # 用户年龄
        'occupation': torch.LongTensor (B,)          # 用户职业编码
    """
    item_seq_list = [torch.tensor(s['item_seq'], dtype=torch.long) for s in batch]
    seq_len_list = [torch.tensor(s['seq_len'], dtype=torch.long) for s in batch]
    pos_item_list = [torch.tensor(s['pos_item'], dtype=torch.long) for s in batch]
    neg_items_list = [torch.tensor(s['neg_items'], dtype=torch.long) for s in batch]
    gender_list = [torch.tensor(s['gender'], dtype=torch.long) for s in batch]
    age_list = [torch.tensor(s['age'], dtype=torch.long) for s in batch]
    occupation_list = [torch.tensor(s['occupation'], dtype=torch.long) for s in batch]
    padding_seqs, masks = [], []
    for seq in item_seq_list: #处理得到padding_seqs和masks,padding_seq是后加0补齐到max_len
        length = len(seq)
        if length >= max_len:
            padding_len = 0
            padding_seq = seq[-max_len:]
        else:
            padding_len = max_len - length
            padding_seq = torch.cat([torch.zeros(padding_len, dtype=torch.long), seq], dim=0)
        mask = [1] + [0] * padding_len + [1] * (max_len - padding_len)
        padding_seqs.append(padding_seq)
        masks.append(torch.tensor(mask, dtype=torch.bool))
    labels = torch.zeros([len(item_seq_list), 1 + len(neg_items_list[0])], dtype=torch.float) #batch, neg + 1
    labels[:, 0] = 1.0
    return {
        'item_seq': torch.stack(padding_seqs, dim=0),
        'pos_item': torch.stack(pos_item_list, dim=0),
        'neg_items': torch.stack(neg_items_list, dim=0),
        'mask': torch.stack(masks, dim=0),
        'seq_len': torch.stack(seq_len_list, dim=0),
        'labels': labels,
        'gender': torch.stack(gender_list, dim=0),
        'age': torch.stack(age_list, dim=0),
        'occupation': torch.stack(occupation_list, dim=0),
    }
