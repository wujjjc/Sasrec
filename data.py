# 处理数据
"""
1. ratings.dat（评分文件）
编码格式：每行一条评分记录，字段之间用 :: 分隔。

字段位置	字段名	含义
1	UserID	用户编号，范围 1 ~ 6040
2	MovieID	电影编号，范围 1 ~ 3952
3	Rating	评分值，5 星制（只取整星，如 1,2,3,4,5）
4	Timestamp	Unix 时间戳（自 epoch 起的秒数），表示评分发生的时间


2. users.dat（用户信息文件）
编码格式：每行一个用户的人口统计学信息，字段之间用 :: 分隔。

字段位置	字段名	含义
1	UserID	用户编号，与 ratings.dat 中的 UserID 对应
2	Gender	性别：M（男）或 F（女）
3	Age	年龄编码（见下方年龄范围映射表）
4	Occupation	职业编码（见下方职业映射表）
5	Zip-code	用户提供的邮政编码（字符串，未校验准确性）

3. movies.dat（电影信息文件）
编码格式：每行一部电影的信息，字段之间用 :: 分隔。

字段位置	字段名	含义
1	MovieID	电影编号，与 ratings.dat 中的 MovieID 对应
2	Title	电影标题（包含上映年份，如 “Toy Story (1995)”）
3	Genres	电影类型，多个类型用竖线 | 分隔（类型列表见下方）


"""
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

np.random.seed(42)
def read_data():
    """
    读取数据文件，返回用户信息、电影信息和评分信息

    :return:
        user_info: 用户信息字典，结构为：
            {
                user_id(str): {
                    'gender': str,      # 性别，'M' 或 'F'
                    'age': str,         # 年龄分段编码，原样字符串
                    'occupation': str,  # 职业编码，原样字符串
                    'zip-code': str,    # 邮政编码
                },
                ...
            }

        movie_info: 电影信息字典，结构为：
            {
                movie_id(int): {
                    'year': int,          # 上映年份
                    'title': str,         # 电影标题（去掉年份后的部分）
                    'genres': List[str],  # 电影类型列表，如 ['Action', 'Adventure']
                },
                ...
            }

        rating_info: 评分信息字典，结构为：
            {
                user_id(int): [
                    {
                        'movie_id': int,   # 电影ID
                        'rating': int,     # 评分，1~5 的整数
                        'timestamp': int,  # Unix 时间戳
                    },
                    ...
                ],
                ...
            }
    """
    user_info, movie_info, rating_info = {}, {}, {}
    print("读取文件中...")
    with open('./ml-1m/users.dat', 'r', encoding='latin-1') as f:
        for line in f:
            user_id, gender, age, occupation, zip_code = line.strip().split('::')
            user_info.setdefault(user_id, {})
            user_info[user_id].setdefault('gender', gender)
            user_info[user_id].setdefault('age', age)
            user_info[user_id].setdefault('occupation', occupation)
            user_info[user_id].setdefault('zip-code', zip_code)
    with open('./ml-1m/movies.dat', 'r', encoding='latin-1') as f:
        for line in f:
            movie_id, title, genres = line.strip().split('::')
            movie_id = int(movie_id)
            movie_info.setdefault(movie_id, {})
            year = title[-5:-1]
            title = title[:-7]
            year = int(year)
            movie_info[movie_id].setdefault('year', year)
            movie_info[movie_id].setdefault('title', title)
            movie_info[movie_id].setdefault('genres', genres.split('|'))
    with open('./ml-1m/ratings.dat', 'r', encoding='latin-1') as f:
        for line in f:
            user_id, movie_id, rating, timestamp = line.strip().split('::')
            user_id = int(user_id)
            rating_info.setdefault(user_id, [])
            rating_info[user_id].append({'movie_id': int(movie_id), 'rating': int(rating), 'timestamp': int(timestamp)})
    print("读取完成")
    return user_info, movie_info, rating_info



def get_user_item_time_dict(rating_info):
    """
    从评分信息中构建用户-物品-时间的字典
    :param rating_info: 用户评分信息，格式为 {user_id: [{'movie_id': int, 'rating': int, 'timestamp': int}, ...], ...}
    :return: 用户-物品-时间的字典，格式为 {user_id: (movie_id: timestamp), ...}(未排序)
    """
    user_item_time_dict = {}
    for user_id, values in rating_info.items():
        user_item_time_dict.setdefault(user_id, [])
        for value in values:
            movie_id = value['movie_id']
            timestamp = value['timestamp']
            user_item_time_dict[user_id].append((movie_id, timestamp))
    # for user_id, values in user_item_time_dict.items():
    #     import  random
    #     if random.random() < 0.001:
    #         print(values)
    return user_item_time_dict

def get_user_item_dict(user_item_time_dict):
    """
    :param user_item_time_dict: 用户-物品-时间的字典，格式为 {user_id: {movie_id: timestamp}, ...}(未排序)
    :return:  user_item_dict:用户-物品字典，格式为{user_id: [movie_id1, movie_id2, ...], ...}(已经按时间顺序排列好了)
    """
    user_item_dict = {}  # 获取user的观看电影序列
    for user_id, values in user_item_time_dict.items():
        # values is a list of (movie_id, timestamp) tuples
        values = sorted(values, key=lambda x: x[1], reverse=False)
        user_item_dict.setdefault(user_id, [])
        for movie_id, timestamp in values:
            user_item_dict[user_id].append(movie_id)
    return user_item_dict
            
            
            
def gen_input(user_item_dict, movie_info, max_len, neg_sample_num=3):
    """
    获取模型输入：对用户序列做截断/填充、将物品ID编码为连续索引（0保留为padding），并构造负样本。

    :param user_item_dict: 用户-物品序列字典，格式为 {user_id: [movie_id1, movie_id2, ...], ...}（按时间升序）
    :param movie_info: 电影信息字典，格式为 {movie_id: {'title': str, 'year': int, 'genres': List[str]}, ...}
    :param max_len: 序列最大长度
    :param neg_sample_num: 每个正样本对应的负采样数量
    :return:
        sample: List[dict]，每个样本包含：
            'item_seq': np.ndarray (seq_len,)         # 历史序列（编码后ID，0为padding）
            'seq_len': int                            # 历史序列真实长度（非0个数）
            'pos_item': int                           # 正样本物品ID（编码后）
            'neg_items': np.ndarray (neg_sample_num,) # 负样本物品ID（编码后）
        item_num: int                                 # 物品总数（含padding索引0）
        item_index_2_rawid: Dict[int, int]            # 编码ID -> 原始movie_id 的映射
        trn_click: list[dict]                                     #验证集,
            'user_id' :int                           # 用户ID
            'item_seq': np.ndarray (seq_len,)         # 历史序列（编码后ID，0为padding）
            'seq_len': int            # 
            'pos_item': int           # 答案
    """
    print("开始处理数据")
    all_movie_ids = set(movie_info.keys()) #所有电影的集合
    processed_seq_dict = {}
    #对物品进行编码
    item_profile = [movie_id for movie_id in movie_info.keys()]
    lbe = LabelEncoder()
    item_profile_ = lbe.fit_transform(item_profile)
    item_profile_ = item_profile_ + 1
    item_num = item_profile_.max() + 1
    item_index_2_rawid = dict(zip(item_profile_, item_profile)) #从现在到原来
    item_index_2_nowid = dict(zip(item_profile, item_profile_)) #从原来到现在
    for user_id, movies in user_item_dict.items(): #处理用户的观看序列，若序列大于maxlen，从最后截断
        movie_ids = []
        if len(movies) > max_len: #大于截断
            movie_ids = movies[-max_len:]
        else:
            movie_ids = list(movies)
        for i,movie_id in enumerate(movie_ids):
            if movie_id !=0:
                movie_ids[i] = item_index_2_nowid[movie_id]
        processed_seq_dict[user_id] = np.array(movie_ids, dtype=np.int64)
    sample = []
    trn_click = []

    for user_id, seq in tqdm(processed_seq_dict.items()):
        len_seq = 0
        watch_movie_ids = set(user_item_dict[user_id]) #用户看过的电影集合
        neg_movie_ids = list(all_movie_ids - watch_movie_ids) #用户没看过的电影
        for i in range(len(seq)):
            if seq[i] == 0: continue #填充0跳过
            if i + 1 == len(seq): continue #倒数第一个元素跳过，没有正样本
            len_seq += 1
            item_seq = np.array(seq[:i+1], dtype=np.int64) #历史序列
            pos_item = seq[i+1] #正样本
            replace = len(neg_movie_ids) < neg_sample_num
            if i + 2 == len(seq): #倒数第二个元素，测试集
                neg_items = np.random.choice(neg_movie_ids, size=100, replace=len(neg_movie_ids) < 100)
                trn_click.append({'user_id': user_id, 'item_seq': item_seq, 'seq_len': len_seq, 'pos_item': pos_item, 'neg_items':neg_items })
                continue
            # 负样本采样
            if len(neg_movie_ids) == 0:
                continue
            neg_items = np.random.choice(neg_movie_ids, size=neg_sample_num, replace=replace)
            for j, neg_item in enumerate(neg_items):
                neg_items[j] = item_index_2_nowid[neg_item]
            neg_items = np.array(neg_items, dtype=np.int64)
            assert len_seq == np.sum(item_seq != 0)
            sample.append({'item_seq': item_seq, 'seq_len': len_seq, 'pos_item': pos_item, 'neg_items': neg_items})
    print("输入处理完成")
    return sample, item_num, item_index_2_rawid, trn_click

def recommend(item_embbeding, output, user_id, train_item_seq, item_index_2_rawid, neg_items, target_id, topk=10):
    """
    :param item_embbeding: 物品嵌入矩阵，shape (item_num, embedding_dim)
    :param output: 用户表示向量，shape (embedding_dim,)
    :param user_id: 用户ID
    :param train_item_seq: 输入历史序列（已按时间排序的历史，编码后ID）
    :param item_index_2_rawid: 编码ID -> 原始movie_id 的映射
    :param topk: 推荐结果的数量
    :return:
        List[Tuple[int, float]]: 推荐结果列表，每个元素是一个元组，包含推荐的原始movie_id和对应的相似度分数，按分数从高到低排序。
    """
    # 计算用户表示与所有物品嵌入的相似度（点积）
    scores = np.dot(item_embbeding, output) # (item_num,)
    # 获取用户看过序列的电影集合
    watch_movie_ids = {item_index_2_rawid.get(item) for item in train_item_seq if item != 0}
    test_rawitems = {item_index_2_rawid.get(neg_item) for neg_item in neg_items}
    test_rawitems.add(target_id)
    # 获取未看过的电影的编码ID和对应的分数
    candidate_items = []
    for movie_id, score in enumerate(scores):
        raw_id = item_index_2_rawid.get(movie_id)
        if raw_id is None or raw_id in watch_movie_ids or raw_id not in test_rawitems:
            continue
        candidate_items.append((raw_id, score))
    # 按分数从高到低排序，并取前topk个
    candidate_items.sort(key=lambda x: x[1], reverse=True)
    recommended_movies = candidate_items[:min(topk, len(candidate_items))]
    # 将编码ID转换为原始movie_id，并返回结果
    return recommended_movies





        






















