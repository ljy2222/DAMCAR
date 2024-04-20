import os
import torch


class Config():
    def __init__(self, dataset_name):
        self.SEED = 42
        self.EMBED_DIMENSIONS = 32
        self.dataset_name = dataset_name
        if self.dataset_name == 'WeChat':
            self.PCA_DIMENSIONS = 64
            self.ROOT_PATH = '/your/data/path/wechat/'
            self.DATA_PATH = os.path.join(self.ROOT_PATH, 'data')
            self.USER_ACTION = os.path.join(self.DATA_PATH, 'user_action.csv')
            self.FEED_INFO = os.path.join(self.DATA_PATH, 'feed_info.csv')
            self.FEED_EMBEDDINGS = os.path.join(self.DATA_PATH, 'feed_embeddings.csv')
            self.FEAT_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
            self.user_sparse_features = ['userid2']
            self.item_sparse_features = ['feedid2', 'authorid', 'bgm_song_id', 'bgm_singer_id']
            self.item_dense_features = (['videoplayseconds'] +
                                        [f'{i}' for i in range(self.PCA_DIMENSIONS)])
            self.target = ['label']
            self.sparse_features = self.user_sparse_features + self.item_sparse_features
            self.dense_features = self.item_dense_features
            self.FEAT_LIST = self.sparse_features + self.dense_features + self.target
        elif self.dataset_name == 'TikTok':
            self.PCA_DIMENSIONS = 32
            self.ROOT_PATH = '/your/data/path/tiktok/'
            self.DATA_PATH = os.path.join(self.ROOT_PATH, 'data')
            self.USER_ACTION = os.path.join(self.DATA_PATH, 'final_train.txt')
            self.FEED_EMBEDDINGS_1 = os.path.join(self.DATA_PATH, 'text_features.txt')
            self.FEED_EMBEDDINGS_2 = os.path.join(self.DATA_PATH, 'video_features.txt')
            self.FEED_EMBEDDINGS_3 = os.path.join(self.DATA_PATH, 'audio_features.txt')
            self.FEAT_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'feed_city']
            self.user_sparse_features = ['userid2']
            self.item_sparse_features = ['feedid2', 'authorid', 'bgm_song_id', 'feed_city']
            self.item_dense_features = (['videoplayseconds', 'title_len'] +
                                        [f'vd{i}' for i in range(self.PCA_DIMENSIONS)] +
                                        [f'ad{i}' for i in range(self.PCA_DIMENSIONS)])
            self.target = ['label']
            self.sparse_features = self.user_sparse_features + self.item_sparse_features
            self.dense_features = self.item_dense_features
            self.FEAT_LIST = self.sparse_features + self.dense_features + self.target

        self.device = 'cpu'
        self.get_device()

    def get_device(self):
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            self.device = 'cuda:0'
            print(f'{self.device} ready ...')