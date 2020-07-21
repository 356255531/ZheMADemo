import torch
import os
import random as rd
import numpy as np

from graph_recsys_benchmark.models import MPAGATRecsysModel
from graph_recsys_benchmark.datasets import MovieLens
from graph_recsys_benchmark.utils import get_folder_path
from graph_recsys_benchmark.utils import get_opt_class, load_model

# Setup data and weights file path
data_folder, weights_folder, logger_folder = \
    get_folder_path(model='Graph', dataset='Movielens1m', loss_type='BPR')

# Setup device
device = 'cpu' if not torch.cuda.is_available() else 'cuda'

# Setup args
dataset_args = {
    'root': data_folder, 'dataset': 'Movielens', 'name': '1m',
    'seed': 2020,
    'if_use_features': False, 'num_negative_samples': 4,
    'num_core': 10, 'num_feat_core': 10,
    'loss_type': 'BPR'
}
model_args = {
    'model_type': 'Graph',
    'if_use_features': False,
    'emb_dim': 16, 'hidden_size': 32,
    'repr_dim': 16, 'dropout': 0,
    'meta_path_steps': [2 for _ in range(10)], 'num_heads': 1,
    'aggr': 'att'
}

print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))


TEMPLATE = {
    'user': 'User {}, which has simliar perference as you, watched this movie.',
    'item': 'This movie is silimar to the movie {}, you have watched before.',
    'year': 'I guess you may like this kinds of movies around year {}',
    'actor': 'I guess you may like movies acted by {}',
    'director': 'I guess you may like movies directed by {}',
    'writer': 'I guess you may like movies written by {}',
    'gender': 'People who is of the same gender ({}) as you may like this movie',
    'genres': 'You may like this movie due to its genres {}.',
    'age': 'People of your age ({}) may like this movie',
    'occ': 'People have the same occupation {} as you may like this movie'
}


def _negative_sampling(u_nid, num_negative_samples, train_splition, item_nid_occs):
    """
    The negative sampling methods used for generating the training batches
    :param u_nid:
    :return:
    """
    train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map = train_splition

    negative_inids = test_pos_unid_inid_map[u_nid] + neg_unid_inid_map[u_nid]
    negative_inids = rd.choices(population=negative_inids, k=num_negative_samples)

    return negative_inids


class MPAGATRecsysModel(MPAGATRecsysModel):
    def update_graph_input(self, dataset):
        user2item_edge_index = torch.from_numpy(dataset.edge_index_nps['user2item']).long().to(device)
        year2item_edge_index = torch.from_numpy(dataset.edge_index_nps['year2item']).long().to(device)
        actor2item_edge_index = torch.from_numpy(dataset.edge_index_nps['actor2item']).long().to(device)
        director2item_edge_index = torch.from_numpy(dataset.edge_index_nps['director2item']).long().to(device)
        writer2item_edge_index = torch.from_numpy(dataset.edge_index_nps['writer2item']).long().to(device)
        genre2item_edge_index = torch.from_numpy(dataset.edge_index_nps['genre2item']).long().to(device)
        age2user_edge_index = torch.from_numpy(dataset.edge_index_nps['age2user']).long().to(device)
        gender2user_edge_index = torch.from_numpy(dataset.edge_index_nps['gender2user']).long().to(device)
        occ2user_edge_index = torch.from_numpy(dataset.edge_index_nps['occ2user']).long().to(device)
        meta_path_edge_indicis_1 = [user2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
        meta_path_edge_indicis_2 = [torch.flip(user2item_edge_index, dims=[0]), user2item_edge_index]
        meta_path_edge_indicis_3 = [year2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
        meta_path_edge_indicis_4 = [actor2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
        meta_path_edge_indicis_5 = [writer2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
        meta_path_edge_indicis_6 = [director2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
        meta_path_edge_indicis_7 = [genre2item_edge_index, torch.flip(user2item_edge_index, dims=[0])]
        meta_path_edge_indicis_8 = [gender2user_edge_index, user2item_edge_index]
        meta_path_edge_indicis_9 = [age2user_edge_index, user2item_edge_index]
        meta_path_edge_indicis_10 = [occ2user_edge_index, user2item_edge_index]

        meta_path_edge_index_list = [
            meta_path_edge_indicis_1, meta_path_edge_indicis_2, meta_path_edge_indicis_3,
            meta_path_edge_indicis_4, meta_path_edge_indicis_5, meta_path_edge_indicis_6,
            meta_path_edge_indicis_7, meta_path_edge_indicis_8, meta_path_edge_indicis_9,
            meta_path_edge_indicis_10
        ]
        self.meta_path_edge_index_list = meta_path_edge_index_list


def get_utils():
    dataset_args['_negative_sampling'] = _negative_sampling
    dataset = MovieLens(**dataset_args)

    model_args['num_nodes'] = dataset.num_nodes
    model_args['dataset'] = dataset
    model = MPAGATRecsysModel(**model_args).to(device)

    opt_class = get_opt_class('adam')
    optimizer = opt_class(
        params=model.parameters(),
        lr=1e-3,
        weight_decay=0
    )

    # Load models
    weights_path = os.path.join(weights_folder, 'run_{}'.format(str(1)))
    if not os.path.exists(weights_path):
        os.makedirs(weights_path, exist_ok=True)
    weights_file = os.path.join(weights_path, 'latest.pkl')
    model, optimizer, last_epoch, rec_metrics = load_model(weights_file, model, optimizer, device)
    model.eval()  # switch model to eval mode

    return dataset, model


class MPAGATRecsys(object):

    def __init__(self):
        self.user_is_built = False

        self.dataset, self.model = get_utils()

        self.picked_iids = []

    def get_top_n_popular_items(self, n=10):
        """
        Get the top n movies from self.data.ratings.
        Remove the duplicates in self.data.ratings and sort it by movie count.
        After you find the top N popular movies' item id,
        look over the details information of item in self.data.movies
        :param n: the number of items, int
        :return: df: popular item dataframe, df
        """
        ratings_df = self.dataset.ratings[['iid', 'movie_count']].drop_duplicates(subset=['iid']).sort_values(by='movie_count', ascending=False)
        popular_iids = ratings_df.iid[:n].to_numpy()
        popular_item_df = self.dataset.items[self.dataset.items.iid.isin(popular_iids)]
        return popular_item_df

    def build_cold_user(self, demographic_info):
        """
        Build user profiles given the historical user interactions
        :param iids: user selected item ids, list
        :param demographic_info: (age, gender, occupation), tuple
        :return:
        """
        if self.user_is_built is True:
            print('The user model already exists!')
        # Build edges for new user
        self.demographic_info = demographic_info
        self.new_user_nid = self.model.x.shape[0]
        self.model.x = torch.nn.Parameter(torch.cat(
            [self.model.x, torch.randn((1, model_args['emb_dim']), device=device)],
            dim=0
        ))

        age_nid = self.dataset.e2nid_dict['age'][int(demographic_info[0])]
        gender_nid = self.dataset.e2nid_dict['gender'][demographic_info[1]]
        occ_nid = self.dataset.e2nid_dict['occ'][int(demographic_info[2])]

        self.dataset.edge_index_nps['age2user'] = np.hstack([self.dataset.edge_index_nps['age2user'], [[age_nid], [self.new_user_nid]]])
        self.dataset.edge_index_nps['gender2user'] = np.hstack([self.dataset.edge_index_nps['gender2user'], [[gender_nid], [self.new_user_nid]]])
        self.dataset.edge_index_nps['occ2user'] = np.hstack([self.dataset.edge_index_nps['occ2user'], [[occ_nid], [self.new_user_nid]]])

        self.model.update_graph_input(self.dataset)
        self.model.eval()

        self.user_is_built = True
        print('cold user building done...')

    def build_user(self, base_iids, demographic_info):
        """
        Build user profiles given the historical user interactions
        :param iids: user selected item ids, list
        :param demographic_info: (age, gender, occupation), tuple
        :return:
        """
        if self.user_is_built is True:
            print('The user model already exists!')
        # Build edges for new user
        self.demographic_info = demographic_info
        self.new_user_nid = self.model.x.shape[0]
        self.model.x = torch.nn.Parameter(torch.cat(
            [self.model.x, torch.randn((1, model_args['emb_dim']), device=device)],
            dim=0
        ))

        age_nid = self.dataset.e2nid_dict['age'][int(demographic_info[0])]
        gender_nid = self.dataset.e2nid_dict['gender'][demographic_info[1]]
        occ_nid = self.dataset.e2nid_dict['occ'][int(demographic_info[2])]

        self.dataset.edge_index_nps['age2user'] = np.hstack([self.dataset.edge_index_nps['age2user'], [[age_nid], [self.new_user_nid]]])
        self.dataset.edge_index_nps['gender2user'] = np.hstack([self.dataset.edge_index_nps['gender2user'], [[gender_nid], [self.new_user_nid]]])
        self.dataset.edge_index_nps['occ2user'] = np.hstack([self.dataset.edge_index_nps['occ2user'], [[occ_nid], [self.new_user_nid]]])

        # Build the interactions
        self.base_iids = [i for i in base_iids]
        base_inids = [self.dataset.e2nid_dict['iid'][iid] for iid in base_iids]
        new_interactions = np.vstack([[self.new_user_nid for _ in range(len(self.base_iids))], base_inids])
        self.dataset.edge_index_nps['user2item'] = np.hstack([self.dataset.edge_index_nps['user2item'], new_interactions])

        self.model.update_graph_input(self.dataset)
        self.model.eval()

        self.user_is_built = True
        print('user building done...')

    def rebuild_user(self, new_iids):
        """
        Build user profiles given the historical user interactions
        :param iids: user selected item ids, list
        :param demographic_info: (age, gender, occupation), tuple
        :return:
        """
        if self.user_is_built is False:
            print('The user is not yet built!')

        # Build the user with new interactions
        self.base_iids = self.base_iids + [i for i in new_iids]
        new_inids = [self.dataset.e2nid_dict['iid'][iid] for iid in new_iids]
        new_interactions = np.vstack([[self.new_user_nid for _ in range(len(self.base_iids))], new_inids])
        self.dataset.edge_index_nps['user2item'] = np.hstack([self.dataset.edge_index_nps['user2item'], new_interactions])

        self.model.update_graph_input(self.dataset)
        self.model.eval()

        self.user_is_built = True
        print('user rebuilding done...')

    def get_recommendations(self, num_recs):
        if not self.user_is_built:
            raise AssertionError('User not initialized!')
        candidate_iids = np.random.choice(self.get_top_n_popular_items(1000).iid, 100)
        candidate_iids = [i for i in candidate_iids if i not in self.picked_iids and i not in self.base_iids]
        candidate_inids = [self.dataset.e2nid_dict['iid'][iid] for iid in candidate_iids]

        unids_t = torch.from_numpy(np.array([self.new_user_nid for _ in range(len(candidate_inids))])).to(device)
        candidate_inids_t = torch.from_numpy(np.array(candidate_inids)).to(device)
        pred = self.model.predict(unids_t, candidate_inids_t).reshape(-1)
        rec_idx = torch.topk(pred, k=num_recs)[1].cpu().numpy()
        rec_iids = np.array(candidate_iids)[rec_idx]

        rec_item_df = self.dataset.items[self.dataset.items.iid.isin(rec_iids)]
        exps = self.get_explanation(rec_iids)

        return rec_item_df, exps

    def get_explanation(self, rec_iids):
        # def get_head_e(attr, inids):
        #     pass
        # rec_inids = [self.dataset.e2nid['iid'][iid] for iid in rec_iids]
        # enids = [get_head_e(self.model.attrs, inid) for inid in rec_inids]
        # entities = [self.dataset.nid2e[nid] for nid in enids]
        # expls = [TEMPLATE[expl_type].format(entity) for expl_type, entity in entities]

        entities = [('age', 20) for _ in range(len(rec_iids))]
        expls = [TEMPLATE[expl_type].format(entity) for expl_type, entity in entities]
        return expls


if __name__ == '__main__':
    recsys = MPAGATRecsys()
    recsys.get_top_n_popular_items()
    recsys.build_user([1, 2, 3, 4], [1, 'M', 1])
    print(recsys.get_recommendations(10))
