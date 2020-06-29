import torch
import os
import random as rd
import numpy as np

from graph_recsys_benchmark.models import PAGAGATRecsysModel
from graph_recsys_benchmark.datasets import MovieLens
from graph_recsys_benchmark.utils import get_folder_path
from graph_recsys_benchmark.utils import get_opt_class, load_model

# Setup data and weights file path
data_folder, weights_folder, logger_folder = \
    get_folder_path(model='Graph', dataset='Movielens1m', loss_type='BPR')

# Setup device
device = 'cpu' if not torch.cuda.is_available() else 'cuda:7'

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
    'num_heads': 1, 'meta_path_steps': [2 for _ in range(10)],
    'aggr': 'concat'
}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))


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


class PAGAGATRecsysModel(PAGAGATRecsysModel):
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
    model = PAGAGATRecsysModel(**model_args).to(device)

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


class PAGARecSys(object):

    def __init__(self, num_recs):
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

        # Build the
        self.base_iids = [i for i in base_iids]
        base_inids = [self.dataset.e2nid_dict['iid'][iid] for iid in base_iids]
        new_interactions = np.vstack([[self.new_user_nid for _ in range(len(self.base_iids))], base_inids])
        self.dataset.edge_index_nps['user2item'] = np.hstack([self.dataset.edge_index_nps['user2item'], new_interactions])

        self.model.update_graph_input(self.dataset)
        self.model.eval()

        self.user_is_built = True
        print('user building done...')

    def rebuild_user(self, base_iids):
        """
        Build user profiles given the historical user interactions
        :param iids: user selected item ids, list
        :param demographic_info: (age, gender, occupation), tuple
        :return:
        """
        if self.user_is_built is False:
            print('The user is not built!')

        # Build the user with new interactions
        self.base_iids = self.base_iids + [i for i in base_iids]
        base_inids = [self.dataset.e2nid_dict['iid'][iid] for iid in base_iids]
        new_interactions = np.vstack([[self.new_user_nid for _ in range(len(self.base_iids))], base_inids])
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
        exps = [self.get_explanation(rec_iid) for rec_iid in rec_iids]

        return rec_item_df, exps

    def get_explanation(self, iid):
        inid = self.dataset.e2nid['iid'][iid]
        row = [self.new_user_nid, self.new_user_nid]
        col = [self.new_user_nid, movie_nid]
        expl_edge_index_np = np.array([row, col])
        edge_index = torch.cat(
            (
                self.data.edge_index,
                self.new_edge_index,
                torch.from_numpy(expl_edge_index_np).long().to(self.device_args['device'])
            ),
            dim=1)
        new_path_np = path.join(edge_index.cpu().numpy(), row)
        new_path = torch.from_numpy(new_path_np).long().to(self.device_args['device'])
        new_node_emb = torch.cat((self.model.node_emb.weight, self.new_user_emb.weight), dim=0)
        att = self.model.forward(new_node_emb, new_path)[1]
        opt_path = new_path[:, torch.argmax(att)].numpy()

        e = self.data.nid2e[0][opt_path[0]]

        if e[0] == 'uid':
            expl = 'Uid0--Iid{}--Uid{}'.format(iid, e[1])
            expl_type = 'UIU'
        elif e[0] == 'iid':
            expl = 'Iid{}--Uid0--Iid{}'.format(
                iid,
                e[1])
            expl_type = 'IUI'
        elif e[0] == 'gender' or e[0] == 'occ':
            expl = 'Iid{}--Uid0--DFType{}--DFValue{}'.format(
                iid,
                e[0],
                e[1]
            )
            expl_type = 'IUDD'
        else:
            expl = 'Uid0--Iid{}--CFType{}--CFValue{}'.format(
                iid,
                e[0],
                e[1]
            )
            expl_type = 'UICC'

        return expl, expl_type


if __name__ == '__main__':
    recsys = PAGARecSys(num_recs=10)
    recsys.get_top_n_popular_items()
    recsys.build_user([1, 2, 3, 4], [1, 'M', 1])
    print(recsys.get_recommendations())