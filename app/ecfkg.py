import torch
import os
import pandas as pd
import numpy as np
import functools
import torch
import tqdm

from graph_recsys_benchmark.models import ECFKGRecsysModel
from graph_recsys_benchmark.datasets import MovieLens
from graph_recsys_benchmark.utils import get_folder_path
from graph_recsys_benchmark.utils import get_opt_class, load_model
from torch_geometric.nn.inits import glorot

K = 3  # BFS max search steps
N_ITER = 4000
DS = 'Movielens1m'

# Setup data and weights file path
data_folder, weights_folder, logger_folder = \
    get_folder_path(model='Graph', dataset='Movielens1m', loss_type='BPR')

# Setup device
device = 'cpu' if not torch.cuda.is_available() else 'cuda'

# Setup args
dataset_args = {
    'root': data_folder, 'dataset': 'Movielens', 'name': '1m',
    'if_use_features': 'False', 'num_negative_samples': 4,
    'num_core': 10, 'num_feat_core': 10,
    'cf_loss_type': 'BPR', 'type': 'hete',
    'sampling_strategy': 'random', 'entity_aware': False
}
model_args = {'model_type': 'Graph', 'emb_dim': 64}


print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))


def get_explanation_text(exp_type, exp_entity):
    if exp_type == ('iid', 'genre', 'iid'):
        expl = 'You may like this movie, since it is similar to the {} movie {} you watched before'.format(exp_entity[2], exp_entity[1])
    elif exp_type == ('iid', 'actor', 'iid'):
        expl = 'You may like this movie, since you may like the actor {} performed in the movie {}'.format(exp_entity[2], exp_entity[1])
    elif exp_type == ('iid', 'director', 'iid'):
        expl = 'You may like this movie, since you may like the director {} who also directed the the movie {}'.format(exp_entity[2], exp_entity[1])
    elif exp_type == ('iid', 'writer', 'iid'):
        expl = 'You may like this movie, since you may like the writer {} who wrote the movie {}'.format(exp_entity[2], exp_entity[1])
    elif exp_type == ('age', 'uid', 'iid'):
        expl = 'You may like this movie, the user who is of the same age like you also likes this movie'
    elif exp_type == ('occ', 'uid', 'iid'):
        expl = 'You may like this movie, the user who has the same occ like you also likes this movie'
    elif exp_type == ('gender', 'uid', 'iid'):
        expl = 'You may like this movie, the user who has the same gender like you also likes this movie'
    elif exp_type == ('iid', 'uid', 'iid'):
        expl = 'You may like this movie, the user who has the movie {} also likes this movie'.format(exp_entity[0])
    else:
        raise NotImplementedError
    return expl


def get_utils():
    dataset = MovieLens(**dataset_args)

    model_args['num_nodes'] = dataset.num_nodes
    model_args['dataset'] = dataset
    model = ECFKGRecsysModel(**model_args).to(device)

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


class MCFKGRecsys(object):
    def __init__(self):
        self.dataset, self.model = get_utils()

        self.user_is_built = False
        self.user_emb = torch.nn.Parameter(torch.Tensor(1, 64).to(device))
        glorot(self.user_emb)

        rating_file = 'checkpoint/data/{}/processed/ratings.csv'.format(DS)
        movie_file = 'checkpoint/data/{}/processed/movies.csv'.format(DS)
        user_file = 'checkpoint/data/{}/processed/users.csv'.format(DS)

        assert os.path.exists(rating_file)
        assert os.path.exists(movie_file)
        assert os.path.exists(user_file)
        self.ratings = pd.read_csv(rating_file, sep=';')
        self.movies = pd.read_csv(movie_file, sep=';').fillna('')
        movie_count = self.ratings['iid'].value_counts()
        movie_count.name = 'movie_count'
        self.sorted_movie_count = movie_count.sort_values(ascending=False)
        self.users = pd.read_csv(user_file, sep=';')

        # build edge_attr
        edge_index_r_nps = [
            (_, np.ones((_.shape[1], 1)) * self.dataset.edge_type_dict[edge_type])
            for edge_type, _ in self.dataset.edge_index_nps.items()
        ]
        r_np = np.vstack([_[1] for _ in edge_index_r_nps])

        r_np = np.vstack([r_np, -r_np])

        self.edge_attr = torch.from_numpy(r_np).long().to(device=device)

        # Create user interactions map
        self.uid_iid_map = {}
        for uid in list(self.ratings.uid.unique()):
            self.uid_iid_map[uid] = list(self.ratings[self.ratings.uid == uid].iid)

        # Create edge_index
        edge_index_np = np.hstack(list(self.dataset.edge_index_nps.values()))
        self.edge_index_np = np.hstack([edge_index_np, np.flip(edge_index_np, axis=0)]).astype(np.long)

        self.recommended_iids = set()
        self.picked_iids = set()

    def get_top_n_popular_items(self, n=10):
        """
        Get the top n movies from self.data.ratings.
        Remove the duplicates in self.data.ratings and sort it by movie count.
        After you find the top N popular movies' item id,
        look over the details information of item in self.data.movies
        :param n: the number of items, int
        :return: df: popular item dataframe, df
        """
        popular_iids = self.sorted_movie_count.index[:n].to_numpy()
        popular_item_df = self.movies.iloc[popular_iids]
        return popular_item_df

    def build_cold_user(self, demographic_info):
        raise NotImplementedError('Not cold start for explainable matrix factorization')

    def build_user(self, base_iids, demographic_info):
        """
        Build user profiles given the historical user interactions
        :param iids: user selected item ids, list
        :param demographic_info: (age, gender, occupation), tuple
        :return:
        """
        self.picked_iids = self.picked_iids.union(base_iids)

        inids = [self.dataset.e2nid_dict['iid'][iid] for iid in self.picked_iids]
        pos_nids = inids + [self.dataset.e2nid_dict['age'][demographic_info[0]], self.dataset.e2nid_dict['gender'][demographic_info[1]], self.dataset.e2nid_dict['occ'][demographic_info[2]]]
        self.first_hop_neighbour_edge_type = [self.dataset.edge_type_dict['user2item'] for _ in self.picked_iids] + \
                                             [
                                                 -self.dataset.edge_type_dict['age2user'],
                                                 -self.dataset.edge_type_dict['gender2user'],
                                                 -self.dataset.edge_type_dict['occ2user']
                                                 ]
        self.first_hop_neighbour_nids = pos_nids
        pos_emb = self.model.x[pos_nids].detach()
        neg_nids = np.random.choice(self.dataset.num_nodes, size=(pos_emb.shape[0],))
        neg_emb = self.model.x[neg_nids].detach()

        rating_r_idx = [self.dataset.edge_type_dict['user2item'] for _ in range(len(base_iids))]
        neg_r_idx = [self.dataset.edge_type_dict['age2user'], self.dataset.edge_type_dict['gender2user'], self.dataset.edge_type_dict['occ2user']]
        r_emb = torch.cat(
            [self.model.r[rating_r_idx],  - self.model.r[neg_r_idx]],
            dim=0
        ).detach()

        optimizer = torch.optim.Adam(
            params=[self.user_emb],
            lr=10e-3,
            weight_decay=0.001
        )

        print('Building user...')
        pbar = tqdm.tqdm(range(N_ITER))
        for _ in pbar:
            pos_sim = torch.sum((self.user_emb.repeat(repeats=(len(pos_nids), 1)) + r_emb) * pos_emb, dim=-1)
            neg_sim = torch.sum((self.user_emb.repeat(repeats=(len(pos_nids), 1)) + r_emb) * neg_emb, dim=-1)

            optimizer.zero_grad()
            loss = - (pos_sim.sigmoid().log().sum() + (-neg_sim).sigmoid().log().sum())
            loss.backward()
            optimizer.step()

            pbar.set_description('Loss: {:.3f}'.format(loss.detach().item()))
        print('User built')

        self.user_is_built = True

    def rebuild_user(self, new_iids):
        self.build_user(new_iids, None)

    def get_recommendations(self, num_recs):
        with torch.no_grad():
            assert self.user_is_built
            candidate_iids_np = np.random.choice(self.get_top_n_popular_items(1000).iid, 100)
            candidate_iids_np = np.array([i for i in candidate_iids_np if i not in self.picked_iids and i not in self.recommended_iids])
            candidate_inids_np = np.array([self.dataset.e2nid_dict['iid'][iid] for iid in candidate_iids_np])
            candidate_inids_t = torch.tensor(candidate_inids_np, dtype=torch.long, device=device)

            rating_r_idx = self.dataset.edge_type_dict['user2item']
            self.r_vec = self.model.r[torch.ones((candidate_inids_t.shape[0]), dtype=torch.long, device=device) * rating_r_idx]
            candidate_item_emb = self.model.x[candidate_inids_t]
            pred = torch.sum((self.user_emb + self.r_vec) * candidate_item_emb, dim=-1).sigmoid()
            rec_idx_t = torch.topk(pred, k=num_recs)[1]
            rec_inids_t = candidate_inids_t[rec_idx_t]
            rec_inids_np = rec_inids_t.detach().cpu().numpy()
            rec_iids_np = candidate_iids_np[rec_idx_t.cpu().numpy()]

            self.recommended_iids = self.recommended_iids.union(rec_iids_np)
            rec_item_df = self.movies[self.movies.iid.isin(rec_iids_np)]

            expl_tuple = [self.get_explanation(rec_inid) for rec_inid in rec_inids_np]
            exps = [_[0] for _ in expl_tuple]
            exp_types = [_[1] for _ in expl_tuple]

        return rec_item_df, exps, exp_types

    def compute_first_step_edge_prob(self, j_nid, edge_type):
        with torch.no_grad():
            if not hasattr(self, 'new_user_neighbour_prob_sum'):
                i_emb_t = self.user_emb
                edge_type_t = torch.tensor(self.first_hop_neighbour_edge_type, dtype=torch.long, device=device)
                signs = torch.sign(edge_type_t)
                signs[signs == 0] = 1
                abs_val = torch.abs(edge_type_t)
                trans_vec = self.model.r[abs_val] * signs.view(-1, 1)
                self.new_user_neighbour_prob_sum = \
                    torch.sum(torch.sum((i_emb_t + trans_vec) * self.model.x[self.first_hop_neighbour_nids], dim=-1).exp())
            j_emb_t = self.model.x[j_nid]

            r_emb_t = self.model.r[edge_type] if edge_type >= 0 else -self.model.r[-edge_type]

            i_emb_t = self.user_emb.squeeze(0)
            pred = torch.sum((i_emb_t + r_emb_t) * j_emb_t).exp()

            prob = (pred / self.new_user_neighbour_prob_sum).item()

            return prob

    def compute_edge_prob(self, i_nid, j_nid):
        with torch.no_grad():
            edge_indices = self.edge_index_np[0] == i_nid
            j_indices_t = torch.tensor(self.edge_index_np[1, edge_indices], dtype=torch.long, device=device)
            j_emb_t = self.model.x[j_indices_t]

            edge_type_t = self.edge_attr[edge_indices].view(-1)
            signs = torch.sign(edge_type_t)
            signs[signs == 0] = 1
            abs_val = torch.abs(edge_type_t)
            r_emb_t = self.model.r[abs_val] * signs.view(-1, 1)

            i_emb_t = self.model.x[i_nid].unsqueeze(0)
            pred = torch.sum((i_emb_t + r_emb_t) * j_emb_t, dim=-1).exp()

            prob = (pred[j_indices_t == j_nid] / torch.sum(pred)).item()

            return prob

    def compute_path_prob(self, path_np):
        j_nid = path_np[0]
        edge_type = self.first_hop_neighbour_edge_type[self.first_hop_neighbour_nids.index(j_nid)]
        first_step_prob = self.compute_first_step_edge_prob(j_nid, edge_type)
        probs = [self.compute_edge_prob(path_np[idx], path_np[idx + 1]) for idx in range(path_np.shape[0] - 1)]
        prob = functools.reduce(lambda x, y: x * y, probs, first_step_prob)

        return prob

    def get_explanation(self, rec_inid):
        edge_index_df = pd.DataFrame({'i': self.edge_index_np[0, :], 'j': self.edge_index_np[1, :]})
        path_df = edge_index_df.loc[edge_index_df.i.isin(self.first_hop_neighbour_nids)]
        path_df = path_df.rename(columns={'i': '1', 'j': '2'})
        path_nps = []
        for k in range(2, K):
            step_df = edge_index_df.rename(columns={'i': str(k), 'j': str(k + 1)})
            path_df = path_df.merge(step_df, on=str(k), how='inner')
            path_df = path_df.loc[path_df[str(k - 1)] != path_df[str(k + 1)]]
            path_np = path_df[path_df[str(k + 1)] == rec_inid].to_numpy()
            path_nps += list(path_np)

            path_df = path_df[rec_inid != path_df[str(k + 1)]]

        path_idx = np.array([self.compute_path_prob(path_np) for path_np in path_nps]).argmax()
        selected_path_np = path_nps[path_idx]

        path_entity_type = [self.dataset.nid2e_dict[nid][0] for nid in selected_path_np]

        path_entity = [self.dataset.nid2e_dict[nid][1] for nid in selected_path_np]
        path_entity_movie_name = []
        for typ, entity in zip(path_entity_type, path_entity):
            if typ == 'iid':
                path_entity_movie_name.append(self.movies.iloc[entity].title)
            else:
                path_entity_movie_name.append(entity)
        return path_entity_type, get_explanation_text(tuple(path_entity_type), path_entity_movie_name)


if __name__ == '__main__':
    recsys = MCFKGRecsys()
    recsys.get_top_n_popular_items()
    recsys.build_user([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 'F', 10])
    print(recsys.get_recommendations(10))
