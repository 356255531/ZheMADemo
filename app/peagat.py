import torch
import os
import numpy as np
import pandas as pd
import tqdm
import functools

from torch_geometric.nn.inits import glorot

from graph_recsys_benchmark.models import PEAGCNRecsysModel
from graph_recsys_benchmark.datasets import MovieLens
from graph_recsys_benchmark.utils import get_folder_path
from graph_recsys_benchmark.utils import get_opt_class, load_model

DS = 'Movielens1m'
K = 3

# Setup data and weights file path
data_folder, weights_folder, logger_folder = \
    get_folder_path(model='Graph', dataset='Movielens1m', loss_type='BPR')

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup args
dataset_args = {
    'root': data_folder, 'dataset': 'Movielens', 'name': '1m',
    'if_use_features': 'False', 'num_negative_samples': 4,
    'num_core': 10, 'num_feat_core': 10,
    'cf_loss_type': 'BPR', 'type': 'hete',
    'sampling_strategy': 'random', 'entity_aware': False
}
model_args = {
    'model_type': 'Graph',
    'jump_mode': 'lstm', 'jump_channels': 64, 'jump_num_layers': 2,
    'if_use_features': False,
    'emb_dim': 64, 'hidden_size': 64,
    'repr_dim': 16, 'dropout': 0,
    'meta_path_steps': [3 for _ in range(5)], 'num_heads': 1,
    'channel_aggr': 'att', 'entity_aware': 'false',
    'entity_aware_coff': 0.1
}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))


def get_explanation_text(exp_type, exp_entity):
    if exp_type == ('iid', 'genre', 'iid'):
        expl = 'We recommend this movie, since you may like {} movies like "{}"'.format(exp_entity[1], exp_entity[0])
    elif exp_type == ('iid', 'actor', 'iid'):
        expl = 'We recommend this movie, since you may like the actor {}, who also acted the movie "{}"'.format(exp_entity[1], exp_entity[0])
    elif exp_type == ('iid', 'director', 'iid'):
        expl = 'We recommend this movie, since you may like the director {}, who also directed the the movie "{}"'.format(exp_entity[1], exp_entity[0])
    elif exp_type == ('iid', 'writer', 'iid'):
        expl = 'We recommend this movie, since you may like the writer {}, who also wrote the movie "{}"'.format(exp_entity[1], exp_entity[0])
    elif exp_type == ('iid', 'uid', 'iid'):
        expl = 'We recommend this movie, since the user, who watched the movie "{}" also likes this movie'.format(exp_entity[0])
    elif exp_type == ('age', 'uid', 'iid'):
        expl = 'You may like this movie, the user who is of the same age like you also likes this movie'
    elif exp_type == ('occ', 'uid', 'iid'):
        expl = 'You may like this movie, the user who has the same occ like you also likes this movie'
    elif exp_type == ('gender', 'uid', 'iid'):
        expl = 'You may like this movie, the user who has the same gender like you also likes this movie'
    elif exp_type == ('iid', 'year', 'iid'):
        expl = 'You may like this movie, since you may like movies from the same age of the movie "{}"'.format(exp_entity[0])
    else:
        import pdb
        pdb.set_trace()
        raise NotImplementedError
    return expl


class PEAGCNRecsysModel(PEAGCNRecsysModel):
    def update_graph_input(self, dataset):
        user2item_edge_index = torch.from_numpy(dataset.edge_index_nps['user2item']).long().to(
            device)
        year2item_edge_index = torch.from_numpy(dataset.edge_index_nps['year2item']).long().to(
            device)
        actor2item_edge_index = torch.from_numpy(dataset.edge_index_nps['actor2item']).long().to(
            device)
        director2item_edge_index = torch.from_numpy(dataset.edge_index_nps['director2item']).long().to(
            device)
        writer2item_edge_index = torch.from_numpy(dataset.edge_index_nps['writer2item']).long().to(
            device)
        genre2item_edge_index = torch.from_numpy(dataset.edge_index_nps['genre2item']).long().to(
            device)
        age2user_edge_index = torch.from_numpy(dataset.edge_index_nps['age2user']).long().to(
            device)
        gender2user_edge_index = torch.from_numpy(dataset.edge_index_nps['gender2user']).long().to(
            device)
        occ2user_edge_index = torch.from_numpy(dataset.edge_index_nps['occ2user']).long().to(
            device)

        meta_path_edge_indicis_1 = torch.cat([torch.flip(year2item_edge_index, dims=[0]), year2item_edge_index,
                                              torch.flip(user2item_edge_index, dims=[0])], dim=1)
        meta_path_edge_indicis_2 = torch.cat([torch.flip(actor2item_edge_index, dims=[0]), actor2item_edge_index,
                                              torch.flip(user2item_edge_index, dims=[0])], dim=1)
        meta_path_edge_indicis_3 = torch.cat([torch.flip(writer2item_edge_index, dims=[0]), writer2item_edge_index,
                                              torch.flip(user2item_edge_index, dims=[0])], dim=1)
        meta_path_edge_indicis_4 = torch.cat([torch.flip(director2item_edge_index, dims=[0]), director2item_edge_index,
                                              torch.flip(user2item_edge_index, dims=[0])], dim=1)
        meta_path_edge_indicis_5 = torch.cat([torch.flip(genre2item_edge_index, dims=[0]), genre2item_edge_index,
                                              torch.flip(user2item_edge_index, dims=[0])], dim=1)
        meta_path_edge_indicis_6 = torch.cat([torch.flip(genre2item_edge_index, dims=[0]), genre2item_edge_index,
                                              torch.flip(user2item_edge_index, dims=[0])], dim=1)
        meta_path_edge_indicis_7 = torch.cat([torch.flip(user2item_edge_index, dims=[0]), torch.flip(age2user_edge_index, dims=[0]),
                                              age2user_edge_index], dim=1)
        meta_path_edge_indicis_8 = torch.cat([torch.flip(user2item_edge_index, dims=[0]), torch.flip(gender2user_edge_index, dims=[0]),
                                              gender2user_edge_index], dim=1)
        meta_path_edge_indicis_9 = torch.cat([torch.flip(user2item_edge_index, dims=[0]), torch.flip(occ2user_edge_index, dims=[0]),
                                              occ2user_edge_index], dim=1)
        meta_path_edge_index_list = [
            meta_path_edge_indicis_1, meta_path_edge_indicis_2, meta_path_edge_indicis_3,
            meta_path_edge_indicis_4, meta_path_edge_indicis_5, meta_path_edge_indicis_6,
            meta_path_edge_indicis_7, meta_path_edge_indicis_8, meta_path_edge_indicis_9
        ]

        return meta_path_edge_index_list


def get_utils():
    dataset = MovieLens(**dataset_args)

    model_args['num_nodes'] = dataset.num_nodes
    model_args['dataset'] = dataset
    model = PEAGCNRecsysModel(**model_args).to(device)

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


class PEAGCNRecsys(object):

    def __init__(self):
        self.dataset, self.model = get_utils()

        self.user_is_built = False

        rating_file = 'checkpoint/data/{}/processed/ratings.csv'.format(DS)
        movie_file = 'checkpoint/data/{}/processed/movies.csv'.format(DS)

        assert os.path.exists(rating_file)
        assert os.path.exists(movie_file)
        self.ratings = pd.read_csv(rating_file, sep=';')
        movies = pd.read_csv(movie_file, sep=';').fillna('')
        movie_count = self.ratings['iid'].value_counts()
        movie_count.name = 'movie_count'
        self.movies = movies.join(movie_count, on='iid')

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
        popular_item_df = self.movies.sort_values('movie_count', ascending=False).iloc[:n]
        return popular_item_df

    def build_cold_user(self, demographic_info):
        """
        Build user profiles given the historical user interactions
        :param iids: user selected item ids, list
        :param demographic_info: (age, gender, occupation), tuple
        :return:
        """
        self.build_user([], demographic_info)

    def build_user(self, base_iids, demographic_info):
        """
        Build user profiles given the historical user interactions
        :param iids: user selected item ids, list
        :param demographic_info: (age, gender, occupation), tuple
        :return:
        """
        assert not self.user_is_built

        self.new_user_nid = self.model.x.shape[0]
        new_user_embedding = torch.Tensor(1, model_args['emb_dim']).to(device=device)
        glorot(new_user_embedding)
        self.model.x = torch.nn.Parameter(
            torch.cat(
                [
                    self.model.x,
                    new_user_embedding
                ],
                dim=0
            )
        )

        self.picked_iids = self.picked_iids.union(base_iids)

        # Build edges for new user
        age = int(demographic_info[0])
        gender = demographic_info[1]
        occ = int(demographic_info[2])

        age_nid = self.dataset.e2nid_dict['age'][age]
        occ_nid = self.dataset.e2nid_dict['occ'][occ]
        inids = [self.dataset.e2nid_dict['iid'][iid] for iid in self.picked_iids]

        self.dataset.edge_index_nps['age2user'] = np.hstack([self.dataset.edge_index_nps['age2user'], [[age_nid], [self.new_user_nid]]])
        self.dataset.edge_index_nps['occ2user'] = np.hstack([self.dataset.edge_index_nps['occ2user'], [[occ_nid], [self.new_user_nid]]])

        if gender not in ['M', 'F']:
            pos_nids = inids + [age_nid, occ_nid]
        else:
            gender_nid = self.dataset.e2nid_dict['gender'][gender]
            pos_nids = inids + [age_nid, occ_nid, gender_nid]

            self.dataset.edge_index_nps['gender2user'] = np.hstack([self.dataset.edge_index_nps['gender2user'], [[gender_nid], [self.new_user_nid]]])
        self.first_hop_neighbour_nids = pos_nids

        # Build the interactions
        base_inids = [self.dataset.e2nid_dict['iid'][iid] for iid in base_iids]
        new_interactions = np.vstack([[self.new_user_nid for _ in range(len(base_iids))], base_inids])
        self.dataset.edge_index_nps['user2item'] = np.hstack([self.dataset.edge_index_nps['user2item'], new_interactions])

        self.model.meta_path_edge_indices = self.model.update_graph_input(self.dataset)
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
        assert self.user_is_built

        new_iids = [iid for iid in new_iids if iid not in self.picked_iids]

        # Build the interactions
        new_inids = [self.dataset.e2nid_dict['iid'][iid] for iid in new_iids]
        new_interactions = np.vstack([[self.new_user_nid for _ in range(len(new_inids))], new_inids])
        self.dataset.edge_index_nps['user2item'] = np.hstack(
            [self.dataset.edge_index_nps['user2item'], new_interactions])

        # Update explanation components
        self.first_hop_neighbour_nids += new_inids

        # Update graph
        self.model.meta_path_edge_indices = self.model.update_graph_input(self.dataset)
        self.model.eval()

    def get_recommendations(self, num_recs):
        assert self.user_is_built

        with torch.no_grad():
            candidate_iids_np = np.random.choice(self.get_top_n_popular_items(1000).iid, 100)
            candidate_iids_np = np.array([i for i in candidate_iids_np if i not in self.picked_iids and i not in self.recommended_iids])
            candidate_inids_np = np.array([self.dataset.e2nid_dict['iid'][iid] for iid in candidate_iids_np])

            candidate_inids_t = torch.tensor(candidate_inids_np, dtype=torch.long, device=device)
            user_emb = self.model.cached_repr[self.new_user_nid].unsqueeze(0)
            candidate_item_emb = self.model.cached_repr[candidate_inids_t]
            pred = torch.sum(user_emb * candidate_item_emb, dim=-1).sigmoid()
            rec_idx = torch.topk(pred, k=num_recs)[1].cpu().numpy()
            rec_iids_np = candidate_iids_np[rec_idx]
            rec_inids_np = candidate_inids_np[rec_idx]
            rec_item_df = self.movies[self.movies.iid.isin(rec_iids_np)]

            pbar = tqdm.tqdm(rec_inids_np, total=rec_inids_np.shape[0])
            expl_types, expl_texts = [], []
            for idx, rec_inid in enumerate(pbar):
                pbar.set_description('Fetching explanation for {} movie'.format(idx + 1))
                path_entity_type, expl_text = self.get_explanation(rec_inid)
                expl_types.append(path_entity_type)
                expl_texts.append(expl_text)

        return rec_item_df, expl_types, expl_texts

    def compute_first_step_edge_prob(self, j_nid):
        with torch.no_grad():
            if not hasattr(self, 'new_user_neighbour_prob_sum'):
                i_emb_t = self.model.cached_repr[self.new_user_nid].unsqueeze(0)
                self.new_user_neighbour_prob_sum = \
                    torch.sum(torch.sum(i_emb_t * self.model.cached_repr[self.first_hop_neighbour_nids], dim=-1).exp())

            i_emb_t = self.model.cached_repr[self.new_user_nid].unsqueeze(0)
            j_emb_t = self.model.cached_repr[j_nid]
            pred = torch.sum(i_emb_t * j_emb_t).exp()

            prob = (pred / self.new_user_neighbour_prob_sum).item()

            return prob

    def compute_edge_prob(self, i_nid, j_nid):
        with torch.no_grad():
            edge_indices = self.edge_index_np[0] == i_nid
            j_indices_t = torch.tensor(self.edge_index_np[1, edge_indices], dtype=torch.long, device=device)
            j_emb_t = self.model.cached_repr[j_indices_t]

            i_emb_t = self.model.cached_repr[i_nid].unsqueeze(0)
            pred = torch.sum(i_emb_t * j_emb_t, dim=-1).exp()

            prob = (pred[j_indices_t == j_nid] / torch.sum(pred)).item()

            return prob

    def compute_path_prob(self, path_np):
        j_nid = path_np[0]
        first_step_prob = self.compute_first_step_edge_prob(j_nid)
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

        if len(path_nps) == 0:
            return [], ""
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
    recsys = PEAGCNRecsys()
    recsys.get_top_n_popular_items()
    recsys.build_user([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 'F', 10])
    print(recsys.get_recommendations(10))
