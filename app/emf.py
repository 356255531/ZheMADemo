import pandas as pd
import torch
import random as rd

import itertools
import os

EMBEDDING_DIM = 30
DS = 'Movielens1m'


TEMPLATE = {
    'p': '\"{}\" is one of the most popular movies that you would like.',
    'u': 'The same users as you like \"{}\"',
    'i': '\"{}\" is simlilar to your watched movie \"{}\"'
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EMFRecsys(object):

    def __init__(self):
        self.user_is_built = False
        self.selected_uid = None

        rating_file = 'checkpoint/data/{}/processed/ratings.csv'.format(DS)
        movie_file = 'checkpoint/data/{}/processed/movies.csv'.format(DS)

        assert os.path.exists(rating_file)
        assert os.path.exists(movie_file)
        self.ratings = pd.read_csv(rating_file, sep=';')
        self.num_users = self.ratings.uid.unique().shape[0]
        self.num_movies = self.ratings.iid.unique().shape[0]

        self.movies = pd.read_csv(movie_file, sep=';').fillna('')
        movie_count = self.ratings['iid'].value_counts()
        movie_count.name = 'movie_count'
        self.sorted_movie_count = movie_count.sort_values(ascending=False)

        # Create user interactions map
        self.uid_iid_map = {}
        for uid in list(self.ratings.uid.unique()):
            self.uid_iid_map[uid] = list(self.ratings[self.ratings.uid == uid].iid)

        # Create embedding
        self.user_embedding = torch.nn.Embedding(self.num_users, EMBEDDING_DIM, max_norm=1).to(device)
        self.item_embedding = torch.nn.Embedding(self.num_movies, EMBEDDING_DIM, max_norm=1).to(device)
        try:
            embedding_file = 'checkpoint/weights/{}/EMF/embedding.pkl'.format(DS)
            with open(embedding_file, mode='rb+') as f:
                checkpoint = torch.load(f, map_location=device)
            self.user_embedding.weight = checkpoint['user_embedding']
            self.item_embedding.weight = checkpoint['item_embedding']
        except:
            print("No checkpoint found! Load new!")

        self.recommended = set()
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

        iid_hit = 0
        uid_iid_number = 10000
        for uid, iids in self.uid_iid_map.items():
            current_iid_hit = len(self.picked_iids.intersection(iids))
            if iid_hit < current_iid_hit:
                self.selected_uid = uid
                iid_hit = current_iid_hit
            elif iid_hit == current_iid_hit:
                if len(iids) < uid_iid_number:
                    uid_iid_number = len(iids)
                    self.selected_uid = uid

        self.user_is_built = True

    def rebuild_user(self, new_iids):
        self.build_user(list(set(self.picked_iids).union(set(new_iids))), None)

    def get_recommendations(self, num_recs):
        with torch.no_grad():
            assert self.user_is_built

            popular_iids = list(self.get_top_n_popular_items(100).iid)
            popular_iids = [iid for iid in popular_iids if iid not in self.recommended and iid not in self.picked_iids][:10]

            all_uids_t = torch.arange(self.num_users, dtype=torch.long, device=device)
            selected_uid_t = torch.tensor([self.selected_uid], dtype=torch.long, device=device)
            batch_user_dists_t = torch.sum(self.user_embedding(all_uids_t) * self.user_embedding(selected_uid_t), dim=-1)
            neighbour_uids_np = torch.topk(batch_user_dists_t, k=11, largest=True)[1][1:].cpu().numpy()
            user_cf_iids = set(itertools.chain.from_iterable([self.uid_iid_map[neighbour] for neighbour in neighbour_uids_np]))
            user_cf_iids = list(user_cf_iids.difference(self.picked_iids))
            user_cf_iids = [iid for iid in user_cf_iids if iid not in self.recommended and iid not in self.picked_iids]
            user_cf_iids = rd.choices(population=user_cf_iids, k=10)

            all_iids_t = torch.arange(self.num_movies, dtype=torch.long, device=device)
            item_cf_iids = []
            for iid in self.picked_iids:
                iid_t = torch.tensor([iid], dtype=torch.long, device=device)
                batch_item_dists = torch.sum(self.item_embedding(all_iids_t) * self.item_embedding(iid_t), dim=-1)
                item_cf_iids.append((torch.topk(batch_item_dists, k=2, largest=True)[1][1].item(), iid))
            item_cf_iids = item_cf_iids if len(item_cf_iids) <= 10 else rd.choices(population=item_cf_iids, k=10)

            candidates_iid_tuples = [('p', iid) for iid in popular_iids]
            candidates_iid_tuples += [('u', iid) for iid in user_cf_iids]
            candidates_iid_tuples += [('i', iid) for iid in item_cf_iids]

            candidate_iids = [candidates_iid[1] if candidates_iid[0] != 'i' else candidates_iid[1][0] for candidates_iid in candidates_iid_tuples]
            candidate_iids_t = torch.tensor(candidate_iids, dtype=torch.long, device=device)
            est_ratings = torch.sum(self.item_embedding(candidate_iids_t) * self.user_embedding(selected_uid_t), dim=-1)
            candidate_indices_t = torch.topk(est_ratings, k=num_recs, largest=True)[1]
            rec_iids = candidate_iids_t[candidate_indices_t].cpu().numpy()
            self.recommended = self.recommended.union(rec_iids)

            rec_item_df = self.movies[self.movies.iid.isin(rec_iids)]
            rec_iid_tuples = [candidates_iid_tuples[idx] for idx in candidate_indices_t.cpu().numpy()]

            exps, exp_types = self.get_explanation(rec_iid_tuples)

        return rec_item_df, exps, exp_types

    def get_explanation(self, rec_iid_tuples):
        expls_type = [rec_iid_tuple[0] for rec_iid_tuple in rec_iid_tuples]
        expls_template = [TEMPLATE[rec_iid_tuple[0]] for rec_iid_tuple in rec_iid_tuples]
        expls = [template.format(self.movies.iloc[rec_iid_tuple[1]].title) if rec_iid_tuple[0] != 'i' else template.format(self.movies.iloc[rec_iid_tuple[1][0]].title, self.movies.iloc[rec_iid_tuple[1][1]].title) for template, rec_iid_tuple in zip(expls_template, rec_iid_tuples)]
        return expls, expls_type


if __name__ == '__main__':
    recsys = EMFRecsys()
    recsys.build_user([1, 2, 3, 4, 5, 6, 7, 8, 9], None)
    print(recsys.get_recommendations(10)[1])