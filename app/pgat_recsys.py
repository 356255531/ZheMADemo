__model__ = 'PAGAT'

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.datasets import MovieLens
from torch_geometric.utils import path
import os
import itertools
import pandas as pd

from .pagat import PAGATNet
from .utils import get_folder_path


def get_dataloader(data, batch_size):
    train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map = \
        data.train_pos_unid_inid_map[0], data.test_pos_unid_inid_map[0], data.neg_unid_inid_map[0]
    train_pos_pair_np = list(itertools.chain.from_iterable([[[k, vv] for vv in v] for k, v in train_pos_unid_inid_map.items()]))
    train_pos_pair_df = pd.DataFrame(train_pos_pair_np, columns=['u_nid', 'pos_i_nid'])
    test_pos_pair_np = list(itertools.chain.from_iterable([[[k, vv] for vv in v] for k, v in test_pos_unid_inid_map.items()]))
    test_pos_pair_df = pd.DataFrame(test_pos_pair_np, columns=['u_nid', 'pos_i_nid'])
    neg_pair_np = list(itertools.chain.from_iterable([[[k, vv] for vv in v] for k, v in neg_unid_inid_map.items()]))
    neg_pair_df = pd.DataFrame(neg_pair_np, columns=['u_nid', 'neg_i_nid'])
    train_dataloader = DataLoader(
        pd.merge(train_pos_pair_df, neg_pair_df, how='inner', on='u_nid').to_numpy(),
        shuffle=True,
        batch_size=batch_size)
    test_dataloader = DataLoader(
        pd.merge(test_pos_pair_df, neg_pair_df, how='inner', on='u_nid').to_numpy(),
        shuffle=True,
        batch_size=batch_size)
    return train_dataloader, test_dataloader


class PGATRecSys(object):

    def __init__(self, num_recs, dataset_args, model_args, device_args):
        self.user_is_built = False

        self.num_recs = num_recs
        self.device_args = device_args

        self.dataset = MovieLens(**dataset_args)
        self.data = self.dataset.data.to(device_args['device'])

        model_path = model_args['model_path']
        model_path = os.path.join(model_path, 'weights{}.pkl'.format(self.dataset.build_suffix()))
        del model_args['model_path']
        self.model = PAGATNet(num_nodes=self.data.num_nodes[0], **model_args).to(device_args['device'])

        self.model.eval()
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print("Model from {} successfully loaded!".format(model_path))
        except:
            print("No weights found in {}! Try use random initialized model.".format(model_path))
        self.recommended = []

    def get_top_n_popular_items(self, n=10):
        """
        Get the top n movies from self.data.ratings.
        Remove the duplicates in self.data.ratings and sort it by movie count.
        After you find the top N popular movies' item id,
        look over the details information of item in self.data.movies
        :param n: the number of items, int
        :return: df: popular item dataframe, df
        """

        ratings_df = self.data.ratings[0][['iid', 'movie_count']]
        ratings_df = ratings_df.sort_values(by='movie_count', ascending=False)
        ratings_df = ratings_df.drop_duplicates(subset=['iid'])
        popular_iids = [iid for iid in ratings_df.iid][:n]
        item_df = self.data.items[0]
        popular_item_df = item_df[item_df.iid.isin(popular_iids)]
        return popular_item_df

    def build_user(self, iids, demographic_info):
        """
        Build user profiles given the historical user interactions
        :param iids: user selected item ids, list
        :param demographic_info: (age, gender, occupation), tuple
        :return:
        """
        self.base_iids = iids
        self.demographic_info = demographic_info
        # Build edges for new user
        self.new_user_nid = self.model.node_emb.weight.shape[0]

        new_user_gender_nid = self.data.e2nid[0]['gender'][demographic_info[0]]
        new_user_occ_nid = self.data.e2nid[0]['occ'][int(demographic_info[1])]
        i_nids = [self.data.e2nid[0]['iid'][iid] for iid in iids]
        row = i_nids + [new_user_gender_nid, new_user_occ_nid]
        col = [self.new_user_nid for i in range(len(iids) + 2)]
        self.new_edge_index = torch.from_numpy(np.array([row + col, col + row])).long().to(self.device_args['device'])
        edge_index = torch.cat([self.data.edge_index, self.new_edge_index], dim=1)

        # Build path begins and ends with
        path_index_np = path.join(edge_index.cpu().numpy(), target=[self.new_user_nid])
        path_index = torch.from_numpy(path_index_np).to(self.device_args['device'])

        # Get new user embedding by applying message passing
        self.new_user_emb = torch.nn.Embedding(1, self.model.node_emb.weight.shape[1], max_norm=1, norm_type=2.0)
        new_node_emb = torch.cat((self.model.node_emb.weight, self.new_user_emb.weight), dim=0)
        self.propagated_new_user_emb = self.model(new_node_emb, path_index)[0][-1, :]

        self.user_is_built = True
        print('user building done...')

    def get_recommendations(self, rs_proportion):
        iids = self.get_top_n_popular_items(200).iid
        iids = [iid for iid in iids if iid not in self.recommended]
        if 'based_iids' in locals():
            iids = [iid for iid in iids if iid not in self.base_iids]
        rec_nids = [self.data.e2nid[0]['iid'][iid] for iid in iids]

        path_index = torch.from_numpy(path.join(self.data.edge_index.cpu().numpy(), target=rec_nids)).to(self.device_args['device'])
        propagated_node_emb = self.model(self.model.node_emb.weight, path_index)[0]
        rec_item_emb = propagated_node_emb[rec_nids, :]
        est_feedback = torch.sum(self.propagated_new_user_emb * rec_item_emb, dim=1).reshape(-1).cpu().detach().numpy()
        rec_iid_idx = [i for i in np.argsort(est_feedback)]

        rec_iids = [iids[idx] for idx in rec_iid_idx]
        # how to know what is the explanation type of rec_iids

        exp_tuple = [self.get_explanation(iid) for iid in rec_iids]
        exp, expl_types = [_[0] for _ in exp_tuple], [_[1] for _ in exp_tuple]

        iui_rec_index = [idx for idx, expl_type in enumerate(expl_types) if expl_type == 'IUI'][:rs_proportion['IUI']]
        iui_rec_iids = [rec_iids[idx] for idx in iui_rec_index]
        iui_rec_exp = [exp[idx] for idx in iui_rec_index]

        uiu_rec_index = [idx for idx, expl_type in enumerate(expl_types) if expl_type == 'UIU'][:rs_proportion['UIU']]
        uiu_rec_iids = [rec_iids[idx] for idx in uiu_rec_index]
        uiu_rec_exp = [exp[idx] for idx in uiu_rec_index]

        iudd_rec_index = [idx for idx, expl_type in enumerate(expl_types) if expl_type == 'IUDD'][:rs_proportion['IUDD']]
        iudd_rec_iids = [rec_iids[idx] for idx in iudd_rec_index]
        iudd_rec_exp = [exp[idx] for idx in iudd_rec_index]

        uicc_rec_index = [idx for idx, expl_type in enumerate(expl_types) if expl_type == 'UICC'][:rs_proportion['UICC']]
        uicc_rec_iids = [rec_iids[idx] for idx in uicc_rec_index]
        uicc_rec_exp = [exp[idx] for idx in uicc_rec_index]

        # iui_rec_padded_index = [idx for idx, expl_type in enumerate(expl_types) if expl_type == 'IUI'][rs_proportion['IUI']:]
        # iui_rec_padded_iids = [rec_iids[idx] for idx in iui_rec_padded_index]
        # iui_rec_padded_exp = [exp[idx] for idx in iui_rec_padded_index]

        temp_final_rec_iids = iui_rec_iids + uiu_rec_iids + iudd_rec_iids + uicc_rec_iids

        padded_rec_index = [idx for idx, expl_type in enumerate(expl_types)]
        padded_rec_iids = [iid for iid in rec_iids if iid not in temp_final_rec_iids]
        padded_rec_exp = [exp[idx] for idx in padded_rec_index]

        final_rec_iids = (temp_final_rec_iids + padded_rec_iids)[:10]

        temp_final_exp = iui_rec_exp + uiu_rec_exp + iudd_rec_exp + uicc_rec_exp
        final_exp = (temp_final_exp + padded_rec_exp)[:10]

        self.recommended += final_rec_iids

        item_df = self.data.items[0]
        rec_item_df = item_df[item_df.iid.isin(final_rec_iids)]

        return rec_item_df, final_exp

    def get_explanation(self, iid):
        movie_nid = self.data.e2nid[0]['iid'][iid]
        row = [movie_nid, self.new_user_nid]
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
    import argparse
    parser = argparse.ArgumentParser()
    default_poster_src = 'https://www.nehemiahmfg.com/wp-content/themes/dante/images/default-thumb.png'

    ########################## Define arguments ##########################
    # Dataset params
    parser.add_argument("--dataset", type=str, default='movielens', help="")
    parser.add_argument("--dataset_name", type=str, default='1m', help="")
    parser.add_argument("--num_core", type=int, default=10, help="")
    parser.add_argument("--step_length", type=int, default=2, help="")
    parser.add_argument("--train_ratio", type=float, default=False, help="")
    parser.add_argument("--debug", default=0.01, help="")

    # Model params
    parser.add_argument("--heads", type=int, default=4, help="")
    parser.add_argument("--dropout", type=float, default=0.6, help="")
    parser.add_argument("--emb_dim", type=int, default=64, help="")
    parser.add_argument("--repr_dim", type=int, default=16, help="")

    # Device params
    parser.add_argument("--device", type=str, default='cpu', help="")
    parser.add_argument("--gpu_idx", type=str, default='0', help="")

    args = parser.parse_args()

    # save id selected by users
    iid_list = []
    iid_list2 = []

    ########################## Define arguments ##########################
    data_folder, weights_folder, logger_folder = get_folder_path(args.dataset + args.dataset_name)

    ########################## Setup Device ##########################
    if not torch.cuda.is_available() or args.device == 'cpu':
        device = 'cpu'
    else:
        device = 'cuda:{}'.format(args.gpu_idx)

    ########################## Define parameters ##########################
    dataset_args = {
        'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
        'num_core': args.num_core, 'step_length': args.step_length, 'train_ratio': args.train_ratio,
        'debug': args.debug
    }
    model_args = {
        'heads': args.heads, 'emb_dim': args.emb_dim,
        'repr_dim': args.repr_dim, 'dropout': args.dropout
    }
    device_args = {'debug': args.debug, 'device': device, 'gpu_idx': args.gpu_idx}
    print('dataset params: {}'.format(dataset_args))
    print('task params: {}'.format(model_args))
    print('device_args params: {}'.format(device_args))

    recsys = PGATRecSys(num_recs=10, dataset_args=dataset_args, model_args=model_args, device_args=device_args)
    recsys.build_user(list(range(10)), ('M', 0))
    print(recsys.get_recommendations()[1])