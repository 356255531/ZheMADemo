import argparse
import torch
import os
import itertools
from torch.utils.data import DataLoader
from torch_geometric.datasets import MovieLens
from torch_geometric.utils.path import join
from torch.optim import Adam
import time
import numpy as np
import tqdm
import random
import pandas as pd

from utils import get_folder_path
from pagat import PAGATNet
from eval_rec_sys import metrics
parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument("--dataset", type=str, default='movielens', help="")
parser.add_argument("--dataset_name", type=str, default='1m', help="")
parser.add_argument("--num_core", type=int, default=10, help="")
parser.add_argument("--num_feat_core", type=int, default=10, help="")
parser.add_argument("--train_ratio", type=float, default=0.8, help="")
parser.add_argument("--debug", default=None, help="")
parser.add_argument("--seed", default=2019, help="")


# Model params
parser.add_argument("--heads", type=int, default=4, help="")
parser.add_argument("--dropout", type=float, default=0.6, help="")
parser.add_argument("--emb_dim", type=int, default=16, help="")
parser.add_argument("--repr_dim", type=int, default=16, help="")
parser.add_argument("--hidden_size", type=int, default=64, help="")

# Train params
parser.add_argument("--path_length", type=int, default=2, help="")
parser.add_argument("--device", type=str, default='cuda', help="")
parser.add_argument("--gpu_idx", type=str, default='0', help="")
parser.add_argument("--epochs", type=int, default=20, help="")
parser.add_argument("--opt", type=str, default='adam', help="")
parser.add_argument("--loss", type=str, default='mse', help="")
parser.add_argument("--batch_size", type=int, default=1, help="")
parser.add_argument("--lr", type=float, default=1e-4, help="")
parser.add_argument("--weight_decay", type=float, default=10e-3, help="")
parser.add_argument("--early_stopping", type=int, default=40, help="")

# Recommender params
parser.add_argument("--num_recs", type=int, default=10, help="")


args = parser.parse_args()

# Setup data and weights file path
data_folder, weights_folder, logger_folder = get_folder_path(args.dataset + args.dataset_name)

# Setup device
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)

# Setup args
dataset_args = {
    'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
    'num_core': args.num_core, 'num_feat_core': args.num_feat_core,
    'seed': args.seed, 'train_ratio': args.train_ratio,
    'debug': args.debug
}
model_args = {
    'heads': args.heads, 'hidden_size': args.hidden_size, 'emb_dim': args.emb_dim,
    'repr_dim': args.repr_dim, 'dropout': args.dropout
}
train_args = {
    'path_length': args.path_length,
    'debug': args.debug,
    'opt': args.opt, 'loss': args.loss,
    'epochs': args.epochs, 'batch_size': args.batch_size,
    'weight_decay': args.weight_decay, 'lr': args.lr, 'device': device,
    'weights_folder': weights_folder, 'logger_folder': logger_folder}
rec_args = {
    'num_recs': args.num_recs
}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))
print('rec params: {}'.format(rec_args))


if __name__ == '__main__':
    randomizer = random.Random(2019)
    dataset = MovieLens(**dataset_args)
    dataset.data = dataset.data.to(train_args['device'])
    model = PAGATNet(num_nodes=dataset.data.num_nodes[0], **model_args).to(train_args['device'])

    optimizer = Adam(model.parameters(), lr=train_args['lr'], weight_decay=train_args['weight_decay'])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    HR_history = []
    NDCG_history = []
    loss_history = []
    for epoch in range(1, train_args['epochs'] + 1):
        data = dataset.data
        train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map = \
            data.train_pos_unid_inid_map[0], data.test_pos_unid_inid_map[0], data.neg_unid_inid_map[0]

        model.train()
        epoch_losses = []
        u_nids = [data.e2nid[0]['uid'][uid] for uid in data.users[0].uid]

        train_bar = randomizer.shuffle(u_nids)
        for u_nid in train_bar:
            pos_i_nids = train_pos_unid_inid_map[u_nid]
            neg_i_nids = neg_unid_inid_map[u_nid]
            pos_i_nid_df = pd.DataFrame({'u_nid': [u_nid for _ in range(len(pos_i_nids))], 'pos_i_nid': pos_i_nids})
            neg_i_nid_df = pd.DataFrame({'u_nid': [u_nid for _ in range(len(pos_i_nids))], 'neg_i_nid': neg_i_nids})
            pos_neg_pair_np = pd.merge(pos_i_nid_df, neg_i_nid_df, how='inner', on='u_nid').to_numpy()
            pos_neg_pair_loader = DataLoader(torch.from_numpy(pos_neg_pair_np).to(train_args['device']), shuffle=True, batch_size=train_args['batch_size'])

            for pos_neg_pair_batch in pos_neg_pair_loader:
                u_nid_t, pos_i_nid_t, neg_i_nid_t = pos_neg_pair_batch.T
                occurred_nids_np = np.concatenate([np.array([u_nid_t]), pos_i_nid_t, neg_i_nid_t])
                edge_index_np = data.edge_index.item()
                edge_index_idx = np.isin(edge_index_np[1, :], occurred_nids_np)
                edge_index_suf = data.edge_index[:, edge_index_idx]
                path_index_batch = join(data.edge_index, edge_index_suf)
                propagated_node_emb = model(model.node_emb.weight, path_index_batch)[0]

                u_nid, pos_i_nid, neg_i_nid = u_nid.to(device), pos_i_nid.to(device), neg_i_nid.to(device)
                u_node_emb, pos_i_node_emb, neg_i_node_emb = propagated_node_emb[u_nid], propagated_node_emb[pos_i_nid], propagated_node_emb[neg_i_nid]
                pred_pos = (u_node_emb * pos_i_node_emb).sum(dim=1)
                pred_neg = (u_node_emb * neg_i_node_emb).sum(dim=1)
                loss = - (pred_pos - pred_neg).sigmoid().log().mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_losses.append(loss.cpu().item())
                train_bar.set_description('Epoch {}: loss {}'.format(epoch, np.mean(epoch_losses)))

        model.eval()
        HR, NDCG, loss = metrics(epoch, model, dataset, train_args, rec_args)

        print('Epoch: {}, HR: {}, NDCG: {}, Loss: {}'.format(epoch, HR, NDCG, loss))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    print('Duration: {}, HR: {}, NDCG: {}, loss: {}'.format(t_start - t_end, np.mean(HR_history), np.mean(NDCG_history), np.mean(loss_history)))

    if not os.path.isdir(weights_folder):
        os.mkdir(weights_folder)
    weights_path = os.path.join(weights_folder, 'weights{}.py'.format(dataset.build_suffix()))
    torch.save(model.state_dict(), weights_path)

