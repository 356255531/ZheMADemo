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
parser.add_argument("--debug", default=0.01, help="")
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
parser.add_argument("--batch_size", type=int, default=256, help="")
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
    data = dataset.data
    train_pos_unid_inid_map, test_pos_unid_inid_map, neg_unid_inid_map = \
        data.train_pos_unid_inid_map[0], data.test_pos_unid_inid_map[0], data.neg_unid_inid_map[0]

    model = PAGATNet(num_nodes=dataset.data.num_nodes[0], **model_args).to(train_args['device'])

    optimizer = Adam(model.parameters(), lr=train_args['lr'], weight_decay=train_args['weight_decay'])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    HR_history = []
    NDCG_history = []
    loss_history = []
    for epoch in range(1, train_args['epochs'] + 1):
        model.train()
        epoch_losses = []
        u_nids = [data.e2nid[0]['uid'][uid] for uid in data.users[0].uid]
        randomizer.shuffle(u_nids)

        train_bar = tqdm.tqdm(u_nids)
        for user_idx, selected_u_nid in enumerate(train_bar):
            num_pos_i = len(train_pos_unid_inid_map[selected_u_nid])
            num_neg_i = len(neg_unid_inid_map[selected_u_nid])
            k = train_args['batch_size'] if num_pos_i >= train_args['batch_size'] and num_neg_i >= train_args['batch_size'] else min(num_pos_i, num_neg_i)
            if k == 0:
                continue
            pos_i_nids = randomizer.choices(train_pos_unid_inid_map[selected_u_nid], k=k)
            neg_i_nids = randomizer.choices(neg_unid_inid_map[selected_u_nid], k=k)
            pos_i_nid_df = pd.DataFrame({'u_nid': [selected_u_nid for _ in range(len(pos_i_nids))], 'pos_i_nid': pos_i_nids})
            neg_i_nid_df = pd.DataFrame({'u_nid': [selected_u_nid for _ in range(len(neg_i_nids))], 'neg_i_nid': neg_i_nids})
            pos_neg_pair_np = pd.merge(pos_i_nid_df, neg_i_nid_df, how='inner', on='u_nid').to_numpy()

            u_nids, pos_i_nids, neg_i_nids = pos_neg_pair_np.T
            occurred_nids_np = np.concatenate([list(set(u_nids)), list(set(pos_i_nids)), list(set(neg_i_nids))])
            edge_index_np = data.edge_index.cpu().numpy()
            path_index_batch_np = join(edge_index_np, target=occurred_nids_np, path_length=train_args['path_length'])
            path_index_batch = torch.from_numpy(path_index_batch_np).to(train_args['device'])
            propagated_node_emb = model(model.node_emb.weight, path_index_batch)[0]

            u_node_emb, pos_i_node_emb, neg_i_node_emb = \
                propagated_node_emb[u_nids], propagated_node_emb[pos_i_nids], propagated_node_emb[neg_i_nids]
            pred_pos = (u_node_emb * pos_i_node_emb).sum(dim=1)
            pred_neg = (u_node_emb * neg_i_node_emb).sum(dim=1)
            loss = - (pred_pos - pred_neg).sigmoid().log().mean()
            loss.backward()
            print(loss.cpu().item())
            optimizer.step()
            optimizer.zero_grad()

            propagated_node_emb = model(model.node_emb.weight, path_index_batch)[0]

            u_node_emb, pos_i_node_emb, neg_i_node_emb = \
                propagated_node_emb[u_nids], propagated_node_emb[pos_i_nids], propagated_node_emb[neg_i_nids]
            pred_pos = (u_node_emb * pos_i_node_emb).sum(dim=1)
            pred_neg = (u_node_emb * neg_i_node_emb).sum(dim=1)
            loss = - (pred_pos - pred_neg).sigmoid().log().mean()
            print(loss.cpu().item())

            loss = loss.cpu().item()
            epoch_losses.append(loss)
            train_bar.set_description('Epoch {}, User {},  loss {:.3f}'.format(epoch, user_idx, np.mean(epoch_losses)))

        # model.eval()
        # HR, NDCG, loss = metrics(epoch, model, dataset, train_args, rec_args)
        #
        # print('Epoch: {}, HR: {}, NDCG: {}, Loss: {}'.format(epoch, HR, NDCG, loss))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    print('Duration: {}, HR: {}, NDCG: {}, loss: {}'.format(t_start - t_end, np.mean(HR_history), np.mean(NDCG_history), np.mean(loss_history)))

    if not os.path.isdir(weights_folder):
        os.mkdir(weights_folder)
    weights_path = os.path.join(weights_folder, 'weights{}.py'.format(dataset.build_suffix()))
    torch.save(model.state_dict(), weights_path)

