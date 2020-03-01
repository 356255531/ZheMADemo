import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm


def hit(hit_vec):
    if hit_vec.sum() > 0:
        return 1
    else:
        return 0


def ndcg(hit_vec):
    ndcg_vec = [np.reciprocal(np.log2(idx+2)) for idx, if_hit in enumerate(hit_vec.cpu().numpy()) if if_hit]
    return np.sum(ndcg_vec)


def metrics(epoch, model, dataset, train_args, rec_args):
    HR, NDCG, losses = [], [], []

    test_loader = DataLoader(dataset.data.test_user_pos_neg_pair[0], shuffle=True, batch_size=train_args['batch_size'])
    test_bar = tqdm.tqdm(test_loader)
    for user_pos_neg_pair_batch in test_bar:
        u_nid, pos_i_nid, neg_i_nid = user_pos_neg_pair_batch.T
        occ_nid = np.concatenate((u_nid, pos_i_nid, neg_i_nid))
        path_index_batch = torch.from_numpy(dataset.data.path_np[0][:, np.isin(dataset.data.path_np[0][-1, :], occ_nid)]).to(
            train_args['device'])
        propagated_node_emb = model(model.node_emb.weight, path_index_batch)[0]
        u_nid, pos_i_nid, neg_i_nid = u_nid.to(train_args['device']), pos_i_nid.to(train_args['device']), neg_i_nid.to(train_args['device'])
        u_node_emb, pos_i_node_emb, neg_i_node_emb = propagated_node_emb[u_nid], propagated_node_emb[pos_i_nid], \
                                                     propagated_node_emb[neg_i_nid]
        pred_pos = (u_node_emb * pos_i_node_emb).sum(dim=1)
        pred_neg = (u_node_emb * neg_i_node_emb).sum(dim=1)

        loss = - (pred_pos - pred_neg).sigmoid().log().mean().item()

        _, indices = torch.topk(torch.cat([pred_pos, pred_neg], dim=0), rec_args['num_recs'])
        hit_vec = indices < train_args['batch_size']

        HR.append(hit(hit_vec))
        NDCG.append(ndcg(hit_vec))
        losses.append(loss)
        test_bar.set_description('Epoch {}: loss {}, HR {}, NDCG {}'.format(epoch, np.mean(losses), np.mean(HR), np.mean(NDCG)))

    return np.mean(HR), np.mean(NDCG), np.mean(losses)