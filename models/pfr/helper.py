from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import torch_dct as dct

def base_train(model, trainloader, optimizer, scheduler, epoch, args, criterion, group=4, rank=6):
    tl = Averager()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)

    for i, (data, train_label) in enumerate(tqdm_gen, 1):

        data, train_label = data.cuda(), train_label.cuda()

        X1 = dct.dct(data[:, 0, :, :])
        X2 = dct.dct(data[:, 1, :, :])
        X3 = dct.dct(data[:, 2, :, :])

        X_low1 = X1.clone()
        X_low2 = X2.clone()
        X_low3 = X3.clone()
        X_high1 = X1.clone()
        X_high2 = X2.clone()
        X_high3 = X3.clone()

        X_low1[:, :, rank:] = 0
        X_low2[:, :, rank:] = 0
        X_low3[:, :, rank:] = 0
        y_low1 = dct.idct(X_low1)
        y_low2 = dct.idct(X_low2)
        y_low3 = dct.idct(X_low3)
        y_low = torch.cat((y_low1.unsqueeze(1), y_low2.unsqueeze(1), y_low3.unsqueeze(1)), dim=1)

        X_high1[:, :, :rank] = 0
        X_high2[:, :, :rank] = 0
        X_high3[:, :, :rank] = 0
        y_high1 = dct.idct(X_high1)
        y_high2 = dct.idct(X_high2)
        y_high3 = dct.idct(X_high3)
        y_high = torch.cat((y_high1.unsqueeze(1), y_high2.unsqueeze(1), y_high3.unsqueeze(1)), dim=1)

        logits_ori, raw = model(data)
        logits_low, raw_low = model(y_low)
        logits_high, raw_high = model(y_high)

        atte_gen = raw_low
        atte_dis = raw_high

        # =========================== Discriminative (High Freq) Analy ==================================

        atte_dis = F.softmax(torch.mul(atte_dis, raw) + raw)
        atte_dis = F.adaptive_avg_pool2d(atte_dis, 1)
        atte_dis = atte_dis.squeeze(-1).squeeze(-1)

        class_num = int(atte_dis.shape[0] / args.episode_shot)

        group_class_num = int(class_num / group)
        group_dis = atte_dis.view([args.episode_shot, class_num, -1])
        mean_dis = group_dis.mean(0)

        neg_l2_dist = pdist(atte_dis, mean_dis).neg().view(class_num * args.episode_shot,
                                                           class_num)  # way*query_shot,way
        atte_dis_logit = neg_l2_dist / atte_dis.shape[-1]
        log_prediction = F.log_softmax(atte_dis_logit, dim=1)
        target = torch.LongTensor(range(class_num)).repeat(args.episode_shot).cuda()
        dis_loss = criterion(log_prediction, target)


        # =========================== General (Low Freq) Analy ==================================

        atte_gen = F.softmax(torch.mul(atte_gen, raw) + raw)
        atte_gen = F.adaptive_avg_pool2d(atte_gen, 1)
        atte_gen = atte_gen.squeeze(-1).squeeze(-1)

        group_gen = atte_gen.view([args.episode_shot, class_num, -1])
        group_gen = group_gen.mean(0)

        inc_mean = group_gen.view(group_class_num, group, -1)
        inc_mean = inc_mean.mean(0)


        neg_l2_dist = pdist(atte_gen, inc_mean).neg().view(class_num * args.episode_shot, group)  # way*query_shot,way

        atte_gen_logit = neg_l2_dist / atte_gen.shape[-1]
        log_prediction = F.log_softmax(atte_gen_logit, dim=1)


        target = torch.LongTensor([j // group_class_num for j in range(class_num)]).repeat(args.episode_shot).cuda()

        tri_loss = criterion(log_prediction, target)

        # =========================== Cross-Entropy ==================================

        logits_ = logits_ori[:, :args.base_class]
        ce_loss = F.cross_entropy(logits_, train_label)

        total_loss = args.alpha*dis_loss + (1-args.alpha)*ce_loss + args.alpha*tri_loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f}'.format(epoch, lrc, total_loss.item()))
        tl.add(total_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta

def pdist(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''

    return torch.cdist(x, y, p=2) ** 2

def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)
            embedding = F.adaptive_avg_pool2d(embedding, 1)
            embedding = embedding.squeeze(-1).squeeze(-1)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list

    return model

def test(model, testloader, epoch,args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()

    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits, _ = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()

        if session == 0:
            print('Session {}, epo {}, test, loss={:.4f} acc={:.4f}'.format(session, epoch, vl, va))
        else:
            print('Session {}, test acc={:.4f}'.format(session, va))

    return vl, va
