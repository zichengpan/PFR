import numpy as np
import torch
from dataloader.sampler import CategoriesSampler
from torch.utils.data import Dataset
def set_up_datasets(args):

    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11


    if args.dataset == 'StanfordDog':
        import dataloader.StanfordDog.StanfordDog as Dataset
        args.base_class = 80
        args.num_classes = 120
        args.way = 5
        args.shot = 5
        args.sessions = 9

    if args.dataset == 'StanfordCar':
        import dataloader.StanfordCar.StanfordCar as Dataset
        args.base_class = 106
        args.num_classes = 196
        args.way = 10
        args.shot = 5
        args.sessions = 10

    if args.dataset == 'Aircraft':
        import dataloader.Aircraft.Aircraft as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    args.Dataset=Dataset
    return args


def get_base_dataloader(args):
    class_index = np.arange(args.base_class)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)
    if args.dataset == 'StanfordDog':
        trainset = args.Dataset.StanfordDog(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.StanfordDog(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'Aircraft':
        trainset = args.Dataset.Aircraft(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.Aircraft(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'StanfordCar':
        trainset = args.Dataset.StanfordCar(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.StanfordCar(root=args.dataroot, train=False, index=class_index)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader

def get_base_dataloader_meta(args, group):
    class_index = np.arange(args.base_class)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'StanfordDog':
        trainset = args.Dataset.StanfordDog(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.StanfordDog(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'Aircraft':
        trainset = args.Dataset.Aircraft(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.Aircraft(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'StanfordCar':
        trainset = args.Dataset.StanfordCar(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.StanfordCar(root=args.dataroot, train=False, index=class_index)

    sampler = CategoriesSampler(trainset.targets, len(trainset.data) // args.batch_size_base, group*args.way,
                                args.episode_shot, group, args.way)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader(args, session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                    index_path=txt_path)
    if args.dataset == 'StanfordDog':
        trainset = args.Dataset.StanfordDog(root=args.dataroot, train=True,
                                             index_path=txt_path)

    if args.dataset == 'Aircraft':
        trainset = args.Dataset.Aircraft(root=args.dataroot, train=True,
                                             index_path=txt_path)

    if args.dataset == 'StanfordCar':
        trainset = args.Dataset.StanfordCar(root=args.dataroot, train=True,
                                             index_path=txt_path)
        
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'StanfordDog':
        testset = args.Dataset.StanfordDog(root=args.dataroot, train=False,
                                            index=class_new)
    if args.dataset == 'Aircraft':
        testset = args.Dataset.Aircraft(root=args.dataroot, train=False,
                                            index=class_new)

    if args.dataset == 'StanfordCar':
        testset = args.Dataset.StanfordCar(root=args.dataroot, train=False,
                                            index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list

