from copy import deepcopy

import torch.nn as nn
from base import Trainer

from .helper import *
from .Network import MYNET


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_up_model()
        self.set_save_path()

    def set_up_model(self):
        self.model = MYNET(self.args)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir is not None:
            logging.info('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir, 
                                              map_location={'cuda:3':'cuda:0'})['params']
        else:
            logging.info('random init params')
            if self.args.start_session > 0:
                logging.info('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_dataloader(self, session, group):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader_meta(self.args, group)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()
        # init train statistics
        result_list = [args]
        group = args.group
        rank = args.rank

        criterion = nn.NLLLoss().cuda()

        for session in range(args.start_session, args.sessions):
            train_set, trainloader, testloader = self.get_dataloader(session, group)
            self.model.load_state_dict(self.best_model_dict)
            if session == 0:  # load base class train img label
                train_set, trainloader_aux, testloader = get_base_dataloader(self.args)
                logging.info(f'new classes for this session:{np.unique(train_set.targets)}')
                optimizer, scheduler = get_optimizer(args, self.model)
                for epoch in range(args.epochs_base):
                    start_time = time.time()

                    tl, ta = base_train(self.model, trainloader, trainloader_aux, optimizer, scheduler, epoch, args,
                                        criterion, group, rank)
                    tsl, tsa = test(self.model, testloader, epoch, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        logging.info('********A better model is found!!**********')
                        logging.info('Saving model to :%s' % save_model_dir)

                    logging.info('best epoch {}, best test acc={:.3f}'.format(
                        self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))

                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]

                    logging.info(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                        '\n still need around %.2f mins to finish this session' % (
                                (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                # Finish base train
                logging.info('>>> Finish Base Train <<<')
                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    logging.info('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'
                    tsl, tsa = test(self.model, testloader, 0, args, session)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        logging.info('The new best test acc of base session={:.3f}'.format(
                            self.trlog['max_acc'][session]))

            # incremental learning sessions
            else:  
                logging.info("training session: [%d]" % session)
                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform

                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                tsl, avgac = test(self.model, testloader, 0, args, session)

                # update results and save model
                self.trlog['max_acc'][session] = float('%.3f' % (avgac * 100))
                self.best_model_dict = deepcopy(self.model.state_dict())

                logging.info(f"Avg Acc:{self.trlog['max_acc'][session]}")
                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        logging.info(f"Base Session Best epoch:{self.trlog['max_acc_epoch']}")
        logging.info('Total time used %.2f mins' % total_time)


    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + 'Cosine-Epo_%d-Lr_%.4f' % (
                self.args.epochs_base, self.args.lr_base)

        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
