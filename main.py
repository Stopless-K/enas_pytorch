import os
import argparse
import torch
from controller.micro_controller import MicroController
from children.micro_child import EnasNet
import numpy as np

from ops import Lr_Scheduler
from data_provider import make_cifar10_dataloader, Provider
from config import Config
config = Config()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output', type=str, required=True)

    parser.add_argument('--num_nodes', type=int, default=5)
    parser.add_argument('--num_operations', type=int, default=5)

    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--lstm_num_layers', type=int, default=1)
    parser.add_argument('--controller_temperature', type=float, default=None)
    parser.add_argument('--controller_tanh_constant', type=float,
                        default=1.10)
    parser.add_argument('--controller_op_tanh_reduce', type=float,
                        default=2.5)
    parser.add_argument('--controller_entropy_weight', type=float, default=0.0001)
    parser.add_argument('--controller_bl_dec', type=float,
                        default=0.99)
    parser.add_argument('--controller_lr', type=float, default=0.0035)
    parser.add_argument('--controller_l2_reg', type=float, default=0)
    parser.add_argument('--controller_train_steps', type=int, default=30)
    parser.add_argument('--controller_train_every', type=int, default=1)    #train controller every # epoch


    parser.add_argument('--child_grad_bound', type=float,
                        default=5.0)  #child gradient clipping
    parser.add_argument('--child_opt', type=str,default='adam')
    parser.add_argument('--child_lr', type=float, default=0.1)
    parser.add_argument('--child_l2_reg', type=float, default=1e-4)
    parser.add_argument('--child_lr_decay_every', type=int, default=100)
    parser.add_argument('--child_decay_type', type=str, default='cosine')
    parser.add_argument('--child_lr_dec_rate', type=float, default=0.1)
    parser.add_argument('--child_lr_cosine', type=bool, default=True)
    parser.add_argument('--child_lr_t_0', type=int, default=10)
    parser.add_argument('--child_lr_t_mul', type=int, default=2)
    parser.add_argument('--child_lr_max', type=float, default=0.05)
    parser.add_argument('--child_lr_min', type=float, default=0.0005)


    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--eval_every_epoch', type=int, default=1)
    parser.add_argument('--save_every_epoch', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=160)




    return parser.parse_args()

def run():
    args = parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)


    #make dataset

    cifar_loader = make_cifar10_dataloader(batch_size= args.batch_size)
    cifar_train_loader = cifar_loader['train']
    cifar_test_loader = cifar_loader['test']
    cifar_provider = Provider(cifar_train_loader)

    #contorller

    controller_lr_scheduler = Lr_Scheduler(lr_dec_every= 1000000) #do not decay controller's learning rate

    controller = MicroController(
        args.hidden_size,
        args.hidden_size,
        args.num_nodes,
        args.num_operations,
        cuda=True,
        lstm_num_layers= args.lstm_num_layers,
        temperature= args.controller_temperature,
        tanh_constant=args.controller_tanh_constant,
        op_tanh_reduce = args.controller_op_tanh_reduce,
        entropy_weight= args.controller_entropy_weight,
        bl_dec= args.controller_bl_dec,
        lr_init= args.controller_lr,
        l2_reg= args.controller_l2_reg,
        lr_scheduler= controller_lr_scheduler
                                ).cuda()


    # child
    child_lr_scheduler= Lr_Scheduler(
        decay_type=args.child_decay_type,
        lr_dec_every= args.child_lr_dec_every,
        lr_dec_rate= args.child_lr_dec_rate,
        lr_cosine= args.child_lr_cosine,
        lr_max= args.child_lr_max,
        lr_min= args.child_lr_mim,
        lr_t_0= args.child_lr_t_0,
        lr_t_mul= args.child_lr_t_mul
        )

    child = EnasNet(
        cifar_provider,
        cifar_test_loader,
        config.in_channel,
        config.img_shape,
        lr_scheduler= child_lr_scheduler,
        num_nodes= args.num_nodes,
        num_classes= config.num_classes,
        num_operations=args.num_operations,
        grad_bound= args.grad_bound,
        opt_params = {
            'opt': args.child_optimizer,
            'lr': args.child_lr
        }
            ).cuda()

    #train

    print('start training.')

    step = 0
    while True:
        #train child
        epoch = child.provider.epoch
        arc, _, _ = controller()
        loss, acc = child.train_step(step, arc)


        if step % args.log_every == 0:
            #log info
            print('loss:{}, acc:{}'.format(loss,acc))

        if epoch % args.eval_every_epoch == 0:
            loss,acc = child.evaluate(arc)
            print('evaluate at step {}, loss:{}, acc:{}'.format(step, loss, acc))

            #start training controller
            if epoch % args.train_controller_every == 0:

                for i in range(args.controller_train_steps):
                    arc, log_prob, entroy = controller()
                    reward = child.evaluate(arc)
                    loss = controller.reinforce_train(i, reward, log_prob, entroy)

                    print('controller loss: {}'.format(loss))

                print('some architectures:')

                for _ in range(10):
                    arc, _, _ = controller()
                    acc = child.evaluate(arc)
                    print('-'*50)
                    print(arc)
                    print('val acc:{}'.format(acc))

        if epoch % args.save_every_epoch == 0:
            torch.save(controller.state_dict(), args.output+'controller_ckpt.pkl')
            torch.save(child.state_dict(),args.output+ 'child_ckpt.pkl')

        step+= 1






