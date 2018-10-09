import torch
from controller.micro_controller import MicroController
from children.micro_child import EnasBlock, Skeletion, EnasNet
import numpy as np

from ops import Lr_Scheduler
from data_provider import make_cifar10_dataloader, Provider

def run_test():

    controller = MicroController(8,8)
    # controller.cuda()
    max_iter = 10000
    for step in range(max_iter):
        actions, log_probs, entropy = controller()
        print(actions)
        reward = torch.cat(actions, -1).sum(-1, True)/ len(actions)
        print(reward)
        loss = controller.reinforce_train(step, reward.float(), log_probs, entropy)

        print(loss)

def test_block():
    block = EnasBlock([3,3], 16, 3, stride=2)
    x = torch.from_numpy(np.random.rand(5, 3, 8,8)).float()
    print(block)

    arc = [0,1,0, 1, 1, 2, 2,2, 2,3,3,4]
    out = block([x,x], arc)

    print(out.shape)

def test_skeletion():
    net = Skeletion(3, 3, num_classes= 10)
    x = torch.from_numpy(np.random.rand(5, 3, 32, 32)).float()

    arc = [0, 1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4]
    print(net(x, arc).shape)

def test_child():
    cifar_loader = make_cifar10_dataloader()
    cifar_train_loader = cifar_loader['train']
    cifar_test_loader = cifar_loader['test']

    cifar_provider = Provider(cifar_train_loader)



    lr_scheduler = Lr_Scheduler(lr_dec_rate=0.5)
    arc = [1, 1, 1, 2, 2, 2, 1, 2, 3, 3, 1, 2]

    enasnet = EnasNet(cifar_provider, cifar_test_loader, 3, (32,32), lr_scheduler,
                      channels=(16,16,32,32,64,64),
                      opt_params={'opt':'adam', 'lr':1e-3}
                      )

    max_iter = 10000
    for i in range(max_iter):
        loss, acc = enasnet.train_step(i, arc)

        if i % 100 == 0:

            print('loss:{0}, acc:{1}'.format(loss, acc))

            acc = enasnet.evaluate(arc)
            print('test acc: {}'.format(acc))


if __name__ == '__main__':
    # run_test()
    # test_block()
    # test_skeletion()
    test_child()