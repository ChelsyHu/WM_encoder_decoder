import torch
import argparse
from torch import distributed as dist

print(torch.cuda.device_count())  # 打印gpu数量
torch.distributed.init_process_group(backend="nccl")  # 并行训练初始化，建议'nccl'模式
print('world_size', torch.distributed.get_world_size()) # 打印当前进程数

# 下面这个参数需要加上，torch内部调用多进程时，会使用该参数，对每个gpu进程而言，其local_rank都是不同的；
parser.add_argument('--local_rank', default=-1, type=int)  
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)  # 设置gpu编号为local_rank;此句也可能看出local_rank的值是什么

def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


'''
多卡训练加载数据:
# Dataset的设计上与单gpu一致，但是DataLoader上不一样。首先解释下原因：多gpu训练是，我们希望
# 同一时刻在每个gpu上的数据是不一样的，这样相当于batch size扩大了N倍，因此起到了加速训练的作用。
# 在DataLoader时，如何做到每个gpu上的数据是不一样的，且gpu1上训练过的数据如何确保接下来不被别
# 的gou再次训练。这时候就得需要DistributedSampler。
# dataloader设置方式如下，注意shuffle与sampler是冲突的，并行训练需要设置sampler，此时务必
# 要把shuffle设为False。但是这里shuffle=False并不意味着数据就不会乱序了，而是乱序的方式交给
# sampler来控制，实质上数据仍是乱序的。
'''
train_sampler = torch.utils.data.distributed.DistributedSampler(My_Dataset)
dataloader = torch.utils.data.DataLoader(ds,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=16,
                                         pin_memory=True,
                                         drop_last=True,
                                         sampler=self.train_sampler)



'''
多卡训练的模型设置：
# 最主要的是find_unused_parameters和broadcast_buffers参数；
# find_unused_parameters：如果模型的输出有不需要进行反传的(比如部分参数被冻结/或者网络前传是动态的)，设置此参数为True;如果你的代码运行
# 后卡住某个地方不动，基本上就是该参数的问题。
# broadcast_buffers：设置为True时，在模型执行forward之前，gpu0会把buffer中的参数值全部覆盖
# 到别的gpu上。注意这和同步BN并不一样，同步BN应该使用SyncBatchNorm。
'''
My_model = My_model.cuda(args.local_rank)  # 将模型拷贝到每个gpu上.直接.cuda()也行，因为多进程时每个进程的device号是不一样的
My_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(My_model) # 设置多个gpu的BN同步
My_model = torch.nn.parallel.DistributedDataParallel(My_model, 
                                                     device_ids=[args.local_rank], 
                                                     output_device=args.local_rank, 
                                                     find_unused_parameters=False, 
                                                     broadcast_buffers=False)

'''开始多卡训练：'''
for epoch in range(200):
    train_sampler.set_epoch(epoch)  # 这句莫忘，否则相当于没有shuffle数据
    My_model.train()
    for idx, sample in enumerate(dataloader):
        inputs, targets = sample[0].cuda(local_rank, non_blocking=True), sample[1].cuda(local_rank, non_blocking=True)
        opt.zero_grad()
        output = My_model(inputs)
        loss = My_loss(output, targets)  # 
        loss.backward()
        opt.step()
        loss = reduce_mean(loss, dist.get_world_size())  # 多gpu的loss进行平均。


'''多卡测试(evaluation)：'''
if local_rank == 0:
    My_model.eval()
    with torch.no_grad():
        acc = My_eval(My_model)
    torch.save(My_model.module.state_dict(), model_save_path)
dist.barrier() # 这一句作用是：所有进程(gpu)上的代码都执行到这，才会执行该句下面的代码

'''
其它代码
'''