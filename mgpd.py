import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.nn import functional as F
from datetime import datetime
import os

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop
from clip.cocoop import get_cocoop
from clip.graph_adapter import get_graph_clip
from clip.graph_adapter_teacher import get_graph_clip_teacher
from clip.graph_adapter_cocoop import get_graph_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed, Logger
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'lr_1e-6'
# log_dir = os.path.join('/data/juices/TTA/TTA_log', timestamp)
# os.makedirs(log_dir)
# logger = Logger(log_dir=os.path.join(log_dir, 'log.txt'))
logger = Logger(log_dir = os.path.join('/model/zmsxxd/RN50', 'log.txt'))
#os.environ['CUDA_VISIBLE_DEVICES']= '0,1,2,3,4,5,6,7'

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def kl_div(stu_logits, tea_logits, temperature = 1.0):
    # print('stu_logits.size()',stu_logits.size())
    # print('tea_logits.size()',tea_logits.size())
    
    L_ukd = F.kl_div(
        F.log_softmax(stu_logits / temperature, dim=1),
        F.softmax(tea_logits / temperature, dim=1),
        reduction='sum',
    ) * (temperature * temperature) / stu_logits.numel()  # 求平均
    
    # loss = self.cfg.TRAINER.PROMPTKD.KD_WEIGHT * L_ukd

    return L_ukd

# 定义MSE损失函数
MSE_loss = nn.MSELoss()

#定义infoNCE_loss，但是batch_size为1，是不是不适用
def approx_infoNCE_loss(q, k):
    # 计算query和key的相似度得分
    similarity_scores = torch.matmul(q, k.t())  # 矩阵乘法计算相似度得分

    # 计算相似度得分的温度参数
    temperature = 0.07

    # 计算logits
    logits = similarity_scores / temperature

    # 构建labels（假设有N个样本）
    N = q.size(0)
    labels = torch.arange(N).to(logits.device)

    # 计算交叉熵损失
    loss = F.cross_entropy(logits, labels)
    
    return loss

def test_time_tuning(model, model_teacher, inputs, inputs_t, optimizer, scaler, args):
    #if args.cocoop:
        #image_feature, pgen_ctx, graph_embedding= inputs
        #pgen_ctx.requires_grad = True
        # optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    #    optimizer = torch.optim.AdamW([pgen_ctx], args.lr)

    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(f"Module with grad: {name}")
    selected_idx = None
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            output, graph_embedding = model(inputs) 
            output_teacher, graph_embedding_teacher = model_teacher(inputs_t)
                # output = model((image_feature, pgen_ctx))
                # output_teacher = model_teacher((teacher_image_feature, teacher_pgen_ctx))

            # if selected_idx is not None:
            #     output = output[selected_idx]
            #     output_teacher = output_teacher[selected_idx]
            # else:
            #     output, selected_idx = select_confident_samples(output, args.selection_p)
            #     output_teacher, selected_idx_teacher = select_confident_samples(output_teacher, args.selection_p)

            # loss = avg_entropy(output)
            loss_logits = kl_div(output,output_teacher)
            loss_graph = MSE_loss(graph_embedding, graph_embedding_teacher.detach())
            loss = loss_logits + loss_graph
            # loss.requires_grad_(True)
        
        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
    #if args.cocoop:
    #    return pgen_ctx

    return


def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    # print("Use GPU: {} for training".format(args.gpu))
    logger.info("Use GPU: {} for training".format(args.gpu))
    #device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes
    if args.cocoop:
        #model = get_graph_cocoop(args, args.arch, args.test_sets, args.gpu, args.n_ctx)
        model = get_graph_cocoop(args, args.arch, args.test_sets, args.gpu, args.n_ctx)
        model_teacher = get_graph_clip_teacher(args, args.arch_t, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)

        model_state = None
        model_state_teacher = None

    else:
        model = get_graph_clip(args, args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
        model_teacher = get_graph_clip_teacher(args, args.arch_t, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
        # if args.load is not None:
        #     print("Use pre-trained soft prompt (CoOp) as initialization")
        #     pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
        #     assert pretrained_ctx.size()[0] == args.n_ctx
        #     with torch.no_grad():
        #         model.prompt_learner[0].ctx.copy_(pretrained_ctx)
        #         model.prompt_learner[0].ctx_init_state = pretrained_ctx
        model_state = None

        # if args.load_t is not None:
        #     print("Use pre-trained soft prompt (CoOp) as initialization")
        #     pretrained_ctx_t = torch.load(args.load_t)['state_dict']['ctx']
        #     assert pretrained_ctx_t.size()[0] == args.n_ctx
        #     with torch.no_grad():
        #         model_teacher.prompt_learner[0].ctx.copy_(pretrained_ctx_t)
        #         model_teacher.prompt_learner[0].ctx_init_state = pretrained_ctx_t
        model_state_teacher = None
    #model = nn.DataParallel(model,device_ids=[0]).cuda()
    #model_teacher = nn.DataParallel(model_teacher,device_ids=[0]).cuda()

    for name, param in model.named_parameters():
        if not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        else:
            if "prompt_generator" not in name:
                param.requires_grad_(False)

    for name, param in model_teacher.named_parameters():
        if not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        else:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
    
    logger.info("=> Model created: visual backbone {}".format(args.arch))

    logger.info("=> Model teacher created: visual backbone {}".format(args.arch_t))
    
    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        model_teacher = model_teacher.cuda(args.gpu)

    # define optimizer
    if args.cocoop:
        optimizer = torch.optim.AdamW([
            {'params':model_teacher.graph_learner.parameters(),'lr':args.lr},
            {'params':model.prompt_generator.parameters()},
            ],lr = args.lr)
        optim_state = deepcopy(optimizer.state_dict())
    else:
        optimizer = torch.optim.AdamW([
            {'params':model_teacher.graph_learner.parameters(),'lr':args.lr},
            {'params':model.prompt_learner.parameters()},
            ],lr = args.lr)
        optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    logger.info('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    for set_id in datasets:
        if args.tpt:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=1, ###n_views为图像增强的数量，用于teacher模型
                                            augmix=len(set_id)>1)
            batchsize = 1
        else:
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.batch_size

        logger.info("evaluating: {}".format(set_id))
        # reset the model
        # Reset classnames of custom CLIP model
        if len(set_id) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all
        if args.cocoop:
            model.prompt_generator.reset_classnames(classnames, args.arch)
            model = model.cpu()
            model_state = model.state_dict()
            model = model.cuda(args.gpu)

            # model_teacher.prompt_generator.reset_classnames(classnames, args.arch)
            model_teacher = model_teacher.cpu()
            model_state_teacher = model_teacher.state_dict()
            model_teacher = model_teacher.cuda(args.gpu)
        else:
            model.reset_classnames(classnames, args.arch)
            # model_teacher.reset_classnames(classnames, args.arch)

        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        logger.info("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
            
        results[set_id] = test_time_adapt_eval(val_loader, model, model_teacher, model_state, model_state_teacher, optimizer, optim_state, scaler, args)
        del val_dataset, val_loader
        try:
            logger.info("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
            logger.info("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

    logger.info("======== Result Summary ========")
    logger.info("params: nstep	lr	bs")
    logger.info("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    logger.info("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    logger.info("\n")


def test_time_adapt_eval(val_loader, model, model_teacher, model_state, model_state_teacher, optimizer, optim_state, scaler, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    model_teacher.eval()

    if not args.cocoop: # no need to reset cocoop because it's fixed
        with torch.no_grad():
            model.reset()
            # model_teacher.reset()
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu,non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu,non_blocking=True)
            image = images
        target = target.cuda(args.gpu,non_blocking=True)
        if args.tpt:
            images = torch.cat(images, dim=0)
        # print('********************************',images.size())
        # print('********************************',images[0].size())
        # reset the tunable prompt to its initial state
        if not args.cocoop: # no need to reset cocoop because it's fixed
            if args.tta_steps > 0:
                with torch.no_grad():
                    model.reset()
                    # model_teacher.reset()
            optimizer.load_state_dict(optim_state)
            test_time_tuning(model, model_teacher, images[0].unsqueeze(0), images[1].unsqueeze(0), optimizer, scaler, args)
        else:
            #with torch.no_grad():
            #    with torch.cuda.amp.autocast():
            #        image_feature, pgen_ctx, graph_embedding = model.gen_ctx(images[0].unsqueeze(0), args.tpt)
            optimizer.load_state_dict(optim_state)
            test_time_tuning(model, model_teacher, images[0].unsqueeze(0), images[1].unsqueeze(0), optimizer, scaler, args)

        # The actual inference goes here
        #if args.tpt:
        #    if args.cocoop:
        #        image_feature = image_feature[0].unsqueeze(0)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if args.cocoop:
                    output, graph_embedding = model(images[0].unsqueeze(0))
                else:
                    output, graph_embedding = model(image)######这里的image是不是也要只取第一个
        # measure accuracy and record loss
        #acc1, acc5 = accuracy(output[0], target, topk=(1, 5))
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
                
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()

    return [top1.avg, top5.avg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--arch', metavar='ARCH', default='ViT-B/32')
    parser.add_argument('--arch_t', metavar='ARCH', default='ViT-L/14')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default="a photo of a", type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--load_t', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--BETA', type=float, default=0.8, help='the weight of graph feature fuse')

    main()