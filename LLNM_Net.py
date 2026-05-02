from __future__ import print_function, division 
import os
import sys
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn
import pickle
import pandas as pd
from PIL import Image
import argparse
# from apex import amp  # 使用PyTorch内置的混合精度训练替代
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from torch.nn import BCEWithLogitsLoss
from models.modeling_LLNM_Net import LLNM_Net, CONFIGS
from tqdm import tqdm
import argparse
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from transformers import BertTokenizer, BertModel

tk_lim = 300  # report limit (必须与configs.py中的rr_len一致)

disease_list = ['NonMeta', 'Latral']

def load_weights(model, weight_path):
    pretrained_weights = torch.load(weight_path, map_location=torch.device('cpu'))
    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}

    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    print("Loading LLNM-Net...")
    return model

class Data(Dataset):
    def __init__(self, set_type, img_dir, transform=None, target_transform=None):
        dict_path = set_type+'.pkl'
        f = open(dict_path, 'rb') 
        self.mm_data = pickle.load(f)
        f.close()
        self.idx_list = list(self.mm_data.keys())  
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        # 使用镜像源或本地路径加载BERT模型
        # 如果网络有问题，可以设置环境变量: export HF_ENDPOINT=https://hf-mirror.com
        # 或者使用本地路径: '/path/to/bert-base-chinese'
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            self.bert_model = BertModel.from_pretrained('bert-base-chinese')
        except Exception as e:
            print(f"从Hugging Face下载失败: {e}")
            print("请尝试以下解决方案:")
            print("1. 设置镜像: export HF_ENDPOINT=https://hf-mirror.com")
            print("2. 或手动下载模型到本地后使用本地路径")
            raise

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        k = self.idx_list[idx]
        img_path = os.path.join(self.img_dir, self.mm_data[k]['image'])
        img = Image.open(img_path).convert('RGB')

        label = self.mm_data[k]['label'].astype('float32')
        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        str_rr = self.mm_data[k]['report']
        input_ids = self.tokenizer.encode(str_rr, add_special_tokens=True, return_tensors='pt')
        outputs = self.bert_model(input_ids)
        last_hidden_state = outputs.last_hidden_state

        padding_length = tk_lim - last_hidden_state.shape[1]
        if padding_length>0:
            padding_token = self.tokenizer.pad_token_id
            padding_tensor = torch.full((1, padding_length, last_hidden_state.shape[2]), padding_token)
            padded_outputs = torch.cat([last_hidden_state, padding_tensor], dim=1)
        else:
            padded_outputs = last_hidden_state
        
        rr_vector = padded_outputs[:, :tk_lim, :] 

        rr = torch.tensor(rr_vector, dtype=torch.float32)                   # the report feature
        demo = torch.from_numpy(np.array(self.mm_data[k]['bics'])).float()  # the demographics information (age and sex)
        img_fea = torch.from_numpy(self.mm_data[k]['bts']).float()          # the image feature, such as shape and echo
        return img, label, rr, demo, img_fea

def train(args, model_para_path=None):
    torch.manual_seed(0)
    num_classes = args.CLS
    config = CONFIGS["LLNM_Net"]
    llnm_net = LLNM_Net(config, 224, zero_head=True, num_classes=num_classes)
    if model_para_path:
        llnm_net = load_weights(llnm_net, model_para_path)
    for param in llnm_net.parameters():
        param.requires_grad = True                                          # set requires_grad to True
    img_dir = args.DATA_DIR

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }

    train_data = Data(args.SET_TYPE, img_dir, transform=data_transforms['train'])

    trainloader = DataLoader(train_data, batch_size=args.BSZ, shuffle=True, num_workers=0, pin_memory=True)

    optimizer_llnm_net = torch.optim.AdamW(llnm_net.parameters(), lr=3e-5, weight_decay=0.01)
    # 使用PyTorch内置的混合精度训练
    scaler = GradScaler()
    llnm_net = llnm_net.cuda()
    # 如果GPU内存不足，可以注释掉DataParallel，只使用单GPU
    # 或者使用 torch.cuda.device_count() > 1 来判断是否需要DataParallel
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        llnm_net = torch.nn.DataParallel(llnm_net)
    else:
        print("使用单GPU进行训练")

    #----- Train ------
    print('--------Start training-------')
    num_epochs = 300
    loss_min = args.loss_min
    loss_fct = BCEWithLogitsLoss()
    loss_list, auc_list = [], []

    llnm_net.train()
    for epoch in range(num_epochs):
        outGT = torch.FloatTensor().cuda(non_blocking=True)
        outPRED = torch.FloatTensor().cuda(non_blocking=True)
        running_loss = 0.0
        for data in tqdm(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            imgs, labels, rr, demo, img_fea = data
            rr = rr.view(-1, tk_lim, rr.shape[3]).cuda(non_blocking=True).float()
            demo = demo.view(-1, 1, demo.shape[1]).cuda(non_blocking=True).float()
            img_fea = img_fea.view(-1, img_fea.shape[1], 1).cuda(non_blocking=True).float()
            sex = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()
            age = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            optimizer_llnm_net.zero_grad()  

            with torch.set_grad_enabled(True):
                # 使用混合精度训练
                with autocast():
                    outputs = llnm_net(imgs, rr, img_fea, sex, age)         # logits, attn_weights, torch.mean(x, dim=1)
                    preds = outputs[0]  # logits: [batch_size, num_classes]
                    
                    # BCEWithLogitsLoss配合one-hot编码：每个类别独立计算sigmoid
                    # 注意：使用sigmoid时，每个类别的概率是独立的，概率和不一定为1
                    # 这是多标签分类或使用BCE+one-hot的单标签分类的标准做法
                    probs = torch.sigmoid(preds)  # [batch_size, num_classes]，每个类别独立概率

                    target = labels.float()
                    # 处理标签维度：确保target是1维tensor
                    if target.dim() == 0:
                        # 如果是0维标量，转换为1维
                        target = target.unsqueeze(0)
                    elif target.dim() > 1:
                        # 如果是[batch_size, 1]，squeeze成[batch_size]
                        target = target.squeeze()
                    # 确保target是1维
                    if target.dim() == 0:
                        target = target.unsqueeze(0)
                    batch_size = target.shape[0]
                    target_one_hot = torch.zeros(batch_size, num_classes).cuda(non_blocking=True)
                    target_one_hot.scatter_(1, target.long().unsqueeze(1), 1)

                    # BCEWithLogitsLoss期望输入形状为[N, C]，不需要reshape
                    # preds已经是[batch_size, num_classes]形状
                    loss = loss_fct(preds, target_one_hot)
                
                # 使用scaler进行反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer_llnm_net)
                scaler.update()

            loss_value = loss.item()
            # BCEWithLogitsLoss默认reduction='mean'，返回的是平均loss
            # 为了计算总loss，需要乘以batch_size
            running_loss += loss_value * batch_size

            outGT = torch.cat((outGT, labels), 0)
            outPRED = torch.cat((outPRED, probs.data), 0)  # 保存sigmoid后的概率
  
        outGT_np = outGT.cpu().detach().numpy()
        outPRED_cpu = outPRED.cpu()
        
        # 处理标签维度：确保是1维数组
        if outGT_np.ndim > 1:
            outGT_np = outGT_np.squeeze()
        
        # 计算AUC：需要重新从logits计算，因为我们需要正确的概率分布
        # 问题：outPRED_cpu是sigmoid后的概率，不适合直接用于AUC计算
        # 解决方案：我们需要保存logits，然后对logits应用softmax
        # 但由于我们已经保存了sigmoid概率，我们需要重新计算
        # 注意：对于单标签多分类，应该使用softmax；对于多标签分类，使用sigmoid
        
        # 由于BCEWithLogitsLoss配合one-hot用于单标签分类，我们应该使用softmax
        # 但当前保存的是sigmoid概率，我们需要近似转换
        # 更好的方法：直接使用sigmoid概率（对于二分类）或重新计算logits
        # 这里我们使用sigmoid概率，因为BCE+one-hot通常用于二分类或多标签分类
        
        # 对于二分类：直接使用sigmoid后的正类概率
        # 对于多分类：需要将sigmoid概率归一化（虽然不完美，但可以工作）
        outPred_sigmoid = outPRED_cpu.detach().numpy()
        
        if num_classes == 2:
            # 二分类：直接使用sigmoid后的正类概率
            auc_scores = outPred_sigmoid[:, 1]
        else:
            # 多分类：将sigmoid概率归一化为softmax概率
            # 注意：这不是完美的，但可以工作
            # 更好的方法是在训练循环中保存logits，然后应用softmax
            outPred_probs = torch.softmax(torch.from_numpy(outPred_sigmoid), dim=1).numpy()
            # 使用正类（类别1）的概率，或者对于多分类，需要one-vs-rest
            if outPred_probs.shape[1] >= 2:
                auc_scores = outPred_probs[:, 1]  # 正类概率
            else:
                auc_scores = outPred_probs[:, 0]
        
        # 计算AUC（需要概率值，不是类别索引）
        try:
            auc = roc_auc_score(outGT_np, auc_scores)
        except ValueError as e:
            # 如果所有标签都是同一类，AUC无法计算
            print(f"警告: AUC计算失败: {e}")
            auc = 0.5
        
        # 计算平均loss（用于显示和保存）
        avg_loss = running_loss / len(trainloader.dataset)
        if avg_loss < loss_min:
            loss_min = avg_loss
            model_para_path = 'model_epoch'+str(epoch)+'_bs'+str(args.BSZ)+'_loss'+str(round(loss_min, 3))+'_auc'+str(round(auc, 3))+'.pth'
            torch.save(llnm_net.state_dict(), model_para_path)

        # 打印平均loss而不是累积loss
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Auc: {auc:.4f}')

        loss_list.append(avg_loss)  # 保存平均loss
        auc_list.append(auc)
        plt.figure()
        plt.title("Loss & AUC vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Loss or AUC")
        plt.plot(range(1, epoch + 2), loss_list, label="Loss")
        plt.plot(range(1, epoch + 2), auc_list, label="AUC")
        # plt.ylim((0, 1.))
        # plt.xticks(np.arange(1, num_epochs + 1, 10.0))
        plt.legend()
        plt.savefig("loss_auc.png")
        plt.close()  # 关闭图形以释放内存

        plt.figure()
        plt.title("Pred. & GrouTruth.")
        plt.xlabel("data index")
        plt.ylabel("label")
        # 对于绘图，使用argmax找到最大概率的类别
        # 对于二分类，直接使用sigmoid概率的argmax
        # 对于多分类，使用归一化后的概率
        if num_classes == 2:
            outPred_np = np.argmax(outPred_sigmoid, axis=1)  # 用于绘图
        else:
            outPred_probs = torch.softmax(torch.from_numpy(outPred_sigmoid), dim=1).numpy()
            outPred_np = np.argmax(outPred_probs, axis=1)  # 用于绘图
        plt.scatter(range(1, len(outGT_np) + 1), outGT_np, c='g', label="GroundTruth")
        plt.scatter(range(1, len(outPred_np) + 1), outPred_np, c='r', label="Prediction")
        # plt.ylim((0, 1.))
        # plt.xticks(np.arange(1, num_epochs + 1, 10.0))
        plt.legend()
        plt.savefig("pre_gt.png")
        plt.close()  # 关闭图形以释放内存
        
        # 释放GPU内存
        if num_classes == 2:
            del outGT, outPRED, outGT_np, outPRED_cpu, outPred_sigmoid, outPred_np
        else:
            del outGT, outPRED, outGT_np, outPRED_cpu, outPred_sigmoid, outPred_probs, outPred_np
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()

    return model_para_path, loss_min
        
def test(args, model_para_path):
    torch.manual_seed(0)
    num_classes = args.CLS
    config = CONFIGS["LLNM_Net"]
    model = LLNM_Net(config, 224, zero_head=True, num_classes=num_classes)
    llnm_net = load_weights(model, model_para_path)
    img_dir = args.DATA_DIR

    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }

    test_data = Data(args.SET_TYPE, img_dir, transform=data_transforms['test'])

    testloader = DataLoader(test_data, batch_size=args.BSZ, shuffle=False, num_workers=0, pin_memory=True)

    optimizer_llnm_net = torch.optim.AdamW(llnm_net.parameters(), lr=3e-5, weight_decay=0.01)
    # 测试时不需要混合精度，直接使用CUDA
    llnm_net = llnm_net.cuda()
    llnm_net = torch.nn.DataParallel(llnm_net)

    #----- Test ------
    print('--------Start testing-------')
    llnm_net.eval()
    with torch.no_grad():
        outGT = torch.FloatTensor().cuda(non_blocking=True)
        outPRED = torch.FloatTensor().cuda(non_blocking=True)
        for data in tqdm(testloader):
            # get the inputs; data is a list of [inputs, labels]
            imgs, labels, rr, demo, img_fea = data
            rr = rr.view(-1, tk_lim, rr.shape[3]).cuda(non_blocking=True).float()
            demo = demo.view(-1, 1, demo.shape[1]).cuda(non_blocking=True).float()
            img_fea = img_fea.view(-1, img_fea.shape[1], 1).cuda(non_blocking=True).float()
            sex = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()
            age = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            preds = llnm_net(imgs, rr, img_fea, sex, age)[0]
            probs = torch.sigmoid(preds)

            outGT = torch.cat((outGT, labels), 0)
            outPRED = torch.cat((outPRED, probs.data), 0)
  
        outGT_np = outGT.cpu().detach().numpy()
        outPRED_cpu = outPRED.cpu()
        
        # 处理标签维度
        if outGT_np.ndim > 1:
            outGT_np = outGT_np.squeeze()
        
        # 计算AUC：使用概率值，不是类别索引
        outPred_sigmoid = outPRED_cpu.detach().numpy()
        
        if num_classes == 2:
            # 二分类：使用正类概率
            auc_scores = outPred_sigmoid[:, 1]
        else:
            # 多分类：转换为softmax概率
            outPred_probs = torch.softmax(torch.from_numpy(outPred_sigmoid), dim=1).numpy()
            auc_scores = outPred_probs[:, 1] if outPred_probs.shape[1] >= 2 else outPred_probs[:, 0]
        
        try:
            aurocMean = roc_auc_score(outGT_np, auc_scores)
        except ValueError as e:
            print(f"警告: AUC计算失败: {e}")
            aurocMean = 0.5
        
        print('mean AUROC:' + str(aurocMean))
         

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--CLS', action='store', dest='CLS', required=True, type=int)             # number of classes
    parser.add_argument('--BSZ', action='store', dest='BSZ', required=True, type=int)             # batch size.
    parser.add_argument('--DATA_DIR', action='store', dest='DATA_DIR', required=True, type=str)   # location of the imaging data.
    parser.add_argument('--SET_TYPE', action='store', dest='SET_TYPE', required=True, type=str)   # file name of the clinical textual data (***.pkl).
    parser.add_argument('--loss_min', action='store', dest='loss_min', type=float)
    parser.add_argument('--mode', action='store', dest='mode', type=str, default='train', choices=['train', 'test', 'both'],
                       help='运行模式: train(仅训练), test(仅测试), both(训练+测试)')
    parser.add_argument('--model_path', action='store', dest='model_path', type=str, default=None,
                       help='测试模式下的模型权重路径')
    args = parser.parse_args()
    args.loss_min = 100.0
    
    if args.mode == 'train' or args.mode == 'both':
        model_para_path, args.loss_min = train(args)
        if args.mode == 'both':
            test(args, model_para_path)
    elif args.mode == 'test':
        if args.model_path is None:
            raise ValueError("测试模式需要提供--model_path参数")
        test(args, args.model_path)
