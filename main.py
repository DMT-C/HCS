from model.net import Encoder, Decoder, Discriminator, COAE
from utils.loss import l2_distance
import argparse
from utils.data import Data_Loader, to_normalization
import torch
from torch.utils.data import Dataset
import torchvision.transforms as Transforms
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import cv2
import os
from utils.Evaluation import Evaluation
from skimage.filters.thresholding import threshold_otsu
from sklearn import metrics
import scipy.io as io
from skimage.transform import resize
from model.poincareEncoder import PoincareEncoder,PoincareDecoder
from utils.loss import l2_regularization,spectral_consistency_loss,complementarity_loss
from geoopt.optim import RiemannianAdam
from geoopt import PoincareBall
import time
# bastrop dataset:      nc1 = 11, nc2 = 3
# california dataset:   nc1 = 7, nc2 = 10
# shuguang dataset:     nc1 = 1, nc2 = 3
# italy dataset:        nc1 = 1, nc2 = 3
# france dataset:       nc1 = 3, nc2 = 3
# yellow dataset:       nc1 = 1, nc2 = 1
# gloucester2 dataset:  nc1 = 3, nc2 = 1
# gloucester1 dataset:  nc1 = 1, nc2 = 3

# Select device，w/ cuda，w/o cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='italy', type=str)  # yellow or bastrop or california or other
# parser.add_argument('--t1_path', default='./data/Yellow/t1.png', type=str)
# parser.add_argument('--t2_path', default='./data/Yellow/t2.png', type=str)
# parser.add_argument('--gt_path', default='./data/Yellow/gt.png', type=str)
# parser.add_argument('--t1_path', default='./data/France/Img7-Ac.png', type=str)
# parser.add_argument('--t2_path', default='./data/France/Img7-Bc.png', type=str)
# parser.add_argument('--gt_path', default='./data/France/Img7-C.png', type=str)
# parser.add_argument('--t1_path', default='./data/California/California.mat', type=str)
# parser.add_argument('--t1_path', default='./data/Bastrop/Cross-sensor-Bastrop-data.mat', type=str)
parser.add_argument('--t1_path', default='./data/Italy/Italy_1.bmp', type=str)
parser.add_argument('--t2_path', default='./data/Italy/Italy_2.bmp', type=str)
parser.add_argument('--gt_path', default='./data/Italy/Italy_gt.png', type=str)
# parser.add_argument('--t1_path', default='./data/Shuguang/T1.png', type=str)
# parser.add_argument('--t2_path', default='./data/Shuguang/T2.png', type=str)
# parser.add_argument('--gt_path', default='./data/Shuguang/GT.png', type=str)
# parser.add_argument('--t1_path', default='./data/Gloucester2/T1-Img17-Bc.png', type=str)
# parser.add_argument('--t2_path', default='./data/Gloucester2/T2-Img17-A.png', type=str)
# parser.add_argument('--gt_path', default='./data/Gloucester2/Img17-C.png', type=str)
# parser.add_argument('--t1_path', default='./data/HalfShuguang/T1_left.png', type=str)
# parser.add_argument('--t2_path', default='./data/HalfShuguang/T2_left.png', type=str)
# parser.add_argument('--gt_path', default='./data/HalfShuguang/GT_left.png', type=str)
parser.add_argument('--t1_nc', default='3', type=int)
parser.add_argument('--t2_nc', default='3', type=int)
parser.add_argument('--patch_size', default=9, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--test_ps', default=9, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--lr_delay', default=range(30), type=float)
parser.add_argument('--optim', default='rmsprop', type=str)  # rmsprop
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--vision_path', default='./MyItaly_0.0001_9_1_rms/', type=str)


parser.add_argument('--hyp_c', type=float,default=1,help='hyperbolic c, 0 enables sphere mode')
parser.add_argument('--kernel_size', type=int,default=3,help='scan kernel size')
args = parser.parse_args()

encoder = PoincareEncoder(in_channels=3,out_channels=32, c=args.hyp_c,kernel_size=args.kernel_size)
decoder1=PoincareDecoder(in_channels=32,out_channels=3, c=args.hyp_c,kernel_size=args.kernel_size)
decoder2=PoincareDecoder(in_channels=32,out_channels=3, c=args.hyp_c,kernel_size=args.kernel_size)


encoder.to(device=device)
#encoder2.to(device=device)
decoder1.to(device=device)
decoder2.to(device=device)

trans = Transforms.Compose([Transforms.ToTensor()])

train_dataset = Data_Loader(data_name=args.data_name, t1_path=args.t1_path,
                            t2_path=args.t2_path, gt_path=args.gt_path,
                            patch_size=args.patch_size, mode='train',
                            transform=Transforms.ToTensor())

test_dataset = Data_Loader(data_name=args.data_name, t1_path=args.t1_path,
                           t2_path=args.t2_path, gt_path=args.gt_path,
                           patch_size=args.test_ps, mode='test',
                           transform=Transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size*6 ,
                                          shuffle=False)

if args.data_name == 'bastrop':
    mat = io.loadmat(args.t1_path)
    x1 = mat['t1_L5'][:, :, 3] # landsat-5: NIR band
    x2 = mat["t2_ALI"][:, :, 5] # OE-AIL: NIR band
    x1 = to_normalization(x1)
    x2 = to_normalization(x2)
    x1 = x1[..., np.newaxis]
    x2 = x2[..., np.newaxis]
    gt2 = mat["ROI_1"]
    gt = gt2 * 255
    o_h, o_w = gt.shape[0], gt.shape[1]


    cv2.imwrite(args.vision_path + "t1.png", x1*255)
    cv2.imwrite(args.vision_path + "t2.png", x2*255)
elif args.data_name == 'california_mat':
    mat = io.loadmat(args.t1_path)
    x1 = mat['logt2_clipped'][:, :, 0] # SAR
    x2 = mat["t1_L8_clipped"][:, :, 3]+1 # landsat-8: NIR band
    x1 = to_normalization(x1)
    # x2 = to_normalization(x2)
    x1 = x1[..., np.newaxis]
    x2 = x2[..., np.newaxis]
    gt2 = mat["ROI"]
    gt = gt2 * 255
    o_h, o_w = gt.shape[0], gt.shape[1]
    # cv2.imwrite(args.vision_path + "gt.png", gt2 * 255)
    cv2.imwrite(args.vision_path + "t1.png", x1*255)
    cv2.imwrite(args.vision_path + "t2.png", x2*255)
elif args.data_name == 'california':
    x1 = cv2.imread(args.t1_path)
    x2 = cv2.imread(args.t2_path)
    gt = cv2.imread(args.gt_path)[:, :, 0] # 0-255
    o_h, o_w = gt.shape[0], gt.shape[1]
    gt2 = gt / 255 # 0-1

    cv2.imwrite(args.vision_path + "image_t1.png", x1)
    cv2.imwrite(args.vision_path + "image_t2.png", x2)
elif args.data_name == 'gloucester1':
    x1 = cv2.imread(args.t1_path)
    x2 = cv2.imread(args.t2_path)
    gt = cv2.imread(args.gt_path)[:, :, 0] # 0-255
    o_h, o_w = gt.shape[0], gt.shape[1]
    gt2 = gt / 255 # 0-1


    cv2.imwrite(args.vision_path + "t1.png", x1)
    cv2.imwrite(args.vision_path + "t2.png", x2)
    # cv2.imwrite(args.vision_path + "gt.png", gt)
elif args.data_name == 'yellow':
    x1 = cv2.imread(args.t1_path)[:, :, 0]
    x2 = cv2.imread(args.t2_path)[:, :, 0]
    gt = cv2.imread(args.gt_path)[:, :, 0]  # 0-255
    o_h, o_w = gt.shape[0], gt.shape[1]
    gt2 = gt / 255  # 0-1

    cv2.imwrite(args.vision_path + "t1.png", x1)
    cv2.imwrite(args.vision_path + "t2.png", x2)
else:
    x1 = cv2.imread(args.t1_path)
    x2 = cv2.imread(args.t2_path)
    gt = cv2.imread(args.gt_path)[:, :, 0]  # 0-255
    o_h, o_w = gt.shape[0], gt.shape[1]
    gt2 = gt / 255  # 0-1

    cv2.imwrite(args.vision_path + "t1.png", x1)
    cv2.imwrite(args.vision_path + "t2.png", x2)

ps = args.patch_size
t1_expand = cv2.copyMakeBorder(x1, ps // 2, ps // 2, ps // 2, ps // 2, cv2.BORDER_CONSTANT, 0)  # cv2.BORDER_DEFAULT
t2_expand = cv2.copyMakeBorder(x2, ps // 2, ps // 2, ps // 2, ps // 2, cv2.BORDER_CONSTANT, 0)
h, w = t1_expand.shape[0], t1_expand.shape[1]

def train():
   
    opt_encoder = optim.RMSprop(encoder.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.5)
    opt_decoder1 = optim.RMSprop(decoder1.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.5)
    opt_decoder2 = optim.RMSprop(decoder2.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.5)# france0.9其他0.5

    scheduler_encoder = torch.optim.lr_scheduler.MultiStepLR(opt_encoder, milestones=args.lr_delay, gamma=0.5)
    scheduler_decoder1 = torch.optim.lr_scheduler.MultiStepLR(opt_decoder1, milestones=args.lr_delay, gamma=0.5)
    scheduler_decoder2 = torch.optim.lr_scheduler.MultiStepLR(opt_decoder2, milestones=args.lr_delay, gamma=0.5)


    MSE_loss = nn.MSELoss(reduction='mean').cuda()

    
    
    for epoch in range(args.epochs):
        ################# Training Generator #####################################################################################
        with tqdm(total=len(train_loader), desc='G Train Epoch #{}'.format(epoch + 1), ncols=190) as t:
            start = time.time()
            for batch_idx, (t1, t2, _) in tqdm(enumerate(train_loader)):

                encoder.train()
                #encoder2.train()
                decoder1.train()
                decoder2.train()

                opt_encoder.zero_grad()

                opt_decoder1.zero_grad()
                opt_decoder2.zero_grad()


                t1 = t1.to(device=device).float()
                t2 = t2.to(device=device).float()

                e_t1, e_t2,f1,f2,h1,h2=encoder(t1,t2)
                t1_hat = decoder1(e_t1)
                t2_hat = decoder2(e_t2)
                t12_hat = decoder2(e_t1)  
                t21_hat = decoder1(e_t2)

                e_t1_hat, e_t2_hat,ff1,ff2,hh1,hh2= encoder(t12_hat,t21_hat)
 
                spectral_loss = MSE_loss(f1,f2)*0.01
                complement_loss=MSE_loss(h1,h2)*0.01
 
                loss_const_t1 = MSE_loss(t1, t1_hat)  # t1 regress
                loss_const_t2 = MSE_loss(t2, t2_hat)  # t2 regress
                loss_commonality_1_b = MSE_loss(e_t2, e_t1)
                loss_commonality_2_b = MSE_loss(e_t1_hat, e_t2_hat)
 
                l2reg_e=l2_regularization(encoder)
                l2reg_d1= l2_regularization(decoder1)
                l2reg_d2= l2_regularization(decoder2)
                l2=(l2reg_e+l2reg_d1+l2reg_d2)

                generator_loss = (loss_const_t1 + loss_const_t2+ loss_commonality_1_b +loss_commonality_2_b+l2+spectral_loss+complement_loss) # loss ablation

                generator_loss.backward()
                opt_encoder.step()
                opt_decoder1.step()
                opt_decoder2.step()
                


                t.set_postfix({
                               'G_loss': '%.5f' % (generator_loss.item()),
                               'G_Const_t1': '%.5f' % (loss_const_t1.item()),
                               'G_Const_t2': '%.5f' % (loss_const_t2.item()),
                               'G_1_Common': '%.5f' % (loss_commonality_1_b.item()),
                               'G_2_Common': '%.5f' % (loss_commonality_2_b.item()),
                               'G_l2reg': '%.5f' % (l2.item()),

                               })
                t.update(1)
            end = time.time()
            print('EEEEFFFF@Epoch: %d, Time: %.2f' % (epoch + 1, end - start))
        scheduler_encoder.step()
        scheduler_decoder1.step()
        scheduler_decoder2.step()


        acc = val(args.patch_size, args.test_ps, epoch)
        print("\n")

        if acc:
            if args.data_name == 'france' or args.data_name=='shuguang':# or args.data_name == 'gloucester1':
                print('No output visual regress image\n')
            else:
                encoder.eval()

                decoder1.eval()
                decoder2.eval()

                t1_tensor = trans(t1_expand)
                t2_tensor = trans(t2_expand)
                t1_tensor = t1_tensor.unsqueeze(0)
                t2_tensor = t2_tensor.unsqueeze(0)
                t1_tensor = t1_tensor.to(device=device).float()
                t2_tensor = t2_tensor.to(device=device).float()
                f_t1, f_t2,_,_,_,_ = encoder(t1_tensor, t2_tensor)
                t1_hat = decoder1(f_t1)
                t2_hat = decoder2(f_t2)
                

                if t1_hat.shape[1] == 3:
                    t1_hat = t1_hat.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
                    t2_hat = t2_hat.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
                elif t1_hat.shape[1] == 1:
                    t1_hat = t1_hat.cpu().detach().numpy().squeeze()
                    t2_hat = t2_hat.cpu().detach().numpy().squeeze()
                    t1_hat = to_normalization(t1_hat) * 255
                t2_hat = to_normalization(t2_hat) * 255

                t1_hat = t1_hat[(ps // 2): h - (ps // 2), (ps // 2): w - (ps // 2)]
                t2_hat = t2_hat[(ps // 2): h - (ps // 2), (ps // 2): w - (ps // 2)]


                cv2.imwrite((args.vision_path + str(epoch + 1) + "/t1_hat.png"), t1_hat.astype(np.uint8))
                cv2.imwrite(args.vision_path + str(epoch + 1) + "/t2_hat.png", t2_hat.astype(np.uint8))
                # cv2.imwrite(args.vision_path + str(epoch + 1) + "/t12.png", f_t12_hat)
                # cv2.imwrite(args.vision_path + str(epoch + 1) + "/t21.png", f_t21_hat)#通道数为2不能保存
            torch.save(encoder.state_dict(),
                       args.vision_path + str(epoch + 1) + '/encoder_ps_' + str(args.patch_size) + '_epoch_' + str(
                           epoch) + '.pth')
            # torch.save(encoder2.state_dict(),
            #            args.vision_path + str(epoch + 1) + '/encoder_ps_' + str(args.patch_size) + '_epoch_' + str(
            #                epoch) + '.pth')
            torch.save(decoder1.state_dict(),
                       args.vision_path + str(epoch + 1) + '/decoder1_ps_' + str(args.patch_size) + '_epoch_' + str(
                           epoch) + '.pth')
            torch.save(decoder2.state_dict(),
                       args.vision_path + str(epoch + 1) + '/decoder2_ps_' + str(args.patch_size) + '_epoch_' + str(
                           epoch) + '.pth')


def val(patch_size, test_ps, epoch):
    encoder.eval()

    decoder1.eval()
    decoder2.eval()

    with torch.no_grad():
        res1 = []
        res2 = []
        Gres = []
        softmax = nn.Softmax(dim=1)
        with tqdm(total=len(test_loader), desc='Test Epoch #{}'.format(epoch + 1), ncols=170, colour='cyan') as t:
            start = time.time()
            for batch_idx, (t1, t2, _) in tqdm(enumerate(test_loader)):
                bat = t1.shape[0]
                t1 = t1.to(device=device).float()
                t2 = t2.to(device=device).float()


                f_t1, f_t2,_,_,_,_ = encoder(t1, t2)
   
                t1_hat = decoder1(f_t1)
                t2_hat = decoder2(f_t2)
                t21_hat = decoder1(f_t2)
                t12_hat = decoder2(f_t1)

                f_t1_hat, f_t2_hat,_,_,_,_ = encoder(t21_hat, t12_hat)


    
                D1_diff = l2_distance(f_t1.contiguous().view(bat, -1), f_t2.contiguous().view(bat, -1))
                D2_diff = l2_distance(f_t1_hat.contiguous().view(bat, -1), f_t2_hat.contiguous().view(bat, -1))

                diff_sum = D1_diff + D2_diff

    
                D1_diff = D1_diff.detach().cpu().numpy()
                D2_diff = D2_diff.detach().cpu().numpy()
                diff_sum = diff_sum.detach().cpu().numpy()
                for i in range(bat):
                    res1.append(D1_diff[i])
                    res2.append(D2_diff[i])
                    Gres.append(diff_sum[i])
                
                t.update(1)
            end = time.time()
            print('EEE###$$$Test Time: %.2f' % (end - start))

    min1 = np.min(np.array(res1))
    max1 = np.max(np.array(res1))
    chageres1 = (np.array(res1) - min1) / (max1 - min1)


    min2 = np.min(np.array(res2))
    max2 = np.max(np.array(res2))
    chageres2 = (np.array(res2) - min2) / (max2 - min2)


    Gmin = np.min(np.array(Gres))
    Gmax = np.max(np.array(Gres))
    Gchageres = (np.array(Gres) - Gmin) / (Gmax - Gmin)


    chageres1 = chageres1.reshape(o_h, o_w)
    chageres2 = chageres2.reshape(o_h, o_w)
    Gchageres = Gchageres.reshape(o_h, o_w)


    FPR_1, TPR_1, thres_1 = metrics.roc_curve(gt2.flatten(), chageres1.flatten())
    FPR_2, TPR_2, thres_2 = metrics.roc_curve(gt2.flatten(), chageres2.flatten())
    FPR_3, TPR_3, thres_3 = metrics.roc_curve(gt2.flatten(), Gchageres.flatten())
    AUC1 = metrics.auc(FPR_1, TPR_1)
    AUC2 = metrics.auc(FPR_2, TPR_2)
    AUC3 = metrics.auc(FPR_3, TPR_3)

    chageres1 = chageres1 * 255
    chageres2 = chageres2 * 255
    Gchageres = Gchageres * 255
    ##########################################################
    thre1 = threshold_otsu(chageres1)
    thre2 = threshold_otsu(chageres2)
    thre3 = threshold_otsu(Gchageres)

    CM1 = (chageres1 > thre1) * 255
    CM2 = (chageres2 > thre2) * 255
    CM3 = (Gchageres > thre3) * 255



    Indicators1 = Evaluation(gt, CM1)
    OA1, kappa1, AA1 = Indicators1.Classification_indicators()
    P1, R1, F11 = Indicators1.ObjectExtract_indicators()
    TP1, TN1, FP1, FN1 = Indicators1.matrix()

    Indicators2 = Evaluation(gt, CM2)
    OA2, kappa2, AA2 = Indicators2.Classification_indicators()
    P2, R2, F12 = Indicators2.ObjectExtract_indicators()
    TP2, TN2, FP2, FN2 = Indicators2.matrix()

    Indicators3 = Evaluation(gt, CM3)
    OA3, kappa3, AA3 = Indicators3.Classification_indicators()
    P3, R3, F13 = Indicators3.ObjectExtract_indicators()
    TP3, TN3, FP3, FN3 = Indicators3.matrix()

    print('AUC_1={}, OA_1={}, kappa_1={} || AUC_2={}, OA_2={}, kappa_2={} || AUC_3={}, OA_3={}, kappa_3={}'.format(AUC1,
                                                                                                                   OA1,
                                                                                                                   kappa1,
                                                                                                                   AUC2,
                                                                                                                   OA2,
                                                                                                                   kappa2,
                                                                                                                   AUC3,
                                                                                                                   OA3,
                                                                                                                    kappa3))


    if True :
        path = args.vision_path + '{}'.format(epoch + 1)
        if not os.path.exists(path):
            os.makedirs(path)
        val_acc = open(args.vision_path + str(epoch + 1) + '/val_acc.txt', 'a')
        val_acc.write('============================================================================\n')
        val_acc.write('Dual Domain:\n')
        val_acc.write('TP={} || TN={} || FP={} || FN={}\n'.format(TP3, TN3, FP3, FN3))
        val_acc.write("\"AUC\":\"" + "{}\"\n".format(AUC3))
        val_acc.write("\"OA\":\"" + "{}\"\n".format(OA3))
        val_acc.write("\"Kappa\":\"" + "{}\"\n".format(kappa3))
        val_acc.write("\"AA\":\"" + "{}\"\n".format(AA3))
        val_acc.write("\"Precision\":\"" + "{}\"\n".format(P3))
        val_acc.write("\"Recall\":\"" + "{}\"\n".format(R3))
        val_acc.write("\"F1\":\"" + "{}\"\n".format(F13))
        val_acc.close()        
        cv2.imwrite(args.vision_path + str(epoch + 1) + '/Tr_ps' + str(patch_size) + '_Te_ps' + str(
            test_ps) + '_DualD_diff_' + str(epoch + 1) + '.png', Gchageres.astype(np.uint8))

        cv2.imwrite(args.vision_path + str(epoch + 1) + '/Tr_ps' + str(patch_size) + '_Te_ps' + str(
            test_ps) + '_DualD_CM_' + str(epoch + 1) + '.png', CM3.astype(np.uint8))

        return True




if __name__ == "__main__":
    train()
    # test()
