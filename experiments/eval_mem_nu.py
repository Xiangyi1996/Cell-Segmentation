# -*- coding: utf-8 -*-
# @Time    : 2020-06-02 22:10
# @Author  : Xiangyi Zhang
# @File    : eval_mem_nu.py
# @Email   : zhangxy9@shanghaitech.edu.cn


from dataloaders.dataset import *
from experiments.parser import args
from networks.Unet import Unet
from torch import nn
import logging
import glob
import cv2
LOG = logging.getLogger('main')
import os.path as osp
import tifffile
from utils.src import *
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
from pylab import plt

'''
1: mem
2: mito
3: nu
4: gr
test_idx = ['784_5', '766_8', '842_17']
val_idx = ['783_5', '766_5', '842_12']
train_idx = ['766_2', '766_7', '766_10', '766_11', '769_5', '769_7', '783_6', '783_12', '784_4', '784_6', '784_7', '785_7', '822_4', '822_6', '822_7', '842_13', '931_11', '931_14']
'''

class cellmapping_test():
    def __init__(self, type='imageall_x'):
        self.type = type
        self.image_root_dir = osp.join('data/image_xyz/', self.type)
        self.checkpoint_dir = 'results/{}/checkpoints/best.pth'.format(args.exp)
        self.device = torch.device("cuda")
        self.gt_path = osp.join('data/mask_xyz')
        self.coarse_label_dir = 'data/coarse_label'

        with open('test/test_idx.txt', 'r') as f1, open('test/val_idx.txt', 'r') as f2, open('test/train_idx.txt', 'r') as f3, open('test/unlabel_idx.txt', 'r') as f4:
            self.test_idx = f1.read().splitlines()
            self.val_idx = f2.read().splitlines()
            self.train_idx = f3.read().splitlines()
            self.unlabel_idx = f4.read().splitlines()


    def save_tiff(self, pred, imagefile):
        savepath = f'data/mask_xyz/{self.type}_mem_nu_predtiff'
        os.makedirs(savepath, exist_ok=True)
        tifffile.imsave(f'{savepath}/{imagefile}.tiff', np.array(pred))
        print(f"save {savepath}/{imagefile}.tiff'")

    def evaluate_iou(self, x, y, num_class):
        """
        :param x: tensor (B, H, W) {0, 5} float
        :param y: tensor (B, 1, H, W) {0, 5} float
        :return: IOU: float
        """
        batch_size, h, w = x.size()
        x = x.reshape(batch_size, -1).long()
        y = y.reshape(batch_size, -1).long()
        acc = accuracy(x, y)
        IOU, Dice = IOU_dice(x, y, num_class)
        return IOU, acc, Dice

    def resize(self, pred, image_ori_path=None):
        label_ori = cv2.imread(image_ori_path, 0)
        pred_resize = cv2.resize(pred.astype(np.uint8), (label_ori.shape[1], label_ori.shape[0]), cv2.INTER_NEAREST)
        return pred_resize

    def get_path(self, image_root_dir):
        images_list = []
        if args.test_idx == 'all':
            for path in os.listdir(os.path.join(image_root_dir)):
                images_list.extend(glob.glob(os.path.join(image_root_dir, path, "*.png")))
        elif args.test_idx == 'iso':
            images_list.extend(glob.glob(os.path.join(image_root_dir, "*.png")))
        else:
            images_list.extend(glob.glob(os.path.join(image_root_dir, args.test_idx, "*.png")))
        images_list.sort()
        return images_list

    def turnGtBigLabel(self, imagefile):
        ##### Evaluate on membrane and nuclear
        ##### Set mito and nuclear as membrane(1), and nuclear(2)
        gt = tifffile.imread(osp.join(self.gt_path, imagefile +'_merged_4organelles_mask.tiff'))
        if args.mode == 'mito':
        ## Small label: mito
            gt[gt != 2] = 0
            gt[gt == 2] = 1
        ## Big label: Membrane and Nuclear
        else:
            gt[gt == 1] = 2
            gt[gt == 4] = 2
            gt[gt == 2] = 1
            gt[gt == 3] = 2
        return gt

    def turnPredMitoLabel(self, pred):
        pred[pred != 1] = 0
        return pred

    def evaluate(self, pred_list, gt):
        pred_list = torch.Tensor(pred_list).cuda()
        gt = torch.Tensor(gt).cuda()
        pred_list = pred_list[70:380, :,70:380]
        gt = gt[70:380, :,70:380]
        iou, acc, dice = self.evaluate_iou(pred_list, gt, num_class=args.num_classes + 1)
        return dice, iou, acc

    ### Inference ###
    def test(self):
        self.model = Unet(n_class=args.num_classes + 1, is_dropout=True)
        self.model = nn.DataParallel(self.model).cuda()
        self.model.load_state_dict(torch.load(self.checkpoint_dir))
        print('Resume from {}'.format(self.checkpoint_dir))
        for imagefile in self.test_idx:
            test_loader = get_data_un(args=args, data_root_dir=osp.join(self.image_root_dir, imagefile))
            self.images_list = self.get_path(osp.join(self.image_root_dir, imagefile))
            self.model.eval()
            self.step(test_loader, imagefile=imagefile)

    def step(self, dataloader, imagefile=None):
        pred_list = []
        self.imagefile= imagefile
        for idx, sample in enumerate(dataloader):
            images = sample["image"]
            images = images.to(self.device)
            outputs = self.model(images)
            pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            if args.mode == 'mito':
                pred = self.turnPredMitoLabel(pred) #turn pred mito 0,1,2 to 0,1
            pred = pred.cpu().numpy()
            for i in range(images.shape[0]):
                image_ori_path = self.images_list[idx * args.batch_size + i]
                pred = self.resize(pred[i].squeeze(), image_ori_path=image_ori_path)
                pred_list.append(pred)
        pred_list = np.array(pred_list)

        if self.type == 'imageall_x':
            #### We only have manual label in x view, so we evaluate here. Otherwise we save the tiff
            #### from three views to do postprocessing.
            self.gt = self.turnGtBigLabel(imagefile)
            dice, iou, acc = self.evaluate(pred_list, self.gt)
            # pred_list = pred_list.cpu().numpy()
            if args.mode == 'mito':
                print("{}, mode: {}\n"
                      "mito: DICE: {:.2%} | IOU: {:.2%}".format(imagefile, self.type, dice[0], iou[0]))
            else:
                print("{}, mode: {}\n"
                      "mem: DICE: {:.2%} | IOU: {:.2%}\n"
                      "nu: DICE: {:.2%} | IOU: {:.2%}".format(imagefile, self.type, dice[0], iou[0],dice[1], iou[1]))
            print('-------------------------------')

        self.save_tiff(pred_list, imagefile)

class Post_process(cellmapping_test):
    def __init__(self):
        super(Post_process, self).__init__(type='imageall_x')
        self.path = f'data/mask_xyz/'

    def getIsolateLabel(self, x, label):
        x_copy = x.copy()
        x_copy[x_copy != label] = 255
        x_copy[x_copy == label] = 1
        x_copy[x_copy == 255] = 0
        return x_copy

    def fix012(self, xyzLabel, xyz, x, y):
        if len(x[xyz==6] == 2) !=  len(x[xyz==6]) : #if 012 appear
            print('012 appear !')

    # def save_tiff_coarse(self, pred, imagefile):
    #     savepath = f'/group/xiangyi/iHuman-SIST/imageall_xyz_mask/fuse3D_mem_nu_coarse_predtiff_0717fixflip'
    #     make_dir(savepath)
    #     tifffile.imsave(f'{savepath}/{imagefile}.tiff', np.array(pred))
    #     print(f"save {savepath}/{imagefile}.tiff'")

    def cropCoarseLabel(self, label, imagefile):
        slice = label.shape[0]
        if glob.glob(osp.join(self.coarse_label_dir, imagefile, '*.png')) == []:
            print('did not have coarse label')
            return label
        mask = plt.imread(glob.glob(osp.join(self.coarse_label_dir, imagefile, '*.png'))[0], 0)
        maskrepeat = np.repeat(mask[None, ...], slice, axis=0)
        label[maskrepeat == 0] = 0
        return label

    def post_fusexyz(self):
        for imagefile in self.test_idx:
            self.imagefile = imagefile
            print(self.imagefile)
            x = tifffile.imread(osp.join(self.path, f'imageall_x_{args.mode}_predtiff', self.imagefile + '.tiff'))
            y = tifffile.imread(osp.join(self.path, f'imageall_y_{args.mode}_predtiff', self.imagefile + '.tiff'))
            z = tifffile.imread(osp.join(self.path, f'imageall_z_{args.mode}_predtiff', self.imagefile + '.tiff'))
            turned_y = np.rot90(y, k=1, axes=(2, 0))  ## y
            turned_z = np.rot90(z, k=3, axes=(1, 0))  ## z
            assert x.shape == turned_y.shape and x.shape == turned_z.shape

            xyzLabel = np.zeros_like(x)
            for p in [0, 1, 2]:
                xyz = x + turned_y + turned_z
                xi = self.getIsolateLabel(x, p)  # x[x==p] = 1 x[x!=p] = 0
                yi = self.getIsolateLabel(turned_y, p)
                zi = self.getIsolateLabel(turned_z, p)
                xyzi = xi + yi + zi
                # only more than two location has this prediction,
                xyzLabel[xyzi >= 2] = p

            # If 012 case appears
            self.fix012(xyzLabel, xyz, x, turned_y)

            ############# Before Coarse Label #############
            if self.imagefile in self.test_idx:
                self.evaluate_again(xyzLabel, 'before')

            ## Optional: using the coarse label to refine the results. You can ignore this code if you don't want to use the extra labels.
            ## If you want to refine, the labels are in https://drive.google.com/drive/folders/12msPtKcQ7IPG7HD9OJZf5c5ZBntEiN1q?usp=sharing

            # self.cropCoarseLabel(xyzLabel, self.imagefile)
            # ############# After #############
            # if self.imagefile in self.test_idx:
            #     self.evaluate_again(xyzLabel, 'after')
            self.save_tiff_coarse(xyzLabel, self.imagefile)

    def evaluate_again(self, xyzLabel, flag='before'):
        gt = self.turnGtBigLabel(self.imagefile)
        dice, iou, _ = self.evaluate(xyzLabel, gt)
        if args.mode == 'mito':
            print("
            # {} coarse label refine \n"
                  "mito: DICE: {:.2%} | IOU: {:.2%}".format(flag, dice[0], iou[0]))
        else:
            print("
            # {} coarse label refine \n"
                  "mem: DICE: {:.2%} | IOU: {:.2%}\n"
                  "nu:  DICE: {:.2%} | IOU: {:.2%}".format(flag, dice[0], iou[0], dice[1], iou[1]))
        print('------------------------------- \n')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for type in ['imageall_x', 'imageall_y', 'imageall_z']:
        print('#############', type, '###############')
        Test = cellmapping_test(type=type)
        Test.test()
    post_process = Post_process()
    post_process.post_fusexyz()

