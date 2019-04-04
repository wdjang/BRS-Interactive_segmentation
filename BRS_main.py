import numpy as np
from collections import OrderedDict
from InputCorrection import *
from generate_click import *
import caffe
import cv2


class constraint(object):
    def __init__(self, loss_functions, parameter_lists):
        self.loss_functions = loss_functions
        self.parameter_lists = parameter_lists

def create_test_batch(img_size, net_size):
    y_list = []
    x_list = []
    interval_len = 400
    for y_id in range(0, int(img_size[0]), interval_len):
        start_id = int(min(y_id, img_size[0]-net_size[0]))
        end_id = int(min(y_id+net_size[0], img_size[0]))
        border_len = 0
        if start_id < 0:
            start_id = 0
            border_len = net_size[0]-end_id
        y_list.append([start_id, end_id, border_len])
        if end_id == img_size[0]:
            break
    for x_id in range(0, int(img_size[1]), interval_len):
        start_id = int(min(x_id, img_size[1]-net_size[1]))
        end_id = int(min(x_id+net_size[1], img_size[1]))
        border_len = 0
        if start_id < 0:
            start_id = 0
            border_len = net_size[1]-end_id
        x_list.append([start_id, end_id, border_len])
        if end_id == img_size[1]:
            break

    return y_list, x_list


def main(im_path, gt_path):

    net_size = [480, 480]
    pred_thold = 0.5
    whole_w = 1.0
    max_iact = float(255.0)

    net_weight = './model/BRS_DenseNet.caffemodel'
    net_model = './model/deploy.prototxt'

    caffe.set_mode_gpu()  # for cpu mode do 'caffe.set_mode_cpu()'
    caffe.set_device(0)
    # load network
    net = caffe.Net(net_model, net_weight, caffe.TEST)

    print('Input image path:', im_path)
    print('Ground-truth path:', gt_path)

    in_img = cv2.imread(im_path)
    img_size = [in_img.shape[0], in_img.shape[1]]

    long_len = max(img_size[0], img_size[1])
    x_wholeLen = (img_size[1]*net_size[1]/long_len)
    y_wholeLen = (img_size[0]*net_size[0]/long_len)
    x_wholeLen = int(32*round(x_wholeLen/32))
    y_wholeLen = int(32*round(y_wholeLen/32))
    whole_img = cv2.resize(in_img, (x_wholeLen, y_wholeLen), interpolation=cv2.INTER_CUBIC)
    whole_netSize = [whole_img.shape[0], whole_img.shape[1]]

    tran_img = in_img.copy()
    tran_img = (tran_img.transpose([2, 0, 1])).astype('float')
    tran_img[0, :, :] -= 103.939
    tran_img[1, :, :] -= 116.779
    tran_img[2, :, :] -= 123.68
    tran_img = tran_img*0.017

    tran_wholeImg = whole_img.copy()
    tran_wholeImg = (tran_wholeImg.transpose([2, 0, 1])).astype('float')
    tran_wholeImg[0, :, :] -= 103.939
    tran_wholeImg[1, :, :] -= 116.779
    tran_wholeImg[2, :, :] -= 123.68
    tran_wholeImg = tran_wholeImg*0.017

    gt_map = cv2.imread(gt_path, 0) > 0

    prev_segmap = np.zeros(img_size)

    y_linspace = np.linspace(0, img_size[0] - 1, img_size[0])
    x_linspace = np.linspace(0, img_size[1] - 1, img_size[1])
    x_meshgrid, y_meshgrid = np.meshgrid(x_linspace, y_linspace)

    y_netspace = np.linspace(0, whole_netSize[0] - 1, whole_netSize[0])
    x_netspace = np.linspace(0, whole_netSize[1] - 1, whole_netSize[1])
    x_netMesh, y_netMesh = np.meshgrid(x_netspace, y_netspace)

    whole_pIact = np.zeros(whole_netSize) + max_iact
    whole_nIact = np.zeros(whole_netSize) + max_iact

    pclick_map = np.zeros(img_size)
    nclick_map = np.zeros(img_size)
    target_mat = np.zeros([1, 1, img_size[0], img_size[1]])
    valid_mat = np.zeros([1, 1, img_size[0], img_size[1]])


    yinfo_list, xinfo_list = create_test_batch(img_size, net_size)

    piact_map = []
    niact_map = []
    for y_subinfo in yinfo_list:
        for x_subinfo in xinfo_list:
            piact_map += [np.zeros(img_size) + max_iact]
            niact_map += [np.zeros(img_size) + max_iact]

    for click_id in range(20):

        #############################################################################
        # Generate a positive or a negative click
        #############################################################################
        fn_map = gt_map & (~(prev_segmap > pred_thold))
        fp_map = (~gt_map) & (prev_segmap > pred_thold)

        usr_map, is_pos = generate_click(fn_map, fp_map, pclick_map + nclick_map, img_size, y_meshgrid, x_meshgrid)

        [y_mlist, x_mlist] = np.where(usr_map == np.max(usr_map))
        yx_click = [y_mlist[0], x_mlist[0]]

        print('Click at (' + str(yx_click[0]) + ', ' + str(yx_click[1]) + ') is generated')

        net_yClick = int(round(yx_click[0]*whole_netSize[0]/img_size[0]))
        net_xClick = int(round(yx_click[1]*whole_netSize[1]/img_size[1]))
        single_netIact = np.sqrt(pow(y_netMesh - net_yClick, 2) + pow(x_netMesh - net_xClick, 2))
        single_netIact = np.minimum(single_netIact, max_iact)


        single_iactmap = np.sqrt(pow(y_meshgrid - yx_click[0], 2) + pow(x_meshgrid - yx_click[1], 2))
        single_iactmap = np.minimum(single_iactmap, max_iact)

        if is_pos == 1:
            for iact_id in range(len(piact_map)):
                piact_map[iact_id] = np.minimum(piact_map[iact_id], single_iactmap)
            whole_pIact = np.minimum(whole_pIact, single_netIact)
            pclick_map[yx_click[0], yx_click[1]] = 1
            target_mat[0, 0, yx_click[0], yx_click[1]] = 1
        else:
            for iact_id in range(len(niact_map)):
                niact_map[iact_id] = np.minimum(niact_map[iact_id], single_iactmap)
            whole_nIact = np.minimum(whole_nIact, single_netIact)
            nclick_map[yx_click[0], yx_click[1]] = 1
            target_mat[0, 0, yx_click[0], yx_click[1]] = 0
        valid_mat[0, 0, yx_click[0], yx_click[1]] = 1

        #############################################################################
        # Entire image
        #############################################################################
        net_inimg = np.zeros([1, 3, whole_netSize[0], whole_netSize[1]], np.float32)
        net_inimg[0] = tran_wholeImg

        net_iactmap = np.zeros([1, 2, whole_netSize[0], whole_netSize[1]], np.float32)
        net_iactmap[0, 0] = 1 - whole_pIact/max_iact
        net_iactmap[0, 1] = 1 - whole_nIact/max_iact

        net.blobs['data'].reshape(1, 3, whole_netSize[0], whole_netSize[1])
        net.blobs['iact'].reshape(1, 2, whole_netSize[0], whole_netSize[1])
        net.reshape()

        net.forward(data=net_inimg, iact=net_iactmap)
        whole_segmap = net.blobs['sig_pred'].data.copy()
        whole_segmap = whole_segmap[0, 0]
        whole_segmap = cv2.resize(whole_segmap, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)

        net.blobs['data'].reshape(1, 3, net_size[0], net_size[1])
        net.blobs['iact'].reshape(1, 2, net_size[0], net_size[1])
        net.reshape()

        #############################################################################
        # Image tiles
        #############################################################################
        iact_id = 0
        win_segMap = np.zeros([len(piact_map), img_size[0], img_size[1]])
        win_countMap = np.zeros([len(piact_map), img_size[0], img_size[1]]) + 1e-9
        for y_subinfo in yinfo_list:
            for x_subinfo in xinfo_list:
                if np.sum(pclick_map[y_subinfo[0]:y_subinfo[1], x_subinfo[0]:x_subinfo[1]]) > 0:
                    net_inimg = np.zeros([1, 3, net_size[0], net_size[1]], np.float32)
                    net_inimg[0, :, :y_subinfo[1]-y_subinfo[0], :x_subinfo[1]-x_subinfo[0]] = tran_img[:, y_subinfo[0]:y_subinfo[1], x_subinfo[0]:x_subinfo[1]].copy()

                    net_iactmap = np.zeros([1, 2, net_size[0], net_size[1]], np.float32)
                    net_iactmap[0, 0, :y_subinfo[1]-y_subinfo[0], :x_subinfo[1]-x_subinfo[0]] = 1 - piact_map[iact_id][y_subinfo[0]:y_subinfo[1], x_subinfo[0]:x_subinfo[1]]/max_iact
                    net_iactmap[0, 1, :y_subinfo[1]-y_subinfo[0], :x_subinfo[1]-x_subinfo[0]] = 1 - niact_map[iact_id][y_subinfo[0]:y_subinfo[1], x_subinfo[0]:x_subinfo[1]]/max_iact

                    net.forward(data=net_inimg, iact=net_iactmap)
                    sub_segmap = net.blobs['sig_pred'].data.copy()
                    sub_segmap = sub_segmap[0, 0, :, :]

                    win_segMap[iact_id, y_subinfo[0]:y_subinfo[1], x_subinfo[0]:x_subinfo[1]] += sub_segmap[:y_subinfo[1]-y_subinfo[0], :x_subinfo[1]-x_subinfo[0]].copy()
                    win_countMap[iact_id, y_subinfo[0]:y_subinfo[1], x_subinfo[0]:x_subinfo[1]] += 1
                iact_id += 1

        seg_map = np.zeros(img_size)
        count_map = np.zeros(img_size) + 1e-9

        seg_map += whole_w*whole_segmap
        count_map += whole_w

        seg_map += np.sum(win_segMap, axis=0)
        count_map += np.sum(win_countMap, axis=0)

        seg_map = np.divide(seg_map, count_map)


        # ================================================================
        # Perform BRS
        # ================================================================
        if is_pos:
            is_brs = seg_map[yx_click[0], yx_click[1]]<=pred_thold
        else:
            is_brs = seg_map[yx_click[0], yx_click[1]]>pred_thold
        if is_brs:
            iact_id = 0
            for y_subinfo in yinfo_list:
                for x_subinfo in xinfo_list:
                    if np.sum(pclick_map[y_subinfo[0]:y_subinfo[1], x_subinfo[0]:x_subinfo[1]]) > 0:
                        if (y_subinfo[0] <= yx_click[0]) & (yx_click[0] < y_subinfo[1]) & (x_subinfo[0] <= yx_click[1]) & (yx_click[1] < x_subinfo[1]):
                            net_inimg = np.zeros([1, 3, net_size[0], net_size[1]], np.float32)
                            net_inimg[0, :, :y_subinfo[1]-y_subinfo[0], :x_subinfo[1]-x_subinfo[0]] = tran_img[:, y_subinfo[0]:y_subinfo[1], x_subinfo[0]:x_subinfo[1]].copy()

                            net_iactmap = np.zeros([1, 2, net_size[0], net_size[1]], np.float32)
                            net_iactmap[0, 0, :y_subinfo[1]-y_subinfo[0], :x_subinfo[1]-x_subinfo[0]] = 1 - piact_map[iact_id][y_subinfo[0]:y_subinfo[1], x_subinfo[0]:x_subinfo[1]]/max_iact
                            net_iactmap[0, 1, :y_subinfo[1]-y_subinfo[0], :x_subinfo[1]-x_subinfo[0]] = 1 - niact_map[iact_id][y_subinfo[0]:y_subinfo[1], x_subinfo[0]:x_subinfo[1]]/max_iact

                            tr_tmat = np.zeros([1, 1, net_size[0], net_size[1]], np.float32)
                            tr_vmat = np.zeros([1, 1, net_size[0], net_size[1]], np.float32)
                            tr_tmat[0, 0, :y_subinfo[1]-y_subinfo[0], :x_subinfo[1]-x_subinfo[0]] = target_mat[:, :, y_subinfo[0]:y_subinfo[1], x_subinfo[0]:x_subinfo[1]].copy()
                            tr_vmat[0, 0, :y_subinfo[1]-y_subinfo[0], :x_subinfo[1]-x_subinfo[0]] = valid_mat[:, :, y_subinfo[0]:y_subinfo[1], x_subinfo[0]:x_subinfo[1]].copy()

                            constraints = OrderedDict()

                            constraints['sig_pred'] = constraint([LossFunctions.mean_loss],
                                                                 [{'target_matrix': tr_tmat,
                                                                   'valid_matrix': tr_vmat}])

                            net.forward(data=net_inimg, iact=net_iactmap)

                            tr_imap = net.blobs['iact'].data[0, :, :, :].copy()

                            reg_param = 1e-3

                            result, correct_segMap = InputCorrection(reg_param, net, tr_imap, 'iact', constraints)

                            opt_iact = result[0].reshape(*net.blobs['iact'].data.shape)

                            piact_map[iact_id][y_subinfo[0]:y_subinfo[1], x_subinfo[0]:x_subinfo[1]] = max_iact*(1-opt_iact[0, 0, :y_subinfo[1]-y_subinfo[0], :x_subinfo[1]-x_subinfo[0]])
                            niact_map[iact_id][y_subinfo[0]:y_subinfo[1], x_subinfo[0]:x_subinfo[1]] = max_iact*(1-opt_iact[0, 1, :y_subinfo[1]-y_subinfo[0], :x_subinfo[1]-x_subinfo[0]])

                            win_segMap[iact_id, y_subinfo[0]:y_subinfo[1], x_subinfo[0]:x_subinfo[1]] = correct_segMap[0, 0, :y_subinfo[1]-y_subinfo[0], :x_subinfo[1]-x_subinfo[0]].copy()

                            whole_w = 0

                    iact_id += 1

            seg_map = np.zeros(img_size)
            count_map = np.zeros(img_size) + 1e-9

            seg_map += whole_w*whole_segmap
            count_map += whole_w

            seg_map += np.sum(win_segMap, axis=0)
            count_map += np.sum(win_countMap, axis=0)

            seg_map = np.divide(seg_map, count_map)

        # ================================================================
        # Evaluation
        # ================================================================
        int_section = (gt_map & (seg_map > pred_thold)).astype('float')
        uni_on = (gt_map | (seg_map > pred_thold)).astype('float')
        iou_score = np.sum(int_section) / np.sum(uni_on)

        print(iou_score)
        print('')

        prev_segmap = seg_map.copy()


if __name__ == '__main__':
    im_path = './data/24077_in.png'
    gt_path = './data/24077_gt.png'
    main(im_path, gt_path)
