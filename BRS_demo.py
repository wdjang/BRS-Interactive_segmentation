import PIL
from PIL import Image, ImageTk
import cv2
from skimage import transform, filters, measure, data
import numpy as np
from collections import OrderedDict
import caffe
from InputCorrection import *
import Tkinter as tk
import tkFileDialog


net_size = [480, 480]
pred_thold = 0.5
whole_w = 1.0
max_iact = float(255.0)

out_rad = 5
in_rad = 3

net_weight = './model/BRS_DenseNet.caffemodel'
net_model = './model/deploy.prototxt'
im_path = './data/24077_in.png'
gt_path = './data/24077_gt.png'

caffe.set_mode_gpu()  # for cpu mode do 'caffe.set_mode_cpu()'
caffe.set_device(0)
# load network
net = caffe.Net(net_model, net_weight, caffe.TEST)

click_id = -1
yx_click = []
is_pos = 1













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

def load_image():
    global click_id, whole_pIact, whole_nIact, whole_w, whole_netSize, cv_imgSize, yx_click, y_netMesh, x_netMesh
    global y_meshgrid, x_meshgrid, piact_map, niact_map, pclick_map, nclick_map
    global target_mat, valid_mat, tran_wholeImg, net, cv_imgSize, net_size, yinfo_list, xinfo_list, tran_img
    global pred_thold, file_ind, img_size
    global is_pos
    global w, v
    global photo_left, b1, b2, b4, b5
    global img_original

    is_pos = 1
    yx_click = []

    b1.pack()
    b1.pack_forget()
    b2.pack()
    b2.pack_forget()
    # b3.pack()
    # b3.pack_forget()
    b4.pack()
    b4.pack_forget()
    b5.pack()
    b5.pack_forget()

    file_ind += 1

    print("%d\n" % (file_ind + 1))
    click_id = -1



    File = tkFileDialog.askopenfilename(parent=root, initialdir="./data", title='Select an image')
    img_original = Image.open(File)
    # img_original = Image.open(impath_list[file_ind])
    photo_left = ImageTk.PhotoImage(img_original)
    img_size = photo_left._PhotoImage__size

    # w = tk.Canvas(root, width=2 * img_size[0] + 120, height=img_size[1])
    w.delete(tk.ALL)
    w.config(width=img_size[0] + 120, height=img_size[1])
    w.pack()
    w.create_image(0, 0, image=photo_left, anchor="nw")
    w.pack()

    temp_map = np.zeros([img_size[1], img_size[0]])
    temp_img = Image.fromarray(np.uint8(temp_map * 255))
    post_segmap = ImageTk.PhotoImage(temp_img)





    # in_img = cv2.imread(impath_list[file_ind])
    in_img = cv2.imread(File)
    cv_imgSize = [in_img.shape[0], in_img.shape[1]]

    long_len = max(cv_imgSize[0], cv_imgSize[1])
    x_wholeLen = (cv_imgSize[1]*net_size[1]/long_len)
    y_wholeLen = (cv_imgSize[0]*net_size[0]/long_len)
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

    prev_segmap = np.zeros(cv_imgSize)
    prev_evalmap = np.zeros(cv_imgSize)

    y_linspace = np.linspace(0, cv_imgSize[0] - 1, cv_imgSize[0])
    x_linspace = np.linspace(0, cv_imgSize[1] - 1, cv_imgSize[1])
    x_meshgrid, y_meshgrid = np.meshgrid(x_linspace, y_linspace)

    y_netspace = np.linspace(0, whole_netSize[0] - 1, whole_netSize[0])
    x_netspace = np.linspace(0, whole_netSize[1] - 1, whole_netSize[1])
    x_netMesh, y_netMesh = np.meshgrid(x_netspace, y_netspace)

    whole_pIact = np.zeros(whole_netSize) + max_iact
    whole_nIact = np.zeros(whole_netSize) + max_iact
    net_trgMat = np.zeros(whole_netSize)
    net_valMat = np.zeros(whole_netSize)

    pclick_map = np.zeros(cv_imgSize)
    nclick_map = np.zeros(cv_imgSize)
    target_mat = np.zeros([1, 1, cv_imgSize[0], cv_imgSize[1]])
    valid_mat = np.zeros([1, 1, cv_imgSize[0], cv_imgSize[1]])

    whole_w = 1.0

    yinfo_list, xinfo_list = create_test_batch(cv_imgSize, net_size)

    piact_map = []
    niact_map = []
    for y_subinfo in yinfo_list:
        for x_subinfo in xinfo_list:
            piact_map += [np.zeros(cv_imgSize) + max_iact]
            niact_map += [np.zeros(cv_imgSize) + max_iact]

    # button with text closing window
    b1 = tk.Button(root, text="Next", command=load_image)
    b1.place(bordermode=tk.OUTSIDE, height=50, width=100, x=img_size[0]+10, y=img_size[1] / 2 - 70)
    
    b2 = tk.Button(root, text="Load", command=load_image)
    b2.place(bordermode=tk.OUTSIDE, height=50, width=100, x=img_size[0]+10, y=img_size[1] / 2 - 150)

    v = tk.StringVar()
    v.set("F")  # initialize

    b4 = tk.Radiobutton(root, text="Foreground", variable=v, value="F", command=fg_click)
    b4.place(bordermode=tk.OUTSIDE, height=30, width=100, x=img_size[0]+10, y=img_size[1]/2+20)
    b5 = tk.Radiobutton(root, text="Background", variable=v, value="B", command=bg_click)
    b5.place(bordermode=tk.OUTSIDE, height=30, width=100, x=img_size[0]+10, y=img_size[1]/2+50)


def getorigin(eventorigin):
    global yx_click, img_size, photo_left, img_original

    if (eventorigin.y < img_size[1]) & (eventorigin.x < img_size[0]):
        yx_click.append([eventorigin.y, eventorigin.x, is_pos])

        np_original = np.asarray(img_original).copy()

        img_original = Image.fromarray(np_original)
        photo_left = ImageTk.PhotoImage(img_original)

        w.create_image(0, 0, image=photo_left, anchor="nw")

        print(yx_click[-1][0], yx_click[-1][1])

        perform_segmentation()

def fg_click():
    global is_pos
    is_pos = 1
    print('Foreground click')

def bg_click():
    global is_pos
    is_pos = 0
    print('Background click')

def perform_segmentation():
    global click_id, whole_pIact, whole_nIact, whole_w, whole_netSize, cv_imgSize, yx_click, y_netMesh, x_netMesh
    global y_meshgrid, x_meshgrid, piact_map, niact_map, pclick_map, nclick_map
    global target_mat, valid_mat, tran_wholeImg, net, cv_imgSize, net_size, yinfo_list, xinfo_list, tran_img
    global pred_thold, img_size
    global is_pos
    global w, v
    global photo_left, b1, b2, b4, b5

    click_id += 1

    net_yClick = int(round(yx_click[-1][0]*whole_netSize[0]/cv_imgSize[0]))
    net_xClick = int(round(yx_click[-1][1]*whole_netSize[1]/cv_imgSize[1]))
    single_netIact = np.sqrt(pow(y_netMesh - net_yClick, 2) + pow(x_netMesh - net_xClick, 2))
    single_netIact = np.minimum(single_netIact, max_iact)


    single_iactmap = np.sqrt(pow(y_meshgrid - yx_click[-1][0], 2) + pow(x_meshgrid - yx_click[-1][1], 2))
    single_iactmap = np.minimum(single_iactmap, max_iact)

    if is_pos == 1:
        for iact_id in range(len(piact_map)):
            piact_map[iact_id] = np.minimum(piact_map[iact_id], single_iactmap)
        whole_pIact = np.minimum(whole_pIact, single_netIact)
        pclick_map[yx_click[-1][0], yx_click[-1][1]] = 1
        target_mat[0, 0, yx_click[-1][0], yx_click[-1][1]] = 1
    else:
        for iact_id in range(len(niact_map)):
            niact_map[iact_id] = np.minimum(niact_map[iact_id], single_iactmap)
        whole_nIact = np.minimum(whole_nIact, single_netIact)
        nclick_map[yx_click[-1][0], yx_click[-1][1]] = 1
        target_mat[0, 0, yx_click[-1][0], yx_click[-1][1]] = 0
    valid_mat[0, 0, yx_click[-1][0], yx_click[-1][1]] = 1





    # Entire image
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
    whole_segmap = cv2.resize(whole_segmap, (cv_imgSize[1], cv_imgSize[0]), interpolation=cv2.INTER_CUBIC)








    net.blobs['data'].reshape(1, 3, net_size[0], net_size[1])
    net.blobs['iact'].reshape(1, 2, net_size[0], net_size[1])
    net.reshape()

    iact_id = 0
    win_segMap = np.zeros([len(piact_map), cv_imgSize[0], cv_imgSize[1]])
    win_countMap = np.zeros([len(piact_map), cv_imgSize[0], cv_imgSize[1]]) + 1e-9
    for y_subinfo in yinfo_list:
        for x_subinfo in xinfo_list:
            if np.sum(pclick_map[y_subinfo[0]:y_subinfo[1], x_subinfo[0]:x_subinfo[1]]) > 0:
                # print('inside')
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

    seg_map = np.zeros(cv_imgSize)
    count_map = np.zeros(cv_imgSize) + 1e-9

    seg_map += whole_w*whole_segmap
    count_map += whole_w

    seg_map += np.sum(win_segMap, axis=0)
    count_map += np.sum(win_countMap, axis=0)

    seg_map = np.divide(seg_map, count_map)


    if is_pos:
        is_brs = seg_map[yx_click[-1][0], yx_click[-1][1]]<=pred_thold
    else:
        is_brs = seg_map[yx_click[-1][0], yx_click[-1][1]]>pred_thold
    if is_brs:
        iact_id = 0
        for y_subinfo in yinfo_list:
            for x_subinfo in xinfo_list:
                if np.sum(pclick_map[y_subinfo[0]:y_subinfo[1], x_subinfo[0]:x_subinfo[1]]) > 0:
                    if (y_subinfo[0] <= yx_click[-1][0]) & (yx_click[-1][0] < y_subinfo[1]) & (x_subinfo[0] <= yx_click[-1][1]) & (yx_click[-1][1] < x_subinfo[1]):
                        # print('Fixing now..')
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
                        # correct input data
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

        seg_map = np.zeros(cv_imgSize)
        count_map = np.zeros(cv_imgSize) + 1e-9

        seg_map += whole_w*whole_segmap
        count_map += whole_w

        seg_map += np.sum(win_segMap, axis=0)
        count_map += np.sum(win_countMap, axis=0)

        seg_map = np.divide(seg_map, count_map)

    logical_map = seg_map > 0.5

    np_origimg = np.asarray(img_original).copy()

    r_origimg = np_origimg[:, :, 0].copy()
    g_origimg = np_origimg[:, :, 1].copy()
    b_origimg = np_origimg[:, :, 2].copy()

    blend_ratio = 0.5

    r_origimg[logical_map] = (1 - blend_ratio) * r_origimg[logical_map] + blend_ratio * 200
    g_origimg[logical_map] = (1 - blend_ratio) * g_origimg[logical_map] + blend_ratio * 200
    b_origimg[logical_map] = (1 - blend_ratio) * b_origimg[logical_map] + blend_ratio * 0

    for yx_inst in yx_click:
        dist_map = np.sqrt(pow(y_meshgrid - yx_inst[0], 2) + pow(x_meshgrid - yx_inst[1], 2))
        outer_map = dist_map < out_rad
        inner_map = dist_map < in_rad
        
        r_origimg[outer_map] = 255
        g_origimg[outer_map] = 255
        b_origimg[outer_map] = 255
        
        if yx_inst[2] == 1:
            r_origimg[inner_map] = 191
            g_origimg[inner_map] = 42
            b_origimg[inner_map] = 42
        else:
            r_origimg[inner_map] = 9
            g_origimg[inner_map] = 33
            b_origimg[inner_map] = 64

    np_origimg[:, :, 0] = r_origimg
    np_origimg[:, :, 1] = g_origimg
    np_origimg[:, :, 2] = b_origimg


    img_resimg = Image.fromarray(np_origimg)
    photo_left = ImageTk.PhotoImage(img_resimg)

    w.create_image(0, 0, image=photo_left, anchor="nw")

    prev_segmap = seg_map.copy()

# ==========================================================================================================
# Main code
# ==========================================================================================================

root = tk.Tk()
root.title("2019_CVPR_PaperID_3244")

file_ind = 0
click_id = -1

print("%d\n" % (file_ind + 1))

File = tkFileDialog.askopenfilename(parent=root, initialdir="./data", title='Select an image')
img_original = Image.open(File)
photo_left = ImageTk.PhotoImage(img_original)
img_size = photo_left._PhotoImage__size
w = tk.Canvas(root, width=img_size[0] + 120, height=img_size[1])
w.pack()
w.create_image(0, 0, image=photo_left, anchor="nw")
w.pack()

temp_map = np.zeros([img_size[1], img_size[0]])
temp_img = Image.fromarray(np.uint8(temp_map * 255))
post_segmap = ImageTk.PhotoImage(temp_img)


yx_click = []

in_img = cv2.imread(File)
cv_imgSize = [in_img.shape[0], in_img.shape[1]]

long_len = max(cv_imgSize[0], cv_imgSize[1])
x_wholeLen = (cv_imgSize[1]*net_size[1]/long_len)
y_wholeLen = (cv_imgSize[0]*net_size[0]/long_len)
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

prev_segmap = np.zeros(cv_imgSize)
prev_evalmap = np.zeros(cv_imgSize)

y_linspace = np.linspace(0, cv_imgSize[0] - 1, cv_imgSize[0])
x_linspace = np.linspace(0, cv_imgSize[1] - 1, cv_imgSize[1])
x_meshgrid, y_meshgrid = np.meshgrid(x_linspace, y_linspace)

y_netspace = np.linspace(0, whole_netSize[0] - 1, whole_netSize[0])
x_netspace = np.linspace(0, whole_netSize[1] - 1, whole_netSize[1])
x_netMesh, y_netMesh = np.meshgrid(x_netspace, y_netspace)

whole_pIact = np.zeros(whole_netSize) + max_iact
whole_nIact = np.zeros(whole_netSize) + max_iact
net_trgMat = np.zeros(whole_netSize)
net_valMat = np.zeros(whole_netSize)

pclick_map = np.zeros(cv_imgSize)
nclick_map = np.zeros(cv_imgSize)
target_mat = np.zeros([1, 1, cv_imgSize[0], cv_imgSize[1]])
valid_mat = np.zeros([1, 1, cv_imgSize[0], cv_imgSize[1]])

whole_w = 1.0

yinfo_list, xinfo_list = create_test_batch(cv_imgSize, net_size)

piact_map = []
niact_map = []
for y_subinfo in yinfo_list:
    for x_subinfo in xinfo_list:
        piact_map += [np.zeros(cv_imgSize) + max_iact]
        niact_map += [np.zeros(cv_imgSize) + max_iact]

# mouseclick event
w.bind("<Button 1>", getorigin)
w.pack()

# button with text closing window
b2 = tk.Button(root, text="Load", command=load_image)
b2.place(bordermode=tk.OUTSIDE, height=50, width=100, x=img_size[0]+10, y=img_size[1] / 2 - 150)

# button with text closing window
b1 = tk.Button(root, text="Next", command=load_image)
b1.place(bordermode=tk.OUTSIDE, height=50, width=100, x=img_size[0] + 10, y=img_size[1] / 2 - 70)

MODES = [
    ("Foreground", "F"),
    ("Background", "B"),
]

v = tk.StringVar()
v.set("F")  # initialize

b4 = tk.Radiobutton(root, text="Foreground", variable=v, value="F", command=fg_click)
b4.place(bordermode=tk.OUTSIDE, height=30, width=100, x=img_size[0] + 10, y=img_size[1] / 2 + 20)
b5 = tk.Radiobutton(root, text="Background", variable=v, value="B", command=bg_click)
b5.place(bordermode=tk.OUTSIDE, height=30, width=100, x=img_size[0] + 10, y=img_size[1] / 2 + 50)

root.mainloop()


















