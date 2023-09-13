from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from skimage import transform as trans


trained_model = "Resnet50_Final.pth"
confidence_threshold = 0.02
top_k = 5000
nms_threshold = 0.4
keep_top_k = 750

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class RetinaFaceCustom:
    def __init__(self) -> None:
        torch.set_grad_enabled(False)
        # net and model
        net = RetinaFace(cfg=cfg_re50, phase = 'test')
        net = load_model(net, trained_model, True)
        net.eval()
        cudnn.benchmark = True
        self.device = torch.device("cuda")
        self.net = net.to(self.device)

    def detect(self, img_bgr):
        img = np.float32(img_bgr)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(cfg_re50, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_re50['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_re50['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        return dets, landms

def crop_face_placeholder(image, landmark, size=256):
    std_ldmk = np.array([[ 94.52540779, 119.8576889 ],
                         [160.51735306, 120.96569252],
                         [130.70940971, 154.07973099],
                         [101.36186028, 183.09577942],
                         [155.84117126, 183.8509903 ]])
    tform = trans.AffineTransform()
    tform.estimate(landmark, std_ldmk)
    M = tform.params[0:2, :]
    croped = cv2.warpAffine(image, M, (size, size), borderValue=cv2.BORDER_REFLECT)
    inv = cv2.invertAffineTransform(M)
    return croped, inv

def put_face_into_placeholder(image, patch, inv):
    inv_patch = cv2.warpAffine(patch, inv, (image.shape[1], image.shape[0]))

    return inv_patch

def softmask_blending(fg, bg, mask):
    blur_kernel = 25
    mask = cv2.erode(mask, np.ones((blur_kernel, blur_kernel), np.uint8))
    softmask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
    blended = fg * softmask + bg * (1 - softmask)
    return blended

if __name__ == "__main__":
    Detector = RetinaFaceCustom()
    img_bgr = cv2.imread("img.png")
    dets, landms = Detector.detect(img_bgr)
    crop, inv = crop_face_placeholder(img_bgr, np.reshape(landms, [5, 2]), size=256)
    patch_mask = np.ones([256, 256, 3])
    inv_mask = put_face_into_placeholder(img_bgr, patch_mask, inv)
    inv_face = put_face_into_placeholder(img_bgr, crop, inv)
    out = inv_face * inv_mask + img_bgr * (1 - inv_mask)
    cv2.imwrite("out.png", out)

