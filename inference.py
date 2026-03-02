# --------------------------------------------------------
# Tensorflow Faster R-CNN Video Inference
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import time
import numpy as np
import torch
import cv2
from PIL import Image

# make sure your lib folder is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "lib")))

import _init_paths
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import load_net
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.utils.blob import im_list_to_blob
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.net_utils import vis_detections_filtered_objects_PIL

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Faster R-CNN Video Inference')
    parser.add_argument('--dataset', dest='dataset', default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file', default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net', default='res101', type=str,
                        help='vgg16, res50, res101, res152')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir', default="./models")
    parser.add_argument('--save_dir', dest='save_dir', default="videos_det")
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--class_agnostic', dest='class_agnostic', action='store_true')
    parser.add_argument('--thresh_hand', type=float, default=0.6)
    parser.add_argument('--thresh_obj', type=float, default=0.7)
    parser.add_argument('--thresh_contact', type=float, default=0.5,
                        help='Only visualize detections with person contact probability >= threshold.')
    parser.add_argument('--thresh_person_contact', type=float, default=0.2,
                        help='Only visualize detections with person contact probability >= threshold.')
    parser.add_argument('--thresh_self_contact', type=float, default=0.5,
                        help='Only visualize detections with person contact probability >= threshold.')
    parser.add_argument('--thresh_no_contact', type=float, default=0.5,
                        help='Only visualize detections with no contact probability >= threshold.')
    parser.add_argument('--thresh_object_contact', type=float, default=0.5,
                        help='Only visualize detections with person contact probability >= threshold.')
    parser.add_argument('--min_contact_streaks', type=int, default=3)
    parser.add_argument('--video_dir', type=str, default="/Users/jing/Synapxe/semantic-segmentation/videos")
    parser.add_argument('--webcam', action='store_true',
                        help='Enable live webcam inference using device index 0.')
    parser.add_argument('--webcam_width', type=int, default=1920,
                        help='Webcam capture width (used only with --webcam).')
    parser.add_argument('--webcam_height', type=int, default=1080,
                        help='Webcam capture height (used only with --webcam).')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save output video (display only for webcam).')
    parser.add_argument('--hand_states', type=str, default="1,2,3,4",
                        help='Comma-separated contact states to visualize. '
                             '0=N (No Contact), 1=S (Self), 2=O (Other), 3=P (Portable), 4=F (Fixed).')
    parser.add_argument('--hand_side', type=str, default="any", choices=["L", "R", "any"],
                        help='Filter hand side to visualize: L, R, or any.')
    parser.add_argument('--contact_confirm_frames', type=int, default=1)
    parser.add_argument('--fps', type=float, default=0.0,
                        help='Override output FPS (useful when input FPS is wrong/variable). 0 = use input FPS.')
    parser.add_argument('--checksession', type=int, default=1)
    parser.add_argument('--checkepoch', type=int, default=8)
    parser.add_argument('--checkpoint', type=int, required=True)
    args = parser.parse_args()
    return args

def _get_image_blob(im):
    """Converts an image into a network input blob"""
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


# -------------------------
# Device Setup
# -------------------------
def setup_device(cuda_flag):
    if cuda_flag and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device.type}")
    return device

def load_model(device, args, use_cuda, pascal_classes):
    model_dir = os.path.join(args.load_dir, f"{args.net}_handobj_100K", args.dataset)
    if not os.path.exists(model_dir):
        raise Exception(f'There is no input directory for loading network from {model_dir}')
    load_name = os.path.join(model_dir, f'faster_rcnn_{args.checksession}_{args.checkepoch}_{args.checkpoint}.pth')

    if args.net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        fasterRCNN = resnet(pascal_classes, int(args.net[3:]), pretrained=False, class_agnostic=args.class_agnostic)

    fasterRCNN.create_architecture()
    print(f"load checkpoint {load_name}")
    if use_cuda:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    if use_cuda:
        cfg.CUDA = True
    fasterRCNN.to(device)
    fasterRCNN.eval()

    return fasterRCNN

def filter_contact_streaks(
        hand_dets,
        obj_dets,
        self_contact_streak,
        person_contact_streak,
        object_contact_streak,
        portable_contact_streak,
        fixed_contact_streak,
        min_streak=3,

):
    has_self = has_person = has_portable = has_fixed = False
    if hand_dets is not None and hand_dets.shape[0] > 0:
        states = hand_dets[:, 5].astype(int)
        has_self = np.any(states == 1)
        has_person = np.any(states == 2)
        has_portable = np.any(states == 3)
        has_fixed = np.any(states == 4)

    # Streak update (reset to 0 if missing this frame)
    self_contact_streak = self_contact_streak + 1 if has_self else 0
    person_contact_streak = person_contact_streak + 1 if has_person else 0
    portable_contact_streak = portable_contact_streak + 1 if has_portable else 0
    fixed_contact_streak = fixed_contact_streak + 1 if has_fixed else 0
    object_contact_streak = object_contact_streak + 1 if has_portable or has_fixed else 0

    # Visualize only when the corresponding contact streak > 3
    vis_hand_dets = hand_dets
    vis_obj_dets = obj_dets
    if hand_dets is not None:
        states = hand_dets[:, 5].astype(int)
        keep_mask = np.ones(states.shape[0], dtype=bool)
        if self_contact_streak <= min_streak:
            keep_mask &= (states != 1)
        if person_contact_streak <= min_streak:
            keep_mask &= (states != 2)
        if object_contact_streak <= min_streak:
            keep_mask &= (states != 3) & (states != 4)
        vis_hand_dets = hand_dets[keep_mask]
        if vis_hand_dets.size == 0:
            vis_hand_dets = None

    if obj_dets is not None:
        states = obj_dets[:, 5].astype(int)
        keep_mask = np.ones(states.shape[0], dtype=bool)
        if self_contact_streak <= min_streak:
            keep_mask &= (states != 1)
        if person_contact_streak <= min_streak:
            keep_mask &= (states != 2)
        if object_contact_streak <= min_streak:
            keep_mask &= (states != 3) & (states != 4)
        vis_obj_dets = obj_dets[keep_mask]
        if vis_obj_dets.size == 0:
            vis_obj_dets = None

    return vis_hand_dets, vis_obj_dets, self_contact_streak, person_contact_streak, object_contact_streak, portable_contact_streak, fixed_contact_streak

def draw_contact_streaks(
    frame_bgr,
    self_contact_streak,
    person_contact_streak,
    object_contact_streak,
    portable_contact_streak,
    fixed_contact_streak,
):
    lines = [
        (f"Self contact streak: {self_contact_streak}", (0, 255, 255)),      # yellow
        (f"Person contact streak: {person_contact_streak}", (0, 255, 0)),    # green
        (f"Object contact streak: {object_contact_streak}", (200, 255, 0)),
        (f"Portable contact streak: {portable_contact_streak}", (255, 200, 0)),
        (f"Fixed contact streak: {fixed_contact_streak}", (255, 255, 255)),  # white
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    margin = 12
    line_gap = 10

    # Measure tallest text once for consistent vertical spacing
    (_, text_h), _ = cv2.getTextSize("Ag", font, scale, thickness)
    step = text_h + line_gap

    h, w = frame_bgr.shape[:2]

    for i, (text, color) in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = max(0, w - tw - margin)
        y = margin + th + i * step
        y = min(h - 5, y)

        # outline
        cv2.putText(frame_bgr, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # text
        cv2.putText(frame_bgr, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    return frame_bgr


def main():
    args = parse_args()

    state_thresholds = {
        0: args.thresh_no_contact,
        1: args.thresh_self_contact,
        2: args.thresh_person_contact,
        3: args.thresh_object_contact,
        4: args.thresh_object_contact
    }

    # Parse hand state filter
    args.hand_states = [int(x) for x in args.hand_states.split(",") if x.strip() != ""]
    device = setup_device(args.cuda)   
    use_cuda = device.type == "cuda"
    # Load config
    if args.cfg_file:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = use_cuda
    np.random.seed(cfg.RNG_SEED)

    # load model
    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
    fasterRCNN = load_model(device, args, use_cuda, pascal_classes)

    # initialize tensor holders (like demo.py)
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    box_info = torch.FloatTensor(1)
    if device.type != "cpu":
        im_data = im_data.to(device)
        im_info = im_info.to(device)
        num_boxes = num_boxes.to(device)
        gt_boxes = gt_boxes.to(device)
        box_info = box_info.to(device)

    # Loop over videos or webcam
    if args.webcam:
        video_files = [None]
    else:
        video_files = glob.glob(os.path.join(args.video_dir, "*.mp4"))
    os.makedirs(args.save_dir, exist_ok=True)

    stop_requested = False
    for video_file in video_files:
        if video_file is None:
            webcam_index = 0
            print(f"Processing webcam {webcam_index}...")
            cap = cv2.VideoCapture(webcam_index)
        else:
            print(f"Processing {video_file}...")
            cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            if video_file is None:
                print(f"Failed to open webcam {webcam_index}")
            else:
                print(f"Failed to open {video_file}")
            continue

        if video_file is None:
            if args.webcam_width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.webcam_width)
            if args.webcam_height > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.webcam_height)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if args.fps and args.fps > 0:
            fps = args.fps
        elif not fps or fps < 1 or fps > 120:
            fps = 30
                    
        # Read first frame to get TRUE dimensions
        ret, frame = cap.read()
        if not ret:
            print("Failed to read first frame.")
            cap.release()
            continue

        height, width = frame.shape[:2]

        if args.no_save:
            output_path = None
            out = None
        else:
            if video_file is None:
                base_name = f"webcam{webcam_index}_{time.strftime('%Y%m%d-%H%M%S')}"
            else:
                base_name = os.path.basename(video_file)[:-4]

            thresh_tag = (
                f"hands-{args.thresh_hand:.2f}-"
                f"objs-{args.thresh_obj:.2f}-"
                f"no-self-person-obj_contact-"
                f"{args.thresh_no_contact:.2f}-"
                f"{args.thresh_self_contact:.2f}-"
                f"{args.thresh_person_contact:.2f}-"
                f"{args.thresh_object_contact:.2f}"
            )
            output_path = os.path.join(
                args.save_dir,
                f"{base_name}_{thresh_tag}_{args.hand_states}det.mp4"
            )

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                raise RuntimeError(f"VideoWriter failed for {output_path}")

        # IMPORTANT: we already consumed first frame
        frame_idx = 0

        self_contact_streak = 0
        person_contact_streak = 0
        object_contact_streak = 0
        portable_contact_streak = 0
        fixed_contact_streak = 0

        while True:

            # First iteration uses already-read frame
            if frame_idx > 0:
                ret, frame = cap.read()
                if not ret:
                    break

            im = frame
            blobs, im_scales = _get_image_blob(im)
            im_data_pt = torch.from_numpy(blobs).permute(0, 3, 1, 2)
            im_info_pt = torch.from_numpy(
                np.array([[im_data_pt.shape[2], im_data_pt.shape[3], im_scales[0]]], dtype=np.float32)
            )

            with torch.no_grad():
                im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                gt_boxes.resize_(1, 1, 5).zero_()
                num_boxes.resize_(1).zero_()
                box_info.resize_(1, 1, 5).zero_()

                rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)
                
                scores = cls_prob.data
                boxes = rois.data[:, :, 1:5]
                contact_vector = loss_list[0][0]
                offset_vector = loss_list[1][0].detach()
                lr_vector = loss_list[2][0].detach()

                probs = torch.softmax(contact_vector, 2)
                contact_probs, contact_indices = torch.max(probs, 2)
                contact_probs = contact_probs.squeeze(0).unsqueeze(-1).float()
                contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()


                lr = torch.sigmoid(lr_vector) > 0.5
                lr = lr.squeeze(0).float()

                if cfg.TEST.BBOX_REG:
                    box_deltas = bbox_pred.data
                    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                        if args.class_agnostic:
                            stds = torch.as_tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS, device=device)
                            means = torch.as_tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS, device=device)
                            box_deltas = box_deltas.view(-1, 4) * stds + means

                            box_deltas = box_deltas.view(1, -1, 4)
                        else:
                            stds = torch.as_tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS, device=device)
                            means = torch.as_tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS, device=device)
                            box_deltas = box_deltas.view(-1, 4) * stds + means
                            box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

                    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                else:
                    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

                pred_boxes /= im_scales[0]

                scores = scores.squeeze()
                pred_boxes = pred_boxes.squeeze()

                # Visualize
                obj_dets, hand_dets = None, None
                for j, cls_name in enumerate(pascal_classes[1:], 1):
                    if cls_name == 'hand':
                        inds = torch.nonzero(scores[:, j] > args.thresh_hand).view(-1)
                    else:
                        inds = torch.nonzero(scores[:, j] > args.thresh_obj).view(-1)

                    if inds.numel() > 0:
                        cls_scores = scores[:, j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        if args.class_agnostic:
                            cls_boxes = pred_boxes[inds, :]
                        else:
                            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                        cls_dets = torch.cat(
                            (
                                cls_boxes, 
                                cls_scores.unsqueeze(1), 
                                contact_indices[inds], 
                                contact_probs[inds],
                                offset_vector.squeeze(0)[inds], 
                                lr[inds]),
                            1
                        )
                        cls_dets = cls_dets[order]
                        if device.type == "mps":
                            keep = nms(cls_boxes[order, :].cpu(), cls_scores[order].cpu(), cfg.TEST.NMS)
                            cls_dets = cls_dets.cpu()[keep.view(-1).long()]
                        else:
                            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                            cls_dets = cls_dets[keep.view(-1).long()]
                        if cls_name == 'targetobject':
                            obj_dets = cls_dets.cpu().numpy()
                        if cls_name == 'hand':
                            hand_dets = cls_dets.cpu().numpy()

                # ==========================================================
                # HAND FILTERING
                # ==========================================================
                if hand_dets is not None:

                    # [x1,y1,x2,y2, score, state, contact_prob, off1, off2, off3, lr]
                    states = hand_dets[:, 5].astype(int)
                    contact_probs_det = hand_dets[:, 6]
                    lr_vals = hand_dets[:, 10]

                    thresholds_array = np.array([
                        state_thresholds.get(s, args.thresh_contact)
                        for s in states
                    ])

                    state_mask = np.isin(states, args.hand_states)
                    prob_mask = contact_probs_det >= thresholds_array

                    keep_mask = state_mask & prob_mask

                    if args.hand_side != "any":
                        if args.hand_side == "L":
                            keep_mask &= (lr_vals == 0)
                        else:
                            keep_mask &= (lr_vals == 1)

                    hand_dets = hand_dets[keep_mask]

                    if hand_dets.size == 0:
                        hand_dets = None

                # ==========================================================
                # OBJECT FILTERING (now symmetric with hands)
                # ==========================================================
                if obj_dets is not None:

                    # assume same structure:
                    # [x1,y1,x2,y2, score, state, contact_prob, off1, off2, off3, lr]
                    states = obj_dets[:, 5].astype(int)
                    contact_probs_det = obj_dets[:, 6]

                    thresholds_array = np.array([
                        state_thresholds.get(s, args.thresh_contact)
                        for s in states
                    ])

                    # if you want to restrict objects to specific states:
                    state_mask = np.isin(states, args.hand_states)  # or define args.obj_states

                    prob_mask = contact_probs_det >= thresholds_array

                    keep_mask = state_mask & prob_mask

                    obj_dets = obj_dets[keep_mask]

                    if obj_dets.size == 0:
                        obj_dets = None

                # If no objects survived filtering, drop hands that claim object contact
                if obj_dets is None and hand_dets is not None:
                    # state codes: 0=N (No Contact), 1=S (Self), 2=O (Other), 3=P (Portable), 4=F (Fixed)
                    object_contact_states = [1, 2, 3, 4]
                    keep_mask = ~np.isin(hand_dets[:, 5], object_contact_states)
                    hand_dets = hand_dets[keep_mask]
                    if hand_dets.size == 0:
                        hand_dets = None

                # ======================= Debug: print detection counts and top scores (every 50 frames) =======================
                if frame_idx % 5 == 0:
                    hand_count = 0 if hand_dets is None else int(hand_dets.shape[0])
                    obj_count = 0 if obj_dets is None else int(obj_dets.shape[0])
                    hand_top = None if hand_dets is None or hand_dets.shape[0] == 0 else float(hand_dets[0, 4])
                    obj_top = None if obj_dets is None or obj_dets.shape[0] == 0 else float(obj_dets[0, 4])
                    
                    # contact probability of the first hand (or None if no hands)
                    hand_contact_prob = None
                    if hand_dets is not None and hand_dets.shape[0] > 0:
                        hand_contact_prob = float(hand_dets[0, 6])

                    if hand_contact_prob is not None:
                        contact_text = f"{hand_contact_prob:.3f}"
                    else:
                        contact_text = "None"

                    print(
                        f"[frame {frame_idx}] detections -> hands: {hand_count}, objs: {obj_count}, "
                        f"top_scores(h,obj): ({hand_top}, {obj_top}), contact_prob(h): {contact_text}"
                    )

                # ====================================================== Debug =======================================================
                vis_hand_dets, vis_obj_dets, self_contact_streak, person_contact_streak, object_contact_streak, portable_contact_streak, fixed_contact_streak = filter_contact_streaks(
                    hand_dets, obj_dets, self_contact_streak, person_contact_streak, object_contact_streak, portable_contact_streak, fixed_contact_streak, args.min_contact_streaks
                )

                im2show = vis_detections_filtered_objects_PIL(frame, vis_obj_dets, vis_hand_dets, args.thresh_hand, args.thresh_obj)
                im2show_rgb = cv2.cvtColor(np.array(im2show), cv2.COLOR_RGB2BGR)
                im2show_rgb = draw_contact_streaks(im2show_rgb, self_contact_streak, person_contact_streak, object_contact_streak, portable_contact_streak, fixed_contact_streak)
                if out is not None:
                    out.write(im2show_rgb)
                if video_file is None:
                    cv2.imshow("Hand-Object Detector", im2show_rgb)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_requested = True
                        break

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"{frame_idx} frames processed...")

        cap.release()
        if out is not None:
            out.release()
        if video_file is None:
            cv2.destroyAllWindows()
        if output_path:
            print(f"Saved annotated video to {output_path}")
        if stop_requested:
            break

if __name__ == "__main__":
    main()
