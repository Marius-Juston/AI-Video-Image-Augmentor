import datetime
import os
import shutil
import time
import urllib.request

import numpy as np
import torch
from imageio import imsave, imread
from torch.autograd import Variable

import dain_model.misc.networks
from dain_model.my_args import args
from dain_model.misc.AverageMeter import AverageMeter

if __name__ == '__main__':
    args.use_cuda = False

    model = dain_model.misc.networks.__dict__[args.netName](
        channel=args.channels,
        filter_size=args.filter_size,
        timestep=args.time_step,
        training=False)

    if args.use_cuda:
        model = model.cuda()

    model_url = 'http://vllab1.ucmerced.edu/~wenbobao/DAIN/best.pth'
    model_path = './model_weights/best.pth'

    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)

    if args.use_cuda:
        pretrained_dict = torch.load(model_path)
    else:
        pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # 4. release the pretrained dict for saving memory
    pretrained_dict = []
    model = model.eval()  # deploy mode

    frames_dir = '../input'
    output_dir = args.frame_output_dir

    fps = 24.35
    TARGET_FPS = fps * 2

    timestep = fps / TARGET_FPS

    time_offsets = [kk * timestep for kk in range(1, int(1.0 / timestep))]

    input_frame = 0
    final_frame = 50

    loop_timer = AverageMeter()

    torch.set_grad_enabled(False)

    # we want to have input_frame between (start_frame-1) and (end_frame-2)
    # this is because at each step we read (frame) and (frame+1)
    # so the last iteration will actuall be (end_frame-1) and (end_frame)
    while input_frame < final_frame - 1:
        input_frame += 1

        start_time = time.time()

        filename_frame_1 = os.path.join(frames_dir, f'{input_frame:0>4d}.png')
        filename_frame_2 = os.path.join(frames_dir, f'{input_frame + 1:0>4d}.png')

        X0 = torch.from_numpy(np.transpose(imread(filename_frame_1), (2, 0, 1)).astype("float32") / 255.0)[:3,:,:].type(
            args.dtype)
        X1 = torch.from_numpy(np.transpose(imread(filename_frame_2), (2, 0, 1)).astype("float32") / 255.0)[:3,:,:].type(
            args.dtype)

        assert (X0.size(1) == X1.size(1))
        assert (X0.size(2) == X1.size(2))

        intWidth = X0.size(2)
        intHeight = X0.size(1)
        channels = X0.size(0)
        if not channels == 3:
            print(f"Skipping {filename_frame_1}-{filename_frame_2} -- expected 3 color channels but found {channels}.")
            continue

        if intWidth != ((intWidth >> 7) << 7):
            intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
            intPaddingLeft = int((intWidth_pad - intWidth) / 2)
            intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
        else:
            intPaddingLeft = 32
            intPaddingRight = 32

        if intHeight != ((intHeight >> 7) << 7):
            intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
            intPaddingTop = int((intHeight_pad - intHeight) / 2)
            intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
        else:
            intPaddingTop = 32
            intPaddingBottom = 32

        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])

        X0 = Variable(torch.unsqueeze(X0, 0))
        X1 = Variable(torch.unsqueeze(X1, 0))
        X0 = pader(X0)
        X1 = pader(X1)

        if args.use_cuda:
            X0 = X0.cuda()
            X1 = X1.cuda()

        y_s, offset, filter = model(torch.stack((X0, X1), dim=0))
        y_ = y_s[args.save_which]

        if args.use_cuda:
            X0 = X0.data.cpu().numpy()
            if not isinstance(y_, list):
                y_ = y_.data.cpu().numpy()
            else:
                y_ = [item.data.cpu().numpy() for item in y_]
            offset = [offset_i.data.cpu().numpy() for offset_i in offset]
            filter = [filter_i.data.cpu().numpy() for filter_i in filter] if filter[0] is not None else None
            X1 = X1.data.cpu().numpy()
        else:
            X0 = X0.data.numpy()
            if not isinstance(y_, list):
                y_ = y_.data.numpy()
            else:
                y_ = [item.data.numpy() for item in y_]
            offset = [offset_i.data.numpy() for offset_i in offset]
            filter = [filter_i.data.numpy() for filter_i in filter]
            X1 = X1.data.numpy()

        X0 = np.transpose(255.0 * X0.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                                  intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0))
        y_ = [np.transpose(255.0 * item.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                                   intPaddingLeft:intPaddingLeft + intWidth], (1, 2, 0)) for item in y_]
        offset = [np.transpose(
            offset_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
            (1, 2, 0)) for offset_i in offset]
        filter = [np.transpose(
            filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
            (1, 2, 0)) for filter_i in filter] if filter is not None else None
        X1 = np.transpose(255.0 * X1.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                                  intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0))

        interpolated_frame_number = 0
        shutil.copy(filename_frame_1,
                    os.path.join(output_dir, f"{input_frame:0>5d}{interpolated_frame_number:0>3d}.png"))
        for item, time_offset in zip(y_, time_offsets):
            interpolated_frame_number += 1
            output_frame_file_path = os.path.join(output_dir, f"{input_frame:0>5d}{interpolated_frame_number:0>3d}.png")
            imsave(output_frame_file_path, np.round(item).astype(np.uint8))

        end_time = time.time()
        loop_timer.update(end_time - start_time)

        frames_left = final_frame - input_frame
        estimated_seconds_left = frames_left * loop_timer.avg
        estimated_time_left = datetime.timedelta(seconds=estimated_seconds_left)
        print(
            f"****** Processed frame {input_frame} | Time per frame (avg): {loop_timer.avg:2.2f}s | Time left: {estimated_time_left} ******************")

    # Copying last frame
    last_frame_filename = os.path.join(frames_dir, str(str(final_frame).zfill(5)) + '.png')
    shutil.copy(last_frame_filename, os.path.join(output_dir, f"{final_frame:0>5d}{0:0>3d}.png"))

    print("Finished processing images.")
