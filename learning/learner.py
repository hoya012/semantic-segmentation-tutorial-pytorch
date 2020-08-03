from helpers.helpers import AverageMeter, ProgressMeter, iouCalc, visim, vislbl
from learning.minicity import MiniCity
from learning.utils import rand_bbox, copyblob
import torch
import torch.nn.functional as F
import cv2
import os
import numpy as np
import time
from PIL import Image

"""
=================
Routine functions
=================
"""

def train_epoch(dataloader, model, criterion, optimizer, lr_scheduler, epoch, void=-1, args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Accuracy', ':.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, loss_running, acc_running],
        prefix="Train, epoch: [{}]".format(epoch))
    
    # input resolution
    if args.crop_size is not None:
        res = args.crop_size[0]*args.crop_size[1]
    else:
        res = args.train_size[0]*args.train_size[1]
    
    # Set model in training mode
    model.train()
    
    end = time.time()
    
    with torch.set_grad_enabled(True):
        # Iterate over data.
        for epoch_step, (inputs, labels, _) in enumerate(dataloader):
            data_time.update(time.time()-end)
            
            if args.copyblob:
                for i in range(inputs.size()[0]):
                    rand_idx = np.random.randint(inputs.size()[0])
                    # wall(3) --> sidewalk(1)
                    copyblob(src_img=inputs[i], src_mask=labels[i], dst_img=inputs[rand_idx], dst_mask=labels[rand_idx], src_class=3, dst_class=1)
                    # fence(4) --> sidewalk(1)
                    copyblob(src_img=inputs[i], src_mask=labels[i], dst_img=inputs[rand_idx], dst_mask=labels[rand_idx], src_class=4, dst_class=1)
                    # bus(15) --> road(0)
                    copyblob(src_img=inputs[i], src_mask=labels[i], dst_img=inputs[rand_idx], dst_mask=labels[rand_idx], src_class=15, dst_class=0)
                    # train(16) --> road(0)
                    copyblob(src_img=inputs[i], src_mask=labels[i], dst_img=inputs[rand_idx], dst_mask=labels[rand_idx], src_class=16, dst_class=0)

            inputs = inputs.float().cuda()
            labels = labels.long().cuda()
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            if args.cutmix:
                # generate mixed sample
                lam = np.random.beta(1., 1.)
                rand_index = torch.randperm(inputs.size()[0]).cuda()
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                labels[:, bbx1:bbx2, bby1:bby2] = labels[rand_index, bbx1:bbx2, bby1:bby2]

            # forward pass
            outputs = model(inputs)
            outputs = outputs['out'] #FIXME for DeepLab V3
            preds = torch.argmax(outputs, 1)
            # cross-entropy loss
            loss = criterion(outputs, labels) 

            # backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            bs = inputs.size(0) # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)
            corrects = torch.sum(preds == labels.data)
            nvoid = int((labels==void).sum())
            acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
            acc_running.update(acc, bs)
            
            # output training info
            progress.display(epoch_step)
            
            # Measure time
            batch_time.update(time.time() - end)
            end = time.time()

        # Reduce learning rate
        lr_scheduler.step(loss_running.avg)
        
    return loss_running.avg, acc_running.avg

    
def validate_epoch(dataloader, model, criterion, epoch, classLabels, validClasses, void=-1, maskColors=None, folder='baseline_run', args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Accuracy', ':.4e')
    iou = iouCalc(classLabels, validClasses, voidClass = void)
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, loss_running, acc_running],
        prefix="Test, epoch: [{}]".format(epoch))
    
    # input resolution
    res = args.test_size[0]*args.test_size[1]
    
    # Set model in evaluation mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for epoch_step, (inputs, labels, filepath) in enumerate(dataloader):
            data_time.update(time.time()-end)
            
            inputs = inputs.float().cuda()
            labels = labels.long().cuda()
    
            # forward
            outputs = model(inputs)
            outputs = outputs['out'] #FIXME
            preds = torch.argmax(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            bs = inputs.size(0) # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)
            corrects = torch.sum(preds == labels.data)
            nvoid = int((labels==void).sum())
            acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
            acc_running.update(acc, bs)
            # Calculate IoU scores of current batch
            iou.evaluateBatch(preds, labels)
            
            # Save visualizations of first batch
            if epoch_step == 0 and maskColors is not None:
                for i in range(inputs.size(0)):
                    filename = os.path.splitext(os.path.basename(filepath[i]))[0]
                    # Only save inputs and labels once
                    if epoch == 0:
                        img = visim(inputs[i,:,:,:], args)
                        label = vislbl(labels[i,:,:], maskColors)
                        if len(img.shape) == 3:
                            cv2.imwrite(folder + '/images/{}.png'.format(filename),img[:,:,::-1])
                        else: 
                            cv2.imwrite(folder + '/images/{}.png'.format(filename),img)
                        cv2.imwrite(folder + '/images/{}_gt.png'.format(filename),label[:,:,::-1])
                    # Save predictions
                    pred = vislbl(preds[i,:,:], maskColors)
                    cv2.imwrite(folder + '/images/{}_epoch_{}.png'.format(filename,epoch),pred[:,:,::-1])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # print progress info
            progress.display(epoch_step)
        
        miou = iou.outputScores()
        print('Accuracy      : {:5.3f}'.format(acc_running.avg))
        print('---------------------')

    return acc_running.avg, loss_running.avg, miou

def predict(dataloader, model, maskColors, folder='baseline_run', mode='val', args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time],
        prefix='Predict: ')
    
    Dataset = MiniCity

    # Set model in evaluation mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for epoch_step, batch in enumerate(dataloader):

            if len(batch) == 2:
                inputs, filepath = batch
            else:
                inputs, _, filepath = batch

            data_time.update(time.time()-end)
            
            inputs = inputs.float().cuda()

            if args.mst:
                batch_idx, _, h, w = inputs.size() #(1, 20, 1024, 2048)
                # only single image is supported for multi-scale testing
                assert(batch_idx == 1)
                with torch.cuda.device_of(inputs):
                    scores = inputs.new().resize_(batch_idx, len(Dataset.validClasses), h, w).zero_().cuda()

                scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2] #FIXME

                for scale in scales:
                    inputs_resized = F.interpolate(inputs, scale_factor=scale, mode='bilinear', align_corners=True)
                    #print("original size {}x{} --> resized to {}x{}".format(h, w, inputs_resized.size()[2], inputs_resized.size()[3]))

                    # forward
                    outputs = model(inputs_resized)
                    outputs = outputs['out'] #FIXME (1, 20, 512, 1024) for scale 0.5

                    score = F.interpolate(outputs, (h, w), mode='bilinear', align_corners=True)
                    scores += score

                    
                    # forward using flipped input
                    with torch.cuda.device_of(inputs_resized):
                        idx = torch.arange(inputs_resized.size(3)-1, -1, -1).type_as(inputs_resized).long()
                    input_resized_flip = inputs_resized.index_select(3, idx)

                    # forward
                    outputs = model(input_resized_flip)
                    outputs = outputs['out'] #FIXME
                    outputs = outputs.index_select(3, idx)

                    score = F.interpolate(outputs, (h, w), mode='bilinear', align_corners=True)
                    scores += score

                # averaging scores
                scores = scores / (2*len(scales))

                preds = torch.argmax(scores, 1) # (1, 512, 1024) 
            else:
                # forward
                outputs = model(inputs)
                outputs = outputs['out'] #FIXME

                preds = torch.argmax(outputs, 1)
            
            # Save visualizations of first batch
            for i in range(inputs.size(0)):
                filename = os.path.splitext(os.path.basename(filepath[i]))[0]
                # Save input
                img = visim(inputs[i,:,:,:], args)
                img = Image.fromarray(img, 'RGB')
                img.save(folder + '/results_color_{}/{}_input.png'.format(mode, filename))
                # Save prediction with color labels
                pred = preds[i,:,:].cpu()
                pred_color = vislbl(pred, maskColors)
                pred_color = Image.fromarray(pred_color.astype('uint8'))
                pred_color.save(folder + '/results_color_{}/{}_prediction.png'.format(mode, filename))
                # Save class id prediction (used for evaluation)
                pred_id = Dataset.trainid2id[pred]
                pred_id = Image.fromarray(pred_id)
                pred_id = pred_id.resize((2048,1024), resample=Image.NEAREST)
                pred_id.save(folder + '/results_{}/{}.png'.format(mode, filename))
            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # print progress info
            progress.display(epoch_step)
