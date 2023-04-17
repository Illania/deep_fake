import cv2
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.videoswap_multispecific import video_swap
import os
import glob
from utils.hasher import Hasher, HashAlgorithm


def init_options(multispecific_dir, video_path, output_path):
    opt = TestOptions()
    opt.initialize()
    opt.gpu_ids = -1
    opt.parser.add_argument('-f')
    opt = opt.parse()
    opt.multisepcific_dir = multispecific_dir
    opt.video_path = video_path
    opt.output_path = output_path
    opt.temp_path = './tmp'
    opt.Arc_path = './arcface_model/arcface_checkpoint.tar'
    opt.isTrain = False
    opt.no_simswaplogo = True
    opt.name = 'people'
    opt.use_mask = True
    return opt


def init_model(opt):
    torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()
    return model


def init_face_detection_model(threshold, size):
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=threshold, det_size=(size, size))
    return app


def prepare_source_persons(model, app, transformer_arcface, multisepcific_dir, crop_size):
    # The specific person to be swapped(source)
    source_specific_id_nonorm_list = []
    source_path = os.path.join(multisepcific_dir, 'SRC_*')
    source_specific_images_path = sorted(glob.glob(source_path))

    with torch.no_grad():
        for source_specific_image_path in source_specific_images_path:
            specific_person_whole = cv2.imread(source_specific_image_path)
            specific_person_align_crop, _ = app.get(specific_person_whole, crop_size)
            specific_person_align_crop_pil = Image.fromarray(
                cv2.cvtColor(specific_person_align_crop[0], cv2.COLOR_BGR2RGB))
            specific_person = transformer_arcface(specific_person_align_crop_pil)
            specific_person = specific_person.view(-1, specific_person.shape[0], specific_person.shape[1],
                                                   specific_person.shape[2])
            # convert numpy to tensor
            specific_person = specific_person.cpu()
            # create latent id
            specific_person_downsample = F.interpolate(specific_person, size=(112, 112))
            specific_person_id_nonorm = model.netArc(specific_person_downsample)
            source_specific_id_nonorm_list.append(specific_person_id_nonorm.clone())

    return specific_person_id_nonorm


def prepare_target_persons(model, app, transformer_arcface, multisepcific_dir, crop_size):
    # The person who provides id information (list)
    target_id_norm_list = []
    target_path = os.path.join(multisepcific_dir, 'DST_*')
    target_images_path = sorted(glob.glob(target_path))

    for target_image_path in target_images_path:
        img_a_whole = cv2.imread(target_image_path)
        img_a_align_crop, _ = app.get(img_a_whole, crop_size)
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
        img_a = transformer_arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
        # convert numpy to tensor
        img_id = img_id.cpu()
        # create latent id
        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)
        target_id_norm_list.append(latend_id.clone())

    return target_id_norm_list


if __name__ == "__main__":
    os.chdir("SimSwap")
    transformer = transforms.Compose([
        transforms.ToTensor(),
    ])

    transformer_arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    detransformer = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
    ])
    options = init_options('../demo/multispecific',
                           '../demo/input.mp4',
                           '../demo/output/output4.mp4')

    crop_size = options.crop_size
    multisepcific_dir = options.multisepcific_dir

    model = init_model(options)
    app = init_face_detection_model(0.6, 640)

    source_specific_id_nonorm_list = prepare_source_persons(model, app, transformer_arcface, multisepcific_dir,
                                                            crop_size)
    target_id_norm_list = prepare_target_persons(model, app, transformer_arcface, multisepcific_dir, crop_size)

    assert len(target_id_norm_list) == len(
        source_specific_id_nonorm_list), "The number of images in source and target  directory must be same !!!"
    video_swap(options.video_path, target_id_norm_list, source_specific_id_nonorm_list, options.id_thres,
               model, app, options.output_path, temp_results_dir=options.temp_path,
               no_simswaplogo=options.no_simswaplogo,
               use_mask=options.use_mask)
    Hasher.check_hash_equals(HashAlgorithm.SHA1, '../demo/output/output3.mp4', '../demo/output/output4.mp4')

