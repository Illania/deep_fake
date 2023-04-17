import os
import glob
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from util.videoswap_multispecific import video_swap
from utils.hasher import Hasher, HashAlgorithm


class SwapManager:

    def __init__(self, app, video_path, output_path, multispecific_dir):
        self.__app = app
        self.__options = self.__init_options(video_path, output_path, multispecific_dir)
        self.__model = self.__init_model(self.__options)
        self.__crop_size = self.__options.crop_size
        self.__multisepcific_dir = self.__options.multisepcific_dir
        self.__transformer_arcface = self.__init_transformer()

    def swap(self):
        options = self.__options
        source_specific_id_nonorm_list = self.__prepare_source_persons()
        target_id_norm_list = self.__prepare_target_persons()

        assert len(target_id_norm_list) == len(source_specific_id_nonorm_list), \
            "The number of images in source and target  directory must be same !!!"
        video_swap(options.video_path, target_id_norm_list, source_specific_id_nonorm_list, options.id_thres,
                   self.__model, self.__app, options.output_path, temp_results_dir=options.temp_path,
                   no_simswaplogo=options.no_simswaplogo,
                   use_mask=options.use_mask)

        Hasher.check_hash_equals(HashAlgorithm.SHA1, self.__options.output_path, '../demo/output.mp4')

    def __init_transformer(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __init_options(self, video_path, output_path, multispecific_dir):
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

    def __init_model(self, options):
        torch.nn.Module.dump_patches = True
        model = create_model(options)
        model.eval()
        return model

    def __prepare_source_persons(self):
        source_specific_id_nonorm_list = []
        source_path = os.path.join(self.__multisepcific_dir, 'SRC_*')
        source_specific_images_path = sorted(glob.glob(source_path))

        with torch.no_grad():
            for source_specific_image_path in source_specific_images_path:
                specific_person_whole = cv2.imread(source_specific_image_path)
                specific_person_align_crop, _ = self.__app.get(specific_person_whole, self.__crop_size)
                specific_person_align_crop_pil = Image.fromarray(
                    cv2.cvtColor(specific_person_align_crop[0], cv2.COLOR_BGR2RGB))
                specific_person = self.__transformer_arcface(specific_person_align_crop_pil)
                specific_person = specific_person.view(-1, specific_person.shape[0], specific_person.shape[1],
                                                       specific_person.shape[2])
                # convert numpy to tensor
                specific_person = specific_person.cpu()
                # create latent id
                specific_person_downsample = F.interpolate(specific_person, size=(112, 112))
                specific_person_id_nonorm = self.__model.netArc(specific_person_downsample)
                source_specific_id_nonorm_list.append(specific_person_id_nonorm.clone())

        return specific_person_id_nonorm

    def __prepare_target_persons(self):
        target_id_norm_list = []
        target_path = os.path.join(self.__multisepcific_dir, 'DST_*')
        target_images_path = sorted(glob.glob(target_path))

        for target_image_path in target_images_path:
            img_a_whole = cv2.imread(target_image_path)
            img_a_align_crop, _ = self.__app.get(img_a_whole, self.__crop_size)
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
            img_a = self.__transformer_arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
            # convert numpy to tensor
            img_id = img_id.cpu()
            # create latent id
            img_id_downsample = F.interpolate(img_id, size=(112, 112))
            latend_id = self.__model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)
            target_id_norm_list.append(latend_id.clone())

        return target_id_norm_list
