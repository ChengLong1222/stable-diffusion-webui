# import os

# import facexlib
# import gfpgan

# import modules.face_restoration
# from modules import paths, shared, devices, modelloader, errors

# model_dir = "GFPGAN"
# user_path = None
# model_path = os.path.join(paths.models_path, model_dir)
# model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
# have_gfpgan = False
# loaded_gfpgan_model = None


# def gfpgann():
#     global loaded_gfpgan_model
#     global model_path
#     if loaded_gfpgan_model is not None:
#         loaded_gfpgan_model.gfpgan.to(devices.device_gfpgan)
#         return loaded_gfpgan_model

#     if gfpgan_constructor is None:
#         return None

#     models = modelloader.load_models(model_path, model_url, user_path, ext_filter="GFPGAN")
#     if len(models) == 1 and models[0].startswith("http"):
#         model_file = models[0]
#     elif len(models) != 0:
#         latest_file = max(models, key=os.path.getctime)
#         model_file = latest_file
#     else:
#         print("Unable to load gfpgan model!")
#         return None
#     if hasattr(facexlib.detection.retinaface, 'device'):
#         facexlib.detection.retinaface.device = devices.device_gfpgan
#     model = gfpgan_constructor(model_path=model_file, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device=devices.device_gfpgan)
#     loaded_gfpgan_model = model

#     return model


# def send_model_to(model, device):
#     model.gfpgan.to(device)
#     model.face_helper.face_det.to(device)
#     model.face_helper.face_parse.to(device)


# def gfpgan_fix_faces(np_image):
#     model = gfpgann()
#     if model is None:
#         return np_image

#     send_model_to(model, devices.device_gfpgan)

#     np_image_bgr = np_image[:, :, ::-1]
#     cropped_faces, restored_faces, gfpgan_output_bgr = model.enhance(np_image_bgr, has_aligned=False, only_center_face=False, paste_back=True)
#     np_image = gfpgan_output_bgr[:, :, ::-1]

#     model.face_helper.clean_all()

#     if shared.opts.face_restoration_unload:
#         send_model_to(model, devices.cpu)

#     return np_image


# gfpgan_constructor = None


# def setup_model(dirname):
#     try:
#         os.makedirs(model_path, exist_ok=True)
#         from gfpgan import GFPGANer
#         from facexlib import detection, parsing  # noqa: F401
#         global user_path
#         global have_gfpgan
#         global gfpgan_constructor

#         load_file_from_url_orig = gfpgan.utils.load_file_from_url
#         facex_load_file_from_url_orig = facexlib.detection.load_file_from_url
#         facex_load_file_from_url_orig2 = facexlib.parsing.load_file_from_url

#         def my_load_file_from_url(**kwargs):
#             return load_file_from_url_orig(**dict(kwargs, model_dir=model_path))

#         def facex_load_file_from_url(**kwargs):
#             return facex_load_file_from_url_orig(**dict(kwargs, save_dir=model_path, model_dir=None))

#         def facex_load_file_from_url2(**kwargs):
#             return facex_load_file_from_url_orig2(**dict(kwargs, save_dir=model_path, model_dir=None))

#         gfpgan.utils.load_file_from_url = my_load_file_from_url
#         facexlib.detection.load_file_from_url = facex_load_file_from_url
#         facexlib.parsing.load_file_from_url = facex_load_file_from_url2
#         user_path = dirname
#         have_gfpgan = True
#         gfpgan_constructor = GFPGANer

#         class FaceRestorerGFPGAN(modules.face_restoration.FaceRestoration):
#             def name(self):
#                 return "GFPGAN"

#             def restore(self, np_image):
#                 return gfpgan_fix_faces(np_image)

#         shared.face_restorers.append(FaceRestorerGFPGAN())
#     except Exception:
#         errors.report("Error setting up GFPGAN", exc_info=True)



from __future__ import annotations

import logging
import os

import torch

from modules import (
    devices,
    errors,
    face_restoration,
    face_restoration_utils,
    modelloader,
    shared,
)

logger = logging.getLogger(__name__)
model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
model_download_name = "GFPGANv1.4.pth"
gfpgan_face_restorer: face_restoration.FaceRestoration | None = None


class FaceRestorerGFPGAN(face_restoration_utils.CommonFaceRestoration):
    def name(self):
        return "GFPGAN"

    def get_device(self):
        return devices.device_gfpgan

    def load_net(self) -> torch.Module:
        for model_path in modelloader.load_models(
            model_path=self.model_path,
            model_url=model_url,
            command_path=self.model_path,
            download_name=model_download_name,
            ext_filter=['.pth'],
        ):
            if 'GFPGAN' in os.path.basename(model_path):
                return modelloader.load_spandrel_model(
                    model_path,
                    device=self.get_device(),
                    expected_architecture='GFPGAN',
                ).model
        raise ValueError("No GFPGAN model found")

    def restore(self, np_image):
        def restore_face(cropped_face_t):
            assert self.net is not None
            return self.net(cropped_face_t, return_rgb=False)[0]

        return self.restore_with_helper(np_image, restore_face)


def gfpgan_fix_faces(np_image):
    if gfpgan_face_restorer:
        return gfpgan_face_restorer.restore(np_image)
    logger.warning("GFPGAN face restorer not set up")
    return np_image


def setup_model(dirname: str) -> None:
    global gfpgan_face_restorer

    try:
        face_restoration_utils.patch_facexlib(dirname)
        gfpgan_face_restorer = FaceRestorerGFPGAN(model_path=dirname)
        shared.face_restorers.append(gfpgan_face_restorer)
    except Exception:
        errors.report("Error setting up GFPGAN", exc_info=True)
