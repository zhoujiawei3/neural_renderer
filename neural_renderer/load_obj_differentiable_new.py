from __future__ import division
import os

import torch
import numpy as np
from skimage.io import imread
from PIL import Image

import neural_renderer.cuda.load_textures_totalUV as load_textures_totalUV_cuda

# import neural_renderer.cuda.load_textures_backward_new as load_textures_cuda_backward_new

#导入function
from torch.autograd import Function
texture_wrapping_dict = {'REPEAT': 0, 'MIRRORED_REPEAT': 1,
                         'CLAMP_TO_EDGE': 2, 'CLAMP_TO_BORDER': 3}
class LoadTextures(Function):
    @staticmethod
    def forward(ctx, image, faces, textures, is_update, texture_wrapping_number, use_bilinear):
        # 保存反向传播所需的输入张量
        # ctx.save_for_backward(image, faces, textures, is_update)
        textures_origin=torch.zeros_like(textures)
        ctx.texture_wrapping_number = texture_wrapping_number
        ctx.use_bilinear = use_bilinear
        #得到image_size
        ctx.image_size=image.shape[1]
        ctx.face_num=faces.shape[0]
        ctx.texture_size=textures.shape[1]
        faces_inv = torch.zeros(faces.shape[0], 3, 3).cuda()
        face_index_map = torch.cuda.IntTensor(ctx.image_size, ctx.image_size).fill_(-1)
        weight_map = torch.cuda.FloatTensor(ctx.image_size, ctx.image_size, 3).fill_(0.0)
        sampling_index_map = torch.cuda.IntTensor(ctx.image_size, ctx.image_size, 8).fill_(0) # 渲染图的某一个像素点对应的点采样
        sampling_weight_map = torch.cuda.FloatTensor(ctx.image_size, ctx.image_size, 8).fill_(0.0) #采样权重
        texture_total_weight = torch.cuda.FloatTensor(ctx.face_num, ctx.texture_size,ctx.texture_size,ctx.texture_size).fill_(0.0)


        face_index_map, weight_map=\
           load_textures_totalUV_cuda.forward_face_index_map_UV(faces,face_index_map, weight_map,faces_inv,ctx.image_size)
        
        textures_origin, sampling_index_map, sampling_weight_map,texture_total_weight=\
            load_textures_totalUV_cuda.load_textures(image,faces,textures_origin,face_index_map, weight_map,sampling_index_map,
                                                           sampling_weight_map,texture_total_weight,is_update, texture_wrapping_number, use_bilinear,ctx.image_size,1e-4)
        

        print(sampling_index_map.shape)
        

        texture_total_weight=1.0/(texture_total_weight+1e-9)
        texture_total_weight_expand=texture_total_weight.unsqueeze(-1).expand(-1,-1,-1,-1,3)

        textures_origin=textures_origin*texture_total_weight_expand
        ctx.save_for_backward(image, faces, textures_origin, is_update,face_index_map,weight_map,sampling_index_map,sampling_weight_map,texture_total_weight)
        # 调用CUDA函数
        return textures_origin
    
    @staticmethod
    def backward(ctx, grad_output):
        image, faces, textures_origin, is_update,face_index_map,weight_map,sampling_index_map,sampling_weight_map,texture_total_weight = ctx.saved_tensors
        texture_wrapping_number = ctx.texture_wrapping_number
        use_bilinear = ctx.use_bilinear
        grad_image = torch.zeros_like(image)

        grad_image=load_textures_totalUV_cuda.load_textures_backward(face_index_map,sampling_weight_map,sampling_index_map,texture_total_weight,grad_image,grad_output,ctx.face_num)

        return grad_image,None, None, None, None, None

def load_mtl(filename_mtl):
    '''
    load color (Kd) and filename of textures from *.mtl
    '''
    texture_filenames = {}
    colors = {}
    material_name = ''
    with open(filename_mtl) as f:
        for line in f.readlines():
            if len(line.split()) != 0:
                if line.split()[0] == 'newmtl':
                    material_name = line.split()[1]
                if line.split()[0] == 'map_Kd':
                    texture_filenames[material_name] = line.split()[1]
                if line.split()[0] == 'Kd':
                    colors[material_name] = np.array(list(map(float, line.split()[1:4])))
    return colors, texture_filenames


def load_textures(filename_obj, filename_mtl, texture_size, texture_wrapping='REPEAT', use_bilinear=True):
    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'vt':
            vertices.append([float(v) for v in line.split()[1:3]])
    vertices = np.vstack(vertices).astype(np.float32)

    # load faces for textures
    faces = []
    material_names = []
    material_name = ''
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            if '/' in vs[0] and '//' not in vs[0]:
                v0 = int(vs[0].split('/')[1])
            else:
                v0 = 0
            for i in range(nv - 2):
                if '/' in vs[i + 1] and '//' not in vs[i + 1]:
                    v1 = int(vs[i + 1].split('/')[1])
                else:
                    v1 = 0
                if '/' in vs[i + 2] and '//' not in vs[i + 2]:
                    v2 = int(vs[i + 2].split('/')[1])
                else:
                    v2 = 0
                faces.append((v0, v1, v2))
                material_names.append(material_name)
        if line.split()[0] == 'usemtl':
            material_name = line.split()[1]
    faces = np.vstack(faces).astype(np.int32) - 1
    faces = vertices[faces] #这种方式很方便的让faces变成[  [ [v0,v1] [v0,v1] [v0,v1] ],]
    faces = torch.from_numpy(faces).cuda()

    colors, texture_filenames = load_mtl(filename_mtl)

    textures = torch.zeros(faces.shape[0], texture_size, texture_size, texture_size, 3, dtype=torch.float32) + 0.5
    textures = textures.cuda()

    #
    for material_name, color in colors.items():
        color = torch.from_numpy(color).cuda()
        for i, material_name_f in enumerate(material_names):
            if material_name == material_name_f:
                textures[i, :, :, :, :] = color[None, None, None, :]

    for material_name, filename_texture in texture_filenames.items():
        filename_texture = os.path.join(os.path.dirname(filename_obj), filename_texture)
        image = imread(filename_texture).astype(np.float32) / 255.

        # texture image may have one channel (grey color)
        if len(image.shape) == 2:
            image = np.stack((image,)*3,-1)
        # or has extral alpha channel shoule ignore for now
        if image.shape[2] == 4:
            image = image[:,:,:3]

        # pytorch does not support negative slicing for the moment
        image = image[::-1, :, :] 
        image = torch.from_numpy(image.copy()).cuda()
        is_update = (np.array(material_names) == material_name).astype(np.int32) #与np结构material_names，每个面有一个
        is_update = torch.from_numpy(is_update).cuda()
        # textures = load_textures_cuda.load_textures(image, faces, textures, is_update,
        #                                             texture_wrapping_dict[texture_wrapping],
        #                                             use_bilinear)
        textures=LoadTextures.apply(image, faces, textures, is_update, texture_wrapping_dict[texture_wrapping], use_bilinear)
    return textures,image,faces,is_update,texture_wrapping_dict[texture_wrapping],use_bilinear

def load_obj_totalUV(filename_obj, normalization=True, texture_size=4, load_texture=False,
             texture_wrapping='REPEAT', use_bilinear=True):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).

    
    """

    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = torch.from_numpy(np.vstack(vertices).astype(np.float32)).cuda()

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = torch.from_numpy(np.vstack(faces).astype(np.int32)).cuda() - 1

    # load textures
    textures = None
    if load_texture:
        for line in lines:
            if line.startswith('mtllib'):
                filename_mtl = os.path.join(os.path.dirname(filename_obj), line.split()[1])
                textures,image,faces_3_2,is_update,wrap_way,bilinear_way = load_textures(filename_obj, filename_mtl, texture_size,
                                         texture_wrapping=texture_wrapping,
                                         use_bilinear=use_bilinear)
        if textures is None:
            raise Exception('Failed to load textures.')

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[0][None, :]
        vertices /= torch.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[0][None, :] / 2

    if load_texture:
        return vertices, faces, textures,image,faces_3_2,is_update,wrap_way,bilinear_way
    else:
        return vertices, faces

