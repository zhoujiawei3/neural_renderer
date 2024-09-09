#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> forward_face_index_map_UV_cuda(
        at::Tensor faces,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor faces_inv,
        int image_size
        );

std::vector<at::Tensor> load_textures_cuda(
        at::Tensor image,
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor sampling_index_map,
        at::Tensor sampling_weight_map,
        at::Tensor texture_total_weight,
        at::Tensor is_update,
        int texture_wrapping,
        int use_bilinear,
        int image_size,
        float eps);



at::Tensor load_textures_backward_cuda(
        at::Tensor face_index_map,
        at::Tensor sampling_weight_map,
        at::Tensor sampling_index_map,
        at::Tensor texture_total_weight,
        at::Tensor grad_rgb_map,
        at::Tensor grad_textures,
        int num_faces);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> forward_face_index_map_UV(
        at::Tensor faces,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor faces_inv,
        int image_size) {

    CHECK_INPUT(faces);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(weight_map);
    CHECK_INPUT(faces_inv)


    return forward_face_index_map_UV_cuda(faces,face_index_map, weight_map, faces_inv, image_size);
}

std::vector<at::Tensor> load_textures(
        at::Tensor image,
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor face_index_map,
        at::Tensor weight_map,
        at::Tensor sampling_index_map,
        at::Tensor sampling_weight_map,
        at::Tensor texture_total_weight,
        at::Tensor is_update,

        int texture_wrapping,
        int use_bilinear,
        int image_size,
        float eps) {
    CHECK_INPUT(image);
    CHECK_INPUT(faces);
    CHECK_INPUT(textures);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(weight_map);
    CHECK_INPUT(sampling_index_map);
    CHECK_INPUT(sampling_weight_map);
    CHECK_INPUT(texture_total_weight);
    CHECK_INPUT(is_update);

    return load_textures_cuda(image,faces, textures, face_index_map,
                                    weight_map,
                                    sampling_index_map, sampling_weight_map,texture_total_weight,is_update,texture_wrapping,
                                     use_bilinear,image_size,eps);
}



at::Tensor load_textures_backward(
        at::Tensor face_index_map,
        at::Tensor sampling_weight_map,
        at::Tensor sampling_index_map,
        at::Tensor texture_total_weight,
        at::Tensor grad_output,
        at::Tensor grad_textures,
        int num_faces) {

    CHECK_INPUT(face_index_map);
    CHECK_INPUT(sampling_weight_map);
    CHECK_INPUT(sampling_index_map);
    CHECK_INPUT(texture_total_weight);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(grad_textures);

//     printf("nowrong\n");
// 打印1


    return load_textures_backward_cuda(face_index_map, sampling_weight_map,
                                  sampling_index_map, texture_total_weight,grad_output,
                                  grad_textures, num_faces);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_face_index_map_UV", &forward_face_index_map_UV, "forward_face_index_map_UV (CUDA)");
    m.def("load_textures", &load_textures, "load_textures (CUDA)");
    m.def("load_textures_backward", &load_textures_backward, "load_textures_backward (CUDA)");
}
