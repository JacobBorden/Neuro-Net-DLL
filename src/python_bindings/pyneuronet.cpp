#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "neural_network/neuronet.h"
#include "math/matrix.h"

namespace py = pybind11;

PYBIND11_MODULE(pyneuronet, m) {
    m.doc() = "Python bindings for NeuroNet";

    py::enum_<NeuroNet::ActivationFunctionType>(m, "ActivationFunctionType")
        .value("None", NeuroNet::ActivationFunctionType::None)
        .value("ReLU", NeuroNet::ActivationFunctionType::ReLU)
        .value("LeakyReLU", NeuroNet::ActivationFunctionType::LeakyReLU)
        .value("ELU", NeuroNet::ActivationFunctionType::ELU)
        .value("Softmax", NeuroNet::ActivationFunctionType::Softmax)
        .value("Sigmoid", NeuroNet::ActivationFunctionType::Sigmoid)
        .value("Tanh", NeuroNet::ActivationFunctionType::Tanh)
        .value("Swish", NeuroNet::ActivationFunctionType::Swish)
        .export_values();

    py::class_<Matrix::Matrix<float>>(m, "Matrix")
        .def(py::init<int, int>())
        .def("rows", &Matrix::Matrix<float>::rows)
        .def("cols", &Matrix::Matrix<float>::cols)
        .def("get", [](const Matrix::Matrix<float>& mat, int r, int c) { return mat[r][c]; })
        .def("set", [](Matrix::Matrix<float>& mat, int r, int c, float v) { mat[r][c] = v; })
        .def(py::init([](const std::vector<std::vector<float>>& data) {
            int rows = data.size();
            int cols = rows > 0 ? data[0].size() : 0;
            Matrix::Matrix<float> mat(rows, cols);
            for(int r = 0; r < rows; ++r) {
                for(int c = 0; c < cols; ++c) {
                    mat[r][c] = data[r][c];
                }
            }
            return mat;
        }))
        .def("to_list", [](const Matrix::Matrix<float>& mat) {
            std::vector<std::vector<float>> data(mat.rows(), std::vector<float>(mat.cols()));
            for(size_t r = 0; r < mat.rows(); ++r) {
                for(size_t c = 0; c < mat.cols(); ++c) {
                    data[r][c] = mat[r][c];
                }
            }
            return data;
        });

    py::class_<NeuroNet::NeuroNet>(m, "NeuroNet")
        .def(py::init<>())
        .def(py::init<int>())
        .def("resize_layer", &NeuroNet::NeuroNet::ResizeLayer)
        .def("set_input_size", &NeuroNet::NeuroNet::SetInputSize)
        .def("get_layer_count", &NeuroNet::NeuroNet::getLayerCount)
        .def("resize_neuro_net", &NeuroNet::NeuroNet::ResizeNeuroNet)
        .def("get_input_size", &NeuroNet::NeuroNet::GetInputSize)
        .def("set_activation_function", [](NeuroNet::NeuroNet& nn, int index, NeuroNet::ActivationFunctionType act) {
            nn.getLayer(index).SetActivationFunction(act);
        })
        .def("set_input", &NeuroNet::NeuroNet::SetInput)
        .def("get_output", &NeuroNet::NeuroNet::GetOutput)
        .def("train", &NeuroNet::NeuroNet::Train)
        .def("save_model", &NeuroNet::NeuroNet::save_model)
        .def_static("load_model", &NeuroNet::NeuroNet::load_model);
}
