//
// Created by jeremy on 7/12/23.
//

#include "Awb.h"

Awb::Awb(std::string modelPath) {
    mWBSess = new Ort::Experimental::Session(mWBSessEnv, modelPath, mWBSessOp);
}

Awb::~Awb() {
    delete mWBSess;
}

f00arr Awb::kernelP(f00arr I) {
    f00arr r = xt::view(I, xt::all(), 0);
    r = xt::expand_dims(r, 1);
    f00arr g = xt::view(I, xt::all(), 1);
    g = xt::expand_dims(g, 1);
    f00arr b = xt::view(I, xt::all(), 2);
    b = xt::expand_dims(b, 1);
    f00arr ones = xt::ones<float>({int(I.shape(0)), 1});
    f00arr merge = xt::concatenate(xt::xtuple(r, g, b, r * g, r * b, g * b, r * r, g * g, b * b, r * g * b, ones), 1);
    merge = xt::cast<f00>(xt::cast<uint8_t>(xt::cast<uint32_t>(merge)));
    return merge;
}

std::tuple<f00arr,f00arr,f00arr>
Awb::getMappingFunc(f00arr image1, f00arr image2) {
    auto img1 = image1.reshape({-1, 3});
    auto img2 = image2.reshape({-1, 3});
    f00arr img2r = xt::view(img2, xt::all(), 0);
    f00arr img2g = xt::view(img2, xt::all(), 1);
    f00arr img2b = xt::view(img2, xt::all(), 2);
    f00arr img1k = kernelP(img1);
    f00arr mr, mg, mb;
    mr = std::get<0>(xt::linalg::lstsq(img1k, img2r));
    mg = std::get<0>(xt::linalg::lstsq(img1k, img2g));
    mb = std::get<0>(xt::linalg::lstsq(img1k, img2b));
    return std::make_tuple(mr, mg, mb);
}

cv::Mat Awb::applyMappingFunc(f00arr image, f00arr mr, f00arr mg, f00arr mb) {
    auto h = image.shape(0), w = image.shape(1), c = image.shape(2);
    auto img = image.reshape({-1, 3});
    auto imgK = kernelP(img);
    f00arr resR = xt::linalg::dot(imgK, mr);
    resR.reshape({h, w, 1});
    f00arr resG = xt::linalg::dot(imgK, mg);
    resG.reshape({h, w, 1});
    f00arr resB = xt::linalg::dot(imgK, mb);
    resB.reshape({h, w, 1});
    f00arr img2 = xt::concatenate(xt::xtuple(resB, resG, resR), 2);
    xt::masked_view(img2, img2 > 1.0) = 1.0;
    xt::masked_view(img2, img2 < 0.0) = 0.0;

    img2.reshape({h, w, c});
    img2 = img2 * 255.0;
    xt::xarray<uint8_t> temp = xt::cast<uint8_t>(img2);
    cv::Mat mat(h, w, CV_8UC3, temp.data(), 0);
    return mat.clone();
}

cv::Mat Awb::predict(cv::Mat img) {
    //--------------img preprocess--------------
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::Mat imgResized;
    cv::resize(img, imgResized, {int(std::round(img.cols / static_cast<float>(std::max(img.cols, img.rows)) * 656.0)),
                                 int(std::round(img.rows / static_cast<float>(std::max(img.cols, img.rows)) * 656.0))});
    auto w = imgResized.cols, h = imgResized.rows;
    int newW, newH;
    if (w % 16 == 0) {
        newW = w;
    } else {
        newW = w + 16 - w % 16;
    }
    if (h % 16 == 0) {
        newH = h;
    } else {
        newH = h + 16 - h % 16;
    }
    if (!(w == newW && h == newH)) {
        cv::resize(imgResized, imgResized, {newW, newH});
    }
    cv::Mat temp;
    imgResized.convertTo(temp, CV_32FC3);
    f00arr imgResizedArr(ADAPT_CV32F3C(temp));
    img.convertTo(temp, CV_32FC3);
    f00arr imgArr(ADAPT_CV32F3C(temp));

    f00arr chwImg = xt::transpose(imgResizedArr, {2, 0, 1});
    chwImg /= 255.0;
    //--------------img preprocess--------------

    //--------------predict--------------
    std::vector<Ort::Value> tensors;
    tensors.push_back(Ort::Experimental::Value::CreateTensor<f32>(
            reinterpret_cast<f32 *>(chwImg.data()), chwImg.size(), {
                    1,
                    int64_t(chwImg.shape(0)),
                    int64_t(chwImg.shape(1)),
                    int64_t(chwImg.shape(2))
            }
    ));
    auto netOut = mWBSess->Run(mWBSess->GetInputNames(), tensors, mWBSess->GetOutputNames());
    f00arr outAwb;
    for (size_t i = 0; i < netOut.size(); i++) {
        auto tensorShape = netOut[i].GetTensorTypeAndShapeInfo().GetShape();
        auto tensorData = netOut[i].GetTensorMutableData<f32>();
        outAwb = xt::cast<f00>(xt::adapt(
                (f32 *) tensorData,
                tensorShape[0] * tensorShape[1] * tensorShape[2] * tensorShape[3],
                xt::no_ownership(),
                std::vector<std::size_t>{
                        static_cast<std::size_t>(tensorShape[0]),
                        static_cast<std::size_t>(tensorShape[1]),
                        static_cast<std::size_t>(tensorShape[2]),
                        static_cast<std::size_t>(tensorShape[3]),
                }
        ));
    }
    outAwb = xt::squeeze(outAwb, 0);
    outAwb = xt::transpose(outAwb, {1, 2, 0});
    //--------------predict--------------

    //--------------img postprocess--------------
    auto [mr, mg, mb] = getMappingFunc(imgResizedArr, outAwb);
    auto balancedImg = applyMappingFunc(imgArr, mr, mg, mb);
    //--------------img postprocess--------------
    return balancedImg;
}