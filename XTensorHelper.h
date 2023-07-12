//
// Created by jeremy on 7/12/23.
//

#ifndef DEEP_WHITE_BALANCE_XTENSORHELPER_H
#define DEEP_WHITE_BALANCE_XTENSORHELPER_H
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xio.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <opencv2/opencv.hpp>

#define ADAPT_CV8U3C(x) (xt::adapt(x.ptr<uchar>(),x.cols*x.rows*x.channels(),xt::no_ownership(),std::vector<std::size_t> {static_cast<std::size_t>(x.rows),static_cast<std::size_t>(x.cols),static_cast<std::size_t>(x.channels())}))
#define ADAPT_CV32F3C(x) (xt::adapt(x.ptr<float>(),x.cols*x.rows*x.channels(),xt::no_ownership(),std::vector<std::size_t> {static_cast<std::size_t>(x.rows),static_cast<std::size_t>(x.cols),static_cast<std::size_t>(x.channels())}))
#define f32 float
#define f64 double
#define f32arr xt::xarray<f32>
#define f64arr xt::xarray<f64>
#define f00 f32
#define f00arr xt::xarray<f00>
#define PRINT_SHAPE(x) for(auto t : x.shape()) std::cout << t << " "; std::cout <<std::endl;

#endif //DEEP_WHITE_BALANCE_XTENSORHELPER_H
