// c++ network  author : liqi
// Nangjing University of Posts and Telecommunications
// date 2017.5.21,20:27
#ifndef NETWORK_H
#define NETWORK_H
#include <math.h>
#include <memory.h>
#include <stdlib.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include "Blob.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


using namespace cv;

void addbias(struct Blob *blob, Dtype *pbias);
void image2Matrix(const Mat &image, const struct Blob *blob);
void featurePad(const Blob *bob, const Blob *outBlob, const int pad);
void feature2Matrix(const Blob *blob, Blob *Matrix, const Weight *weight);
void maxPooling(const Blob *blob, Blob *Matrix, int kernelSize, int stride);
void relu(struct Blob *Blob, Dtype *pbias);
void prelu(struct Blob *Blob, Dtype *pbias, Dtype *prelu_gmma);
void convolution(const Weight *weight, const Blob *blob, Blob *outBlob,
                 const struct Blob *matrix);
void fullconnect(const Weight *weight, const Blob *blob, Blob *outBlob);
void readData(string filename, long dataNumber[], Dtype *pTeam[]);
void readData(const float *datasource, int sourcelen, long dataNumber[],
              Dtype *pTeam[], int num);
long initConvAndFc(struct Weight *weight, int schannel, int lchannel,
                   int kersize, int stride, int pad);
void initpRelu(struct pRelu *prelu, int width);
void softmax(const struct Blob *Blob);

void image2MatrixInit(Mat &image, struct Blob *Blob);
void featurePadInit(const Blob *blob, Blob *outBlob, const int pad);
void maxPoolingInit(const Blob *blob, Blob *Matrix, int kernelSize, int stride);
void feature2MatrixInit(const Blob *blob, Blob *Matrix, const Weight *weight);
void convolutionInit(const Weight *weight, const Blob *blob, Blob *outBlob,
                     const struct Blob *matrix);
void fullconnectInit(const Weight *weight, Blob *outBlob);

bool cmpScore(struct orderScore lsh, struct orderScore rsh);
void nms(vector<struct Bbox> &boundingBox_,
         vector<struct orderScore> &bboxScore_, const float overlap_threshold,
         string modelname = "Union");
void refineAndSquareBbox(vector<struct Bbox> &vecBbox, const int &height,
                         const int &width);

#endif