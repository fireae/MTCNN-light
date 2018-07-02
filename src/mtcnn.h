#ifndef MTCNN_H
#define MTCNN_H
#include "network.h"

class Pnet {
 public:
  Pnet();
  ~Pnet();
  void run(Mat &image, float scale);

  float nms_threshold;
  Dtype Pthreshold;
  bool firstFlag;
  vector<struct Bbox> boundingBox_;
  vector<orderScore> bboxScore_;

 private:
  // the image for mxnet conv
  struct Blob *rgb;
  struct Blob *conv1_matrix;
  // the 1th layer's out conv
  struct Blob *conv1;
  struct Blob *maxPooling1;
  struct Blob *maxPooling_matrix;
  // the 3th layer's out
  struct Blob *conv2;
  struct Blob *conv3_matrix;
  // the 4th layer's out   out
  struct Blob *conv3;
  struct Blob *score_matrix;
  // the 4th layer's out   out
  struct Blob *score_;
  // the 4th layer's out   out
  struct Blob *location_matrix;
  struct Blob *location_;

  // Weight
  struct Weight *conv1_wb;
  struct pRelu *prelu_gmma1;
  struct Weight *conv2_wb;
  struct pRelu *prelu_gmma2;
  struct Weight *conv3_wb;
  struct pRelu *prelu_gmma3;
  struct Weight *conv4c1_wb;
  struct Weight *conv4c2_wb;

  void generateBbox(const struct Blob *score, const struct Blob *location,
                    Dtype scale);
};

class Rnet {
 public:
  Rnet();
  ~Rnet();
  float Rthreshold;
  void run(Mat &image);
  struct Blob *score_;
  struct Blob *location_;

 private:
  struct Blob *rgb;

  struct Blob *conv1_matrix;
  struct Blob *conv1_out;
  struct Blob *pooling1_out;

  struct Blob *conv2_matrix;
  struct Blob *conv2_out;
  struct Blob *pooling2_out;

  struct Blob *conv3_matrix;
  struct Blob *conv3_out;

  struct Blob *fc4_out;

  // Weight
  struct Weight *conv1_wb;
  struct pRelu *prelu_gmma1;
  struct Weight *conv2_wb;
  struct pRelu *prelu_gmma2;
  struct Weight *conv3_wb;
  struct pRelu *prelu_gmma3;
  struct Weight *fc4_wb;
  struct pRelu *prelu_gmma4;
  struct Weight *score_wb;
  struct Weight *location_wb;

  void RnetImage2MatrixInit(struct Blob *Blob);
};

class Onet {
 public:
  Onet();
  ~Onet();
  void run(Mat &image);
  float Othreshold;
  struct Blob *score_;
  struct Blob *location_;
  struct Blob *keyPoint_;

 private:
  struct Blob *rgb;
  struct Blob *conv1_matrix;
  struct Blob *conv1_out;
  struct Blob *pooling1_out;

  struct Blob *conv2_matrix;
  struct Blob *conv2_out;
  struct Blob *pooling2_out;

  struct Blob *conv3_matrix;
  struct Blob *conv3_out;
  struct Blob *pooling3_out;

  struct Blob *conv4_matrix;
  struct Blob *conv4_out;

  struct Blob *fc5_out;

  // Weight
  struct Weight *conv1_wb;
  struct pRelu *prelu_gmma1;
  struct Weight *conv2_wb;
  struct pRelu *prelu_gmma2;
  struct Weight *conv3_wb;
  struct pRelu *prelu_gmma3;
  struct Weight *conv4_wb;
  struct pRelu *prelu_gmma4;
  struct Weight *fc5_wb;
  struct pRelu *prelu_gmma5;
  struct Weight *score_wb;
  struct Weight *location_wb;
  struct Weight *keyPoint_wb;
  void OnetImage2MatrixInit(struct Blob *Blob);
};

class mtcnn {
 public:
  mtcnn(int row, int col);
  ~mtcnn();
  void findFace(Mat &image);

 private:
  Mat reImage;
  float nms_threshold[3];
  vector<float> scales_;
  Pnet *simpleFace_;
  vector<struct Bbox> firstBbox_;
  vector<struct orderScore> firstOrderScore_;
  Rnet refineNet;
  vector<struct Bbox> secondBbox_;
  vector<struct orderScore> secondBboxScore_;
  Onet outNet;
  vector<struct Bbox> thirdBbox_;
  vector<struct orderScore> thirdBboxScore_;
};

#endif