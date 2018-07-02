#include "network.h"
#include <Eigen/Eigen>
#include "gemm.h"

#define MAP_SVECTOR(name, ptr, N) Eigen::Map<Eigen::VectorXf> name(ptr, N)
#define MAP_CONST_SVECTOR(name, ptr, N) \
  Eigen::Map<const Eigen::VectorXf> name(ptr, N)

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    MatXf;

#define MAP_SMATRIX(name, ptr, M, N) Eigen::Map<MatXf> name(ptr, M, N)
#define MAP_CONST_SMATRIX(name, ptr, M, N) \
  Eigen::Map<const MatXf> name(ptr, M, N)

void addbias(struct Blob *Blob, Dtype *pbias) {
  if (Blob->pdata == NULL) {
    cout << "Relu feature is NULL!!" << endl;
    return;
  }
  if (pbias == NULL) {
    cout << "the  Relu bias is NULL!!" << endl;
    return;
  }
  Dtype *op = Blob->pdata;
  Dtype *pb = pbias;

  long dis = Blob->width * Blob->height;
  for (int channel = 0; channel < Blob->channel; channel++) {
    for (int col = 0; col < dis; col++) {
      *op = *op + *pb;
      op++;
    }
    pb++;
  }
}
void image2MatrixInit(Mat &image, struct Blob *Blob) {
  if ((image.data == NULL) || (image.type() != CV_8UC3)) {
    cout << "image's type is wrong!!Please set CV_8UC3" << endl;
    return;
  }
  Blob->channel = image.channels();
  Blob->height = image.rows;
  Blob->width = image.cols;

  Blob->pdata = (Dtype *)malloc(Blob->channel * Blob->height * Blob->width *
                                sizeof(Dtype));
  if (Blob->pdata == NULL) cout << "the image2MatrixInit failed!!" << endl;
  memset(Blob->pdata, 0,
         Blob->channel * Blob->height * Blob->width * sizeof(Dtype));
}
void image2Matrix(const Mat &image, const struct Blob *Blob) {
  if ((image.data == NULL) || (image.type() != CV_8UC3)) {
    cout << "image's type is wrong!!Please set CV_8UC3" << endl;
    return;
  }
  if (Blob->pdata == NULL) {
    return;
  }
  Dtype *p = Blob->pdata;
  for (int rowI = 0; rowI < image.rows; rowI++) {
    for (int colK = 0; colK < image.cols; colK++) {
      *p = (image.at<Vec3b>(rowI, colK)[0] - 127.5) *
           0.0078125;  // opencvµÄÍ¨µÀÅÅÐòÊÇRGB
      *(p + image.rows * image.cols) =
          (image.at<Vec3b>(rowI, colK)[1] - 127.5) * 0.0078125;
      *(p + 2 * image.rows * image.cols) =
          (image.at<Vec3b>(rowI, colK)[2] - 127.5) * 0.0078125;
      p++;
    }
  }
}
void featurePadInit(const Blob *blob, Blob *outBlob, const int pad) {
  if (pad <= 0) {
    cout << "the data needn't to pad,please check you network!" << endl;
    return;
  }
  outBlob->channel = blob->channel;
  outBlob->height = blob->height + 2 * pad;
  outBlob->width = blob->width + 2 * pad;
  long RowByteNum = outBlob->width * sizeof(Dtype);
  outBlob->pdata =
      (Dtype *)malloc(outBlob->channel * outBlob->height * RowByteNum);
  if (outBlob->pdata == NULL) cout << "the featurePadInit is failed!!" << endl;
  memset(outBlob->pdata, 0, outBlob->channel * outBlob->height * RowByteNum);
}
void featurePad(const Blob *blob, const Blob *outBlob, const int pad) {
  Dtype *p = outBlob->pdata;
  Dtype *pIn = blob->pdata;

  for (int row = 0; row < outBlob->channel * outBlob->height; row++) {
    if ((row % outBlob->height) < pad ||
        (row % outBlob->height > (outBlob->height - pad - 1))) {
      p += outBlob->width;
      continue;
    }
    p += pad;
    memcpy(p, pIn, blob->width * sizeof(Dtype));
    p += blob->width + pad;
    pIn += blob->width;
  }
}
void feature2MatrixInit(const Blob *blob, Blob *Matrix, const Weight *weight) {
  int kernelSize = weight->kernelSize;
  int stride = weight->stride;
  int w_out = (blob->width - kernelSize) / stride +
              1;  //Õâ¸ö¹«Ê½Ò»¶¨Òª¸ãÇå³þ£¬¿ÉÒÔ×Ô¼ºÈ¥»­¸ö¾ØÕó¿´¿´
  int h_out = (blob->height - kernelSize) / stride + 1;
  Matrix->width = blob->channel * kernelSize * kernelSize;  //Î´×ªÖÃÇ°µÄ¿í¶È
  Matrix->height = w_out * h_out;
  Matrix->channel = 1;
  Matrix->pdata =
      (Dtype *)malloc(Matrix->width * Matrix->height * sizeof(Dtype));
  if (Matrix->pdata == NULL) cout << "the feature2MatrixInit failed!!" << endl;
  memset(Matrix->pdata, 0, Matrix->width * Matrix->height * sizeof(Dtype));
}
void feature2Matrix(const Blob *blob, Blob *Matrix, const Weight *weight) {
  if (blob->pdata == NULL) {
    cout << "the feature2Matrix Blob is NULL!!" << endl;
    return;
  }
  int kernelSize = weight->kernelSize;
  int stride = weight->stride;
  int w_out = (blob->width - kernelSize) / stride +
              1;  //Õâ¸ö¹«Ê½Ò»¶¨Òª¸ãÇå³þ£¬¿ÉÒÔ×Ô¼ºÈ¥»­¸ö¾ØÕó¿´¿´
  int h_out = (blob->height - kernelSize) / stride + 1;

  Dtype *p = Matrix->pdata;
  Dtype *pIn;
  Dtype *ptemp;
  for (int row = 0; row < h_out; row++) {
    for (int col = 0; col < w_out; col++) {
      pIn = blob->pdata + row * stride * blob->width + col * stride;

      for (int channel = 0; channel < blob->channel; channel++) {
        ptemp = pIn + channel * blob->height * blob->width;
        for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
          memcpy(p, ptemp, kernelSize * sizeof(Dtype));
          p += kernelSize;
          ptemp += blob->width;
        }
      }
    }
  }
}
void convolutionInit(const Weight *weight, const Blob *blob, Blob *outBlob,
                     const struct Blob *matrix) {
  outBlob->channel = weight->selfChannel;
  outBlob->width = (blob->width - weight->kernelSize) / weight->stride + 1;
  outBlob->height = (blob->height - weight->kernelSize) / weight->stride + 1;
  outBlob->pdata =
      (Dtype *)malloc(weight->selfChannel * matrix->height * sizeof(Dtype));
  if (outBlob->pdata == NULL) cout << "the convolutionInit is failed!!" << endl;
  memset(outBlob->pdata, 0,
         weight->selfChannel * matrix->height * sizeof(Dtype));
}

void convolution(const Weight *weight, const Blob *blob, Blob *outBlob,
                 const struct Blob *matrix) {
  if (blob->pdata == NULL) {
    cout << "the feature is NULL!!" << endl;
    return;
  }
  if (weight->pdata == NULL) {
    cout << "the weight is NULL!!" << endl;
    return;
  }

  // MAP_SMATRIX(eC, outBlob->pdata, weight->selfChannel, matrix->height);
  if (weight->pad == 0) {
    // C←αAB + βC
    //                1              2            3              4     C's size
    //                5              k     alpha     A*              A'col
    //                B*           B'col    beta      C*           C'col
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, weight->selfChannel,
    //             matrix->height, matrix->width, 1, weight->pdata,
    //             matrix->width, matrix->pdata, matrix->width, 0,
    //             outBlob->pdata, matrix->height);
    gemm(0, 1, weight->selfChannel, matrix->height, matrix->width, 1.0,
         weight->pdata, matrix->width, matrix->pdata, matrix->width, 0.0,
         outBlob->pdata, matrix->height);
    // MAP_CONST_SMATRIX(eA, weight->pdata, weight->selfChannel, matrix->width);
    // MAP_CONST_SMATRIX(eB, matrix->pdata, matrix->height, matrix->width);
    // eC.noalias() += eA * eB.transpose();
    for (int i = 0; i < 5; i++) {
      printf("weight %f\n", weight->pdata[i]);
    }
    for (int i = 0; i < 5; i++) {
      printf("matrix %f\n", matrix->pdata[i]);
    }
    for (int i = 0; i < 5; i++) {
      printf("outBlob %f\n", outBlob->pdata[i]);
    }
  } else {
    struct Blob *padBlob = new Blob;
    featurePad(blob, padBlob, weight->pad);
    // C←αAB + βC
    //                1              2            3              4     C's size
    //                5              k     alpha     A*              A'col
    //                B*           B'col    beta      C*           C'col
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, weight->selfChannel,
    //             matrix->height, matrix->width, 1, weight->pdata,
    //             matrix->width, matrix->pdata, matrix->width, 0,
    //             outBlob->pdata, matrix->height);
    gemm(0, 1, weight->selfChannel, matrix->height, matrix->width, 1,
         weight->pdata, matrix->width, matrix->pdata, matrix->width, 0,
         outBlob->pdata, matrix->height);
    // MAP_CONST_SMATRIX(eA, weight->pdata, weight->selfChannel, matrix->width);
    // MAP_CONST_SMATRIX(eB, matrix->pdata, matrix->height, matrix->width);
    // eC.noalias() += eA * eB.transpose();
    freeBlob(padBlob);
  }
}

void maxPoolingInit(const Blob *blob, Blob *Matrix, int kernelSize,
                    int stride) {
  Matrix->width = ceil((float)(blob->width - kernelSize) / stride + 1);
  Matrix->height = ceil((float)(blob->height - kernelSize) / stride + 1);
  Matrix->channel = blob->channel;
  Matrix->pdata = (Dtype *)malloc(Matrix->channel * Matrix->width *
                                  Matrix->height * sizeof(Dtype));
  if (Matrix->pdata == NULL) cout << "the maxPoolingInit is failed!!" << endl;
  memset(Matrix->pdata, 0,
         Matrix->channel * Matrix->width * Matrix->height * sizeof(Dtype));
}

void maxPooling(const Blob *blob, Blob *Matrix, int kernelSize, int stride) {
  if (blob->pdata == NULL) {
    cout << "the feature2Matrix Blob is NULL!!" << endl;
    return;
  }
  Dtype *p = Matrix->pdata;
  Dtype *pIn;
  Dtype *ptemp;
  Dtype maxNum = 0;
  if ((blob->width - kernelSize) % stride == 0) {
    for (int row = 0; row < Matrix->height; row++) {
      for (int col = 0; col < Matrix->width; col++) {
        pIn = blob->pdata + row * stride * blob->width + col * stride;
        for (int channel = 0; channel < blob->channel; channel++) {
          ptemp = pIn + channel * blob->height * blob->width;
          maxNum = *ptemp;
          for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
            for (int i = 0; i < kernelSize; i++) {
              if (maxNum < *(ptemp + i + kernelRow * blob->width))
                maxNum = *(ptemp + i + kernelRow * blob->width);
            }
          }
          *(p + channel * Matrix->height * Matrix->width) = maxNum;
        }
        p++;
      }
    }
  } else {
    int diffh = 0, diffw = 0;
    for (int channel = 0; channel < blob->channel; channel++) {
      pIn = blob->pdata + channel * blob->height * blob->width;
      for (int row = 0; row < Matrix->height; row++) {
        for (int col = 0; col < Matrix->width; col++) {
          ptemp = pIn + row * stride * blob->width + col * stride;
          maxNum = *ptemp;
          diffh = row * stride - blob->height + 1;
          diffw = col * stride - blob->height + 1;
          for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
            if ((kernelRow + diffh) > 0) break;
            for (int i = 0; i < kernelSize; i++) {
              if ((i + diffw) > 0) break;
              if (maxNum < *(ptemp + i + kernelRow * blob->width))
                maxNum = *(ptemp + i + kernelRow * blob->width);
            }
          }
          *p++ = maxNum;
        }
      }
    }
  }
}

void relu(struct Blob *blob, Dtype *pbias) {
  if (blob->pdata == NULL) {
    cout << "the  Relu feature is NULL!!" << endl;
    return;
  }
  if (pbias == NULL) {
    cout << "the  Relu bias is NULL!!" << endl;
    return;
  }
  Dtype *op = blob->pdata;
  Dtype *pb = pbias;

  long dis = blob->width * blob->height;
  for (int channel = 0; channel < blob->channel; channel++) {
    for (int col = 0; col < dis; col++) {
      *op += *pb;
      if (*op < 0) *op = 0;
      op++;
    }
    pb++;
  }
}
void prelu(struct Blob *blob, Dtype *pbias, Dtype *prelu_gmma) {
  if (blob->pdata == NULL) {
    cout << "the  Relu feature is NULL!!" << endl;
    return;
  }
  if (pbias == NULL) {
    cout << "the  Relu bias is NULL!!" << endl;
    return;
  }
  Dtype *op = blob->pdata;
  Dtype *pb = pbias;
  Dtype *pg = prelu_gmma;

  long dis = blob->width * blob->height;
  for (int channel = 0; channel < blob->channel; channel++) {
    for (int col = 0; col < dis; col++) {
      *op = *op + *pb;
      *op = (*op > 0) ? (*op) : ((*op) * (*pg));
      op++;
    }
    pb++;
    pg++;
  }
}
void fullconnectInit(const Weight *weight, Blob *outBlob) {
  outBlob->channel = weight->selfChannel;
  outBlob->width = 1;
  outBlob->height = 1;
  outBlob->pdata = (Dtype *)malloc(weight->selfChannel * sizeof(Dtype));
  if (outBlob->pdata == NULL) cout << "the fullconnectInit is failed!!" << endl;
  memset(outBlob->pdata, 0, weight->selfChannel * sizeof(Dtype));
}
void fullconnect(const Weight *weight, const Blob *blob, Blob *outBlob) {
  if (blob->pdata == NULL) {
    cout << "the fc feature is NULL!!" << endl;
    return;
  }
  if (weight->pdata == NULL) {
    cout << "the fc weight is NULL!!" << endl;
    return;
  }
  memset(outBlob->pdata, 0, weight->selfChannel * sizeof(Dtype));
  // Y←αAX + βY    β must be 0(zero)
  //               row         no trans         A's row               A'col
  // cblas_sgemv(CblasRowMajor, CblasNoTrans, weight->selfChannel,
  //             weight->lastChannel, 1, weight->pdata, weight->lastChannel,
  //             Blob->pdata, 1, 0, outBlob->pdata, 1);
  gemm(0, 0, weight->selfChannel, 1, weight->lastChannel, 1.0, weight->pdata,
       weight->lastChannel, blob->pdata, 1, 0, outBlob->pdata, 1);
}

void readData(const float *datasource, int sourcelen, long dataNumber[],
              Dtype *pTeam[], int num) {
  int p = 0;
  for (int i = 0; i < num; ++i) {
    int remain = sourcelen - p;
    int cplen = std::min(remain, int(dataNumber[i]));
    memcpy(pTeam[i], datasource + p, cplen * sizeof(float));
    p += cplen;
  }
}

void readData(string filename, long dataNumber[], Dtype *pTeam[]) {
  ifstream in(filename.data());
  string line;
  if (in) {
    int i = 0;
    int count = 0;
    int pos = 0;
    while (getline(in, line)) {
      try {
        if (i < dataNumber[count]) {
          line.erase(0, 1);
          pos = line.find(']');
          line.erase(pos, 1);
          *(pTeam[count])++ = atof(line.data());
        } else {
          count++;
          dataNumber[count] += dataNumber[count - 1];

          line.erase(0, 1);
          pos = line.find(']');
          line.erase(pos, 1);
          *(pTeam[count])++ = atof(line.data());
        }
        i++;
      } catch (exception &e) {
        cout << " yichang " << i << endl;
        return;
      }
    }
  } else {
    cout << "no such file" << filename << endl;
  }
}
long initConvAndFc(struct Weight *weight, int schannel, int lchannel,
                   int kersize, int stride, int pad) {
  weight->selfChannel = schannel;
  weight->lastChannel = lchannel;
  weight->kernelSize = kersize;
  weight->stride = stride;
  weight->pad = pad;
  weight->pbias = (Dtype *)malloc(schannel * sizeof(Dtype));
  if (weight->pbias == NULL) cout << "neicun muyou shenqing chengong!!";
  memset(weight->pbias, 0, schannel * sizeof(Dtype));
  long byteLenght = weight->selfChannel * weight->lastChannel *
                    weight->kernelSize * weight->kernelSize;
  weight->pdata = (Dtype *)malloc(byteLenght * sizeof(Dtype));
  if (weight->pdata == NULL) cout << "neicun muyou shenqing chengong!!";
  memset(weight->pdata, 0, byteLenght * sizeof(Dtype));

  return byteLenght;
}
void initpRelu(struct pRelu *prelu, int width) {
  prelu->width = width;
  prelu->pdata = (Dtype *)malloc(width * sizeof(Dtype));
  if (prelu->pdata == NULL) cout << "prelu apply for memory failed!!!!";
  memset(prelu->pdata, 0, width * sizeof(Dtype));
}
void softmax(const struct Blob *blob) {
  if (blob->pdata == NULL) {
    cout << "the softmax's pdata is NULL , Please check !" << endl;
    return;
  }
  Dtype *p2D = blob->pdata;
  Dtype *p3D = NULL;
  long mapSize = blob->width * blob->height;
  Dtype eleSum = 0;
  for (int row = 0; row < blob->height; row++) {
    for (int col = 0; col < blob->width; col++) {
      eleSum = 0;
      for (int channel = 0; channel < blob->channel; channel++) {
        p3D = p2D + channel * mapSize;
        *p3D = exp(*p3D);
        eleSum += *p3D;
      }
      for (int channel = 0; channel < blob->channel; channel++) {
        p3D = p2D + channel * mapSize;
        *p3D = (*p3D) / eleSum;
      }
      p2D++;
    }
  }
}

bool cmpScore(struct orderScore lsh, struct orderScore rsh) {
  if (lsh.score < rsh.score)
    return true;
  else
    return false;
}
void nms(vector<struct Bbox> &boundingBox_,
         vector<struct orderScore> &bboxScore_, const float overlap_threshold,
         string modelname) {
  if (boundingBox_.empty()) {
    return;
  }
  std::vector<int> heros;
  // sort the score
  sort(bboxScore_.begin(), bboxScore_.end(), cmpScore);

  int order = 0;
  float IOU = 0;
  float maxX = 0;
  float maxY = 0;
  float minX = 0;
  float minY = 0;
  while (bboxScore_.size() > 0) {
    order = bboxScore_.back().oriOrder;
    bboxScore_.pop_back();
    if (order < 0) continue;
    heros.push_back(order);
    boundingBox_.at(order).exist = false;  // delete it

    for (int num = 0; num < boundingBox_.size(); num++) {
      if (boundingBox_.at(num).exist) {
        // the iou
        maxX = (boundingBox_.at(num).x1 > boundingBox_.at(order).x1)
                   ? boundingBox_.at(num).x1
                   : boundingBox_.at(order).x1;
        maxY = (boundingBox_.at(num).y1 > boundingBox_.at(order).y1)
                   ? boundingBox_.at(num).y1
                   : boundingBox_.at(order).y1;
        minX = (boundingBox_.at(num).x2 < boundingBox_.at(order).x2)
                   ? boundingBox_.at(num).x2
                   : boundingBox_.at(order).x2;
        minY = (boundingBox_.at(num).y2 < boundingBox_.at(order).y2)
                   ? boundingBox_.at(num).y2
                   : boundingBox_.at(order).y2;
        // maxX1 and maxY1 reuse
        maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
        maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
        // IOU reuse for the area of two bbox
        IOU = maxX * maxY;
        if (!modelname.compare("Union"))
          IOU = IOU /
                (boundingBox_.at(num).area + boundingBox_.at(order).area - IOU);
        else if (!modelname.compare("Min")) {
          IOU = IOU / ((boundingBox_.at(num).area < boundingBox_.at(order).area)
                           ? boundingBox_.at(num).area
                           : boundingBox_.at(order).area);
        }
        if (IOU > overlap_threshold) {
          boundingBox_.at(num).exist = false;
          for (vector<orderScore>::iterator it = bboxScore_.begin();
               it != bboxScore_.end(); it++) {
            if ((*it).oriOrder == num) {
              (*it).oriOrder = -1;
              break;
            }
          }
        }
      }
    }
  }
  for (int i = 0; i < heros.size(); i++)
    boundingBox_.at(heros.at(i)).exist = true;
}
void refineAndSquareBbox(vector<struct Bbox> &vecBbox, const int &height,
                         const int &width) {
  if (vecBbox.empty()) {
    cout << "Bbox is empty!!" << endl;
    return;
  }
  float bbw = 0, bbh = 0, maxSide = 0;
  float h = 0, w = 0;
  float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
  for (vector<struct Bbox>::iterator it = vecBbox.begin(); it != vecBbox.end();
       it++) {
    if ((*it).exist) {
      bbh = (*it).x2 - (*it).x1 + 1;
      bbw = (*it).y2 - (*it).y1 + 1;
      x1 = (*it).x1 + (*it).regreCoord[1] * bbh;
      y1 = (*it).y1 + (*it).regreCoord[0] * bbw;
      x2 = (*it).x2 + (*it).regreCoord[3] * bbh;
      y2 = (*it).y2 + (*it).regreCoord[2] * bbw;

      h = x2 - x1 + 1;
      w = y2 - y1 + 1;

      maxSide = (h > w) ? h : w;
      x1 = x1 + h * 0.5 - maxSide * 0.5;
      y1 = y1 + w * 0.5 - maxSide * 0.5;
      (*it).x2 = round(x1 + maxSide - 1);
      (*it).y2 = round(y1 + maxSide - 1);
      (*it).x1 = round(x1);
      (*it).y1 = round(y1);

      // boundary check
      if ((*it).x1 < 0) (*it).x1 = 0;
      if ((*it).y1 < 0) (*it).y1 = 0;
      if ((*it).x2 > height) (*it).x2 = height - 1;
      if ((*it).y2 > width) (*it).y2 = width - 1;

      it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
    }
  }
}