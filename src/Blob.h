#ifndef Blob_H
#define Blob_H
#include <stdlib.h>
#include <iostream>

using namespace std;
#define Dtype float

struct Blob {
  Dtype *pdata;
  int width;
  int height;
  int channel;
};

typedef struct Blob Blob;

struct pRelu {
  Dtype *pdata;
  int width;
};

struct Weight {
  Dtype *pdata;
  Dtype *pbias;
  int lastChannel;
  int selfChannel;
  int kernelSize;
  int stride;
  int pad;
};

struct Bbox {
  float score;
  int x1;
  int y1;
  int x2;
  int y2;
  float area;
  bool exist;
  Dtype ppoint[10];
  Dtype regreCoord[4];
};

struct orderScore {
  Dtype score;
  int oriOrder;
};

void freeBlob(struct Blob *Blob);
void freeWeight(struct Weight *weight);
void freepRelu(struct pRelu *prelu);
void BlobShow(const struct Blob *Blob);
void BlobShowE(const struct Blob *Blob, int channel, int row);
void weightShow(const struct Weight *weight);
void pReluShow(const struct pRelu *prelu);
#endif