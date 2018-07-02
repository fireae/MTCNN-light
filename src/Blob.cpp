#include "Blob.h"

void freeBlob(struct Blob *Blob) {
  if (Blob->pdata == NULL)
    cout << "Blob is NULL!" << endl;
  else
    free(Blob->pdata);
  Blob->pdata = NULL;
  delete Blob;
}

void freepRelu(struct pRelu *prelu) {
  if (prelu->pdata == NULL)
    cout << "prelu is NULL!" << endl;
  else
    free(prelu->pdata);
  prelu->pdata = NULL;
  delete prelu;
}

void freeWeight(struct Weight *weight) {
  if (weight->pdata == NULL)
    cout << "weight is NULL!" << endl;
  else
    free(weight->pdata);
  weight->pdata = NULL;
  delete weight;
}

void BlobShow(const struct Blob *Blob) {
  if (Blob->pdata == NULL) {
    cout << "Blob is NULL, please check it !" << endl;
    return;
  }
  cout << "the data is :" << endl;
  Dtype *p = Blob->pdata;
  // Blob->channel
  for (int channel = 0; channel < Blob->channel; channel++) {
    cout << "the " << channel << "th channel data is :" << endl;
    // Blob->height
    for (int i = 0; i < Blob->height; i++) {
      for (int k = 0; k < Blob->width; k++) {
        cout << *p++ << " ";
      }
      cout << endl;
    }
  }
  p = NULL;
}

void BlobShowE(const struct Blob *Blob, int channel, int row) {
  if (Blob->pdata == NULL) {
    cout << "the Blob is NULL, please check it !" << endl;
    return;
  }
  cout << "the data is :" << endl;
  Dtype *p = Blob->pdata + channel * Blob->width * Blob->height;
  // Blob->channel
  cout << "the " << channel << "th channel data is :" << endl;
  // Blob->height

  for (int i = 0; i < Blob->height; i++) {
    if (i < 0) {
      for (int k = 0; k < Blob->width; k++) {
        cout << *p++ << " ";
      }
      cout << endl;
    } else if (i == row) {
      p += i * Blob->width;
      for (int k = 0; k < Blob->width; k++) {
        if (k % 4 == 0) cout << endl;
        cout << *p++ << " ";
      }
      cout << endl;
    }
  }
  p = NULL;
}

void pReluShow(const struct pRelu *prelu) {
  if (prelu->pdata == NULL) {
    cout << "the prelu is NULL, please check it !" << endl;
    return;
  }
  cout << "the data is :" << endl;
  Dtype *p = prelu->pdata;
  for (int i = 0; i < prelu->width; i++) {
    cout << *p++ << " ";
  }
  cout << endl;
  p = NULL;
}

void weightShow(const struct Weight *weight) {
  if (weight->pdata == NULL) {
    cout << "the weight is NULL, please check it !" << endl;
    return;
  }
  cout << "the weight data is :" << endl;
  Dtype *p = weight->pdata;
  for (int channel = 0; channel < weight->selfChannel; channel++) {
    cout << "the " << channel << "th channel data is :" << endl;
    for (int i = 0; i < weight->lastChannel; i++) {
      for (int k = 0; k < weight->kernelSize * weight->kernelSize; k++) {
        cout << *p++ << " ";
      }
      cout << endl;
    }
  }
  p = NULL;
}