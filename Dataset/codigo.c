#include <stdlib.h>
#include <tensor.h>
#include <stdio.h>
#include <math.h>

#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    int B, C, H, W;
    float *data;
    float *grad;
} Tensor;

Tensor create_tensor(int B, int C, int H, int W);
void zero_grad(Tensor *t);
float get(Tensor *t, int b, int c, int h, int w);
void set(Tensort *t, int b, int c, int h, int w, float val);

 #endif


 //Creamos el tensor
 Tensor create_tensor(int B, int C, int H, int W) {
   Tensor t;
   t.B = B;
   t.C = C;
   t.H = H;
   t.W = W;

   int size = B*C*H*W;

   t.data = malloc(sizeof(float)*size);
   t.grad = malloc(sizeof(float)*size);

   for(int i=0; i<size; i++){
        t.data[i]=0;
        t.grad[i]=0;
   }
   return t;
 }

 int index(Tensor *t, int b, int c, int h, int w){
    return b*(t->C*t->H*t->W)
        + c*(t->H*t->W)
        + h*(t->W)
        + w;
 }

float get(Tensor *t, int b, int c, int h, int w){
  return t-> data[index(t,b,c,h,w)];
}

void set (Tensor *t, int b, int c, int h, int w, float val){
  t->data[index(t,b,c,h,w)]= val;
}

typedef struct {
  int in_channels;
    int out_channels;
    int kernel;
    int stride;
    int padding;

    Tensor weights;
    Tensor bias;

    Tensor input_cache;

}Conv2D;

Conv2D create_conv2d(int in_c, int out_c, int kernel) {

    Conv2D conv;

    conv.in_channels = in_c;
    conv.out_channels = out_c;
    conv.kernel = kernel;
    conv.stride = 1;
    conv.padding = 0;

    conv.weights = create_tensor(out_c, in_c, kernel, kernel);
    conv.bias = create_tensor(1, out_c, 1, 1);

    // Inicialización simple
    int size = out_c*in_c*kernel*kernel;
    for(int i=0;i<size;i++)
        conv.weights.data[i] = ((float)rand()/RAND_MAX - 0.5f)*0.1;

    return conv;
}

void conv2d_forward(Conv2D *conv, Tensor *input, Tensor *output) {

    for(int b=0;b<input->B;b++)
    for(int oc=0;oc<conv->out_channels;oc++)
    for(int h=0;h<output->H;h++)
    for(int w=0;w<output->W;w++) {

        float sum = conv->bias.data[oc];

        for(int ic=0;ic<conv->in_channels;ic++)
        for(int kh=0;kh<conv->kernel;kh++)
        for(int kw=0;kw<conv->kernel;kw++) {

            int ih = h + kh;
            int iw = w + kw;

            float val = get(input,b,ic,ih,iw);
            float weight = get(&conv->weights,oc,ic,kh,kw);

            sum += val * weight;
        }

        set(output,b,oc,h,w,sum);
    }
}


