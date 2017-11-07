#include<node.h>
#include "tensor.h"

#define GET_CONTENTS(view) \
(static_cast<unsigned char*>(view->Buffer()->GetContents().Data()) + view->ByteOffset())

namespace nodetensor {

using v8::FunctionCallbackInfo;
using v8::Isolate;
using v8::Local;
using v8::MaybeLocal;
using v8::Object;
using v8::String;
using v8::Value;
using v8::Number;
using v8::Context;

using tensor::Tensor;

Tensor cTensorFromJSTensor(Isolate* isolate, const Local<Value> jsTensor) {
  Tensor cTensor;

  Local<Context> context = isolate->GetCurrentContext();

  Local<Object> obj = jsTensor->ToObject();

  Local<Value> tempValue;
  MaybeLocal<Value> tempMaybe;

  tempMaybe = obj->Get(context, String::NewFromUtf8(isolate,"data"));
  if(tempMaybe.IsEmpty()) {
    cTensor.data = NULL;
    return cTensor;
  }
  tempValue = tempMaybe.ToLocalChecked();
  cTensor.data = reinterpret_cast<double*>(GET_CONTENTS(tempValue.As<v8::Float64Array>()));

  tempMaybe = obj->Get(context, String::NewFromUtf8(isolate,"dimensions"));
  if(tempMaybe.IsEmpty()) {
    cTensor.data = NULL;
    return cTensor;
  }
  tempValue = tempMaybe.ToLocalChecked();
  cTensor.dimensions = reinterpret_cast<uint32_t*>(GET_CONTENTS(tempValue.As<v8::Uint32Array>()));

  tempMaybe = obj->Get(context, String::NewFromUtf8(isolate,"strides"));
  if(tempMaybe.IsEmpty()) {
    cTensor.data = NULL;
    return cTensor;
  }
  tempValue = tempMaybe.ToLocalChecked();
  cTensor.strides = reinterpret_cast<uint32_t*>(GET_CONTENTS(tempValue.As<v8::Uint32Array>()));

  tempMaybe = obj->Get(context, String::NewFromUtf8(isolate,"numDimensions"));
  if(tempMaybe.IsEmpty()) {
    cTensor.data = NULL;
    return cTensor;
  }
  tempValue = tempMaybe.ToLocalChecked();
  cTensor.numDimensions= tempValue->Uint32Value();

  return cTensor;
}

void Method(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  Tensor T = cTensorFromJSTensor(isolate, args[0]);
  T.data[3] = 99;
  args.GetReturnValue().Set(Number::New(isolate, T.strides[1]));
}

void init(Local<Object> exports) {
  NODE_SET_METHOD(exports, "hello", Method);
}

NODE_MODULE(NODE_GYP_MODULE_NAME, init)

}