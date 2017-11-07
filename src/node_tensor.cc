#include<node.h>
#include "tensor.h"
#include <iostream>

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
using v8::Exception;

using tensor::Tensor;

Tensor cTensorFromJSTensor(Isolate* isolate, const Local<Value> jsTensor) {
  Tensor cTensor;

  Local<Context> context = isolate->GetCurrentContext();

  Local<Object> obj = jsTensor->ToObject();

  Local<Value> tempValue;
  MaybeLocal<Value> tempMaybe;
  tempMaybe = obj->Get(context, String::NewFromUtf8(isolate,"numDimensions"));
  if(tempMaybe.IsEmpty()) {
    cTensor.data = NULL;
    return cTensor;
  }
  tempValue = tempMaybe.ToLocalChecked();
  if(!tempValue->IsUint32()) {
    cTensor.data = NULL;
    return cTensor;
  }
  cTensor.numDimensions = tempValue->Uint32Value();


  tempMaybe = obj->Get(context, String::NewFromUtf8(isolate,"initial_offset"));
  if(tempMaybe.IsEmpty()) {
    cTensor.data = NULL;
    return cTensor;
  }
  tempValue = tempMaybe.ToLocalChecked();
  if(!tempValue->IsUint32()) {
    cTensor.data = NULL;
    return cTensor;
  }
  cTensor.initial_offset = tempValue->Uint32Value();


  tempMaybe = obj->Get(context, String::NewFromUtf8(isolate,"dimensions"));
  if(tempMaybe.IsEmpty()) {
    cTensor.data = NULL;
    return cTensor;
  }
  tempValue = tempMaybe.ToLocalChecked();
  if(!tempValue->IsUint32Array()) {
    cTensor.data = NULL;
    return cTensor;
  }
  if(tempValue.As<v8::Uint32Array>()->Length() != cTensor.numDimensions) {
    cTensor.data = NULL;
    return cTensor;
  }
  cTensor.dimensions = reinterpret_cast<uint32_t*>(GET_CONTENTS(tempValue.As<v8::Uint32Array>()));


  tempMaybe = obj->Get(context, String::NewFromUtf8(isolate,"strides"));
  if(tempMaybe.IsEmpty()) {
    cTensor.data = NULL;
    return cTensor;
  }
  tempValue = tempMaybe.ToLocalChecked();
  if(!tempValue->IsUint32Array()) {
    cTensor.data = NULL;
    return cTensor;
  }
  if(tempValue.As<v8::Uint32Array>()->Length() != cTensor.numDimensions) {
    cTensor.data = NULL;
    return cTensor;
  }
  cTensor.strides = reinterpret_cast<uint32_t*>(GET_CONTENTS(tempValue.As<v8::Uint32Array>()));


  tempMaybe = obj->Get(context, String::NewFromUtf8(isolate,"data"));
  if(tempMaybe.IsEmpty()) {
    cTensor.data = NULL;
    return cTensor;
  }
  tempValue = tempMaybe.ToLocalChecked();
  if(!tempValue->IsFloat64Array()) {
    cTensor.data = NULL;
    return cTensor;
  }
  if(tempValue.As<v8::Float64Array>()->Length() < cTensor.maximumOffset()) {
    cTensor.data = NULL;
    return cTensor;
  }
  cTensor.data = reinterpret_cast<double*>(GET_CONTENTS(tempValue.As<v8::Float64Array>()));

  return cTensor;
}


void contract(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  // Check the number of arguments passed.
  if (args.Length() < 4) {
    // Throw an Error that is passed back to JavaScript
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "Requires 4 arguments: source1, source2, dest, dimsToContract")));
    return;
  }


}
void Method(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  Tensor T = cTensorFromJSTensor(isolate, args[0]);
  if(T.isValid()) {
    T.data[3] = 99;
    args.GetReturnValue().Set(Number::New(isolate, T.strides[1]));
  }
}

void init(Local<Object> exports) {
  NODE_SET_METHOD(exports, "hello", Method);
}

NODE_MODULE(NODE_GYP_MODULE_NAME, init)

}