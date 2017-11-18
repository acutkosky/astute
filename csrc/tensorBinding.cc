#include<node.h>
#include "tensor.h"
#include "mathops.h"
#include <iostream>
#include <random>
#include <string>

#define GET_CONTENTS(view) \
(static_cast<unsigned char*>(view->Buffer()->GetContents().Data()) + view->ByteOffset())

#define CREATE_OP(name) \
void name(const FunctionCallbackInfo<Value>& args) { \
  Isolate* isolate = args.GetIsolate(); \
  if(args.Length() < 2) { \
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Requires 2 arguments: source, dest"))); \
    return; \
  } \
 \
  Tensor source = cTensorFromJSTensor(isolate, args[0]); \
  Tensor dest = cTensorFromJSTensor(isolate, args[1]); \
 \
  if(!source.isValid() || !dest.isValid()) { \
    return; \
  } \
 \
  TensorError error = tensor::NoError; \
  tensor::name(source, dest, &error); \
  if(error != tensor::NoError) { \
    std::string errorString = std::string("Error in " #name ": ") + makeErrorString(error); \
    isolate->ThrowException(Exception::TypeError( \
        String::NewFromUtf8(isolate, errorString.c_str()) )); \
    return; \
  } \
}
//end CREATE_OP definition

#define CREATE_BINARY_OP(name) void name(const FunctionCallbackInfo<Value>& args) { \
  Isolate* isolate = args.GetIsolate(); \
  if(args.Length() < 3) { \
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "Requires 3 arguments: source1, source2, dest"))); \
    return; \
  } \
 \
  Tensor source1 = cTensorFromJSTensor(isolate, args[0]); \
  Tensor source2 = cTensorFromJSTensor(isolate, args[1]); \
  Tensor dest = cTensorFromJSTensor(isolate, args[2]); \
 \
  if(!source1.isValid() || !source2.isValid() || !dest.isValid()) { \
    return; \
  } \
  TensorError error = tensor::NoError; \
  tensor::name(source1, source2, dest, &error); \
  if(error != tensor::NoError) { \
    std::string errorString = std::string("Error in " #name ": ") + makeErrorString(error); \
    isolate->ThrowException(Exception::TypeError( \
        String::NewFromUtf8(isolate, errorString.c_str()) )); \
    return; \
  } \
}
//end CREATE_BINARY_OP definition

#define DECLARE_OP(name) NODE_SET_METHOD(exports, #name, name);
#define DECLARE_BINARY_OP(name) NODE_SET_METHOD(exports, #name, name);

namespace nodetensor {

using std::cout;
using std::endl;

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
using tensor::TensorError;


void print2DTensor(Tensor& T) {
  uint32_t print_array[2];
  for(uint32_t i=0; i<T.shape[0]; i++) {
    for(uint32_t j=0; j<T.shape[1]; j++) {
      print_array[0] = i;
      print_array[1] = j;
      cout<<T.at(print_array)<<" ";
    }
    cout<<endl;
  }
}

void print1DTensor(Tensor& T) {
  uint32_t print_array[1];
  for(uint32_t i=0; i<T.shape[0]; i++) {
    print_array[0] = i;
    cout<<T.at(print_array)<<" ";
  }
  cout<<endl;
}

void printTensor(Tensor& T) {
  if(T.numDimensions == 1)
    print1DTensor(T);
  if(T.numDimensions == 2)
    print2DTensor(T);
}

void print(uint32_t* p) {
  cout<<p[0]<<" "<<p[1]<<endl;
}

std::string makeErrorString(TensorError error) {
  switch(error) {
    case tensor::MemoryLeakError:
      return std::string("MemoryLeakError");
      break;
    case tensor::DimensionMismatchError:
      return std::string("DimensionMismatchError");
      break;
    case tensor::IndexOutOfBounds:
      return std::string("IndexOutOfBounds");
      break;
    case tensor::SizeMismatchError:
      return std::string("SizeMismatchError");
      break;
    case tensor::NoError:
      return std::string("No Error");
      break;
  }
  return std::string("Unknown Error");
}


/**
  * extracts a Tensor object as defined in tensor.h from a js object with
  * fields of the same name. All C++ arrays in the Tensor object correspond
  * to either Float64Array or Uint32Array objects in the js object.
  * Does some error checking to attempt to save you from buffer-overflows
  * down the line.
  * If the error checking fails, the returned Tensor object has data=NULL
  * and the isValid() method will return false.
  *
  * It is the responsibility of the caller to check isValid() and return an
  * appropriate error to the javascript context.
  **/
Tensor cTensorFromJSTensor(Isolate* isolate, const Local<Value> jsTensor) {
  Tensor cTensor;

  Local<Context> context = isolate->GetCurrentContext();

  Local<Object> obj = jsTensor->ToObject();
  Local<Value> tempValue;
  MaybeLocal<Value> tempMaybe;
  tempMaybe = obj->Get(context, String::NewFromUtf8(isolate,"numDimensions"));
  if(tempMaybe.IsEmpty()) {
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "invalid Tensor data; must define numDimensions")));
    cTensor.data = NULL;
    return cTensor;
  }
  tempValue = tempMaybe.ToLocalChecked();
  if(!tempValue->IsUint32()) {
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "invalid Tensor data; numDimensions must be integer")));
    cTensor.data = NULL;
    return cTensor;
  }
  cTensor.numDimensions = tempValue->Uint32Value();

  tempMaybe = obj->Get(context, String::NewFromUtf8(isolate,"initial_offset"));
  if(tempMaybe.IsEmpty()) {
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "invalid Tensor data; must define initial_offset")));
    cTensor.data = NULL;
    return cTensor;
  }
  tempValue = tempMaybe.ToLocalChecked();
  if(!tempValue->IsUint32()) {
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "invalid Tensor data; initial_offset must be integer")));
    cTensor.data = NULL;
    return cTensor;
  }
  cTensor.initial_offset = tempValue->Uint32Value();


  tempMaybe = obj->Get(context, String::NewFromUtf8(isolate,"shape"));
  if(tempMaybe.IsEmpty()) {
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "invalid Tensor data; must define shape")));
    cTensor.data = NULL;
    return cTensor;
  }
  tempValue = tempMaybe.ToLocalChecked();
  if(!tempValue->IsUint32Array()) {
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "invalid Tensor data; shape must be Uint32Array")));
    cTensor.data = NULL;
    return cTensor;
  }
  if(tempValue.As<v8::Uint32Array>()->Length() != cTensor.numDimensions) {
    cTensor.data = NULL;
    return cTensor;
  }
  cTensor.shape = reinterpret_cast<uint32_t*>(GET_CONTENTS(tempValue.As<v8::Uint32Array>()));


  tempMaybe = obj->Get(context, String::NewFromUtf8(isolate,"strides"));
  if(tempMaybe.IsEmpty()) {
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "invalid Tensor data; must define strides")));
    cTensor.data = NULL;
    return cTensor;
  }
  tempValue = tempMaybe.ToLocalChecked();
  if(!tempValue->IsUint32Array()) {
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "invalid Tensor data; strides must be Uint32Array")));
    cTensor.data = NULL;
    return cTensor;
  }
  if(tempValue.As<v8::Uint32Array>()->Length() != cTensor.numDimensions) {
    cTensor.data = NULL;
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "invalid Tensor data; strides has wrong length")));
    return cTensor;
  }
  cTensor.strides = reinterpret_cast<uint32_t*>(GET_CONTENTS(tempValue.As<v8::Uint32Array>()));


  tempMaybe = obj->Get(context, String::NewFromUtf8(isolate,"data"));
  if(tempMaybe.IsEmpty()) {
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "invalid Tensor data; must define data")));
    cTensor.data = NULL;
    return cTensor;
  }
  tempValue = tempMaybe.ToLocalChecked();
  if(!tempValue->IsFloat64Array()) {
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "invalid Tensor data; data must be Float64Array")));
    cTensor.data = NULL;
    return cTensor;
  }
  if(tempValue.As<v8::Float64Array>()->Length() < cTensor.maximumOffset()) {
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "invalid Tensor data; data buffer too small")));
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
        String::NewFromUtf8(isolate, "Requires 4 arguments: source1, source2, dimsToContract, dest")));
    return;
  }

  Tensor source1 = cTensorFromJSTensor(isolate, args[0]);
  Tensor source2 = cTensorFromJSTensor(isolate, args[1]);
  Tensor dest = cTensorFromJSTensor(isolate, args[3]);

  if(!args[2]->IsUint32()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "dimsToContract must be a positive integer")));
    return;
  }
  uint32_t dimsToContract = args[2]->Uint32Value();


  if(!source1.isValid() || !source2.isValid() || !dest.isValid()) {
    return;   
  }

  TensorError error = tensor::NoError;

  tensor::contract(source1, source2, dimsToContract, dest, &error);

  if(error != tensor::NoError) {
    std::string errorString = std::string("Error in tensor contraction: ") + makeErrorString(error);
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, errorString.c_str()) ));
    return;
  }

}

void scalarProduct(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  // Check the number of arguments passed.
  if (args.Length() < 2) {
    // Throw an Error that is passed back to JavaScript
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "Requires 2 arguments: source1, source2")));
    return;
  }
  Tensor source1 = cTensorFromJSTensor(isolate, args[0]);
  Tensor source2 = cTensorFromJSTensor(isolate, args[1]);
  if(!source1.isValid() || !source2.isValid()) {
    return;
  }

  TensorError error = tensor::NoError;
  double product = tensor::scalarProduct(source1, source2, &error);
  if(error != tensor::NoError) {
    std::string errorString = std::string("Error in scalarProduct: ") + makeErrorString(error);
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, errorString.c_str()) ));
    return;
  }
  args.GetReturnValue().Set(Number::New(isolate, product));
}

void subTensor(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  // Check the number of arguments passed.
  if (args.Length() < 5) {
    // Throw an Error that is passed back to JavaScript
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "Requires 3 arguments: source1, heldCoords, heldValues, numHeld, dest")));
    return;
  }
  Tensor source = cTensorFromJSTensor(isolate, args[0]);
  Tensor dest = cTensorFromJSTensor(isolate, args[4]);
  if(!source.isValid() || !dest.isValid()) {
    return;
  }

  if(!args[3]->IsUint32()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "numHeld must be a positive integer")));
    return;
  }
  uint32_t numHeld = args[3]->Uint32Value();

  if(!args[1]->IsUint32Array()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "heldCoords must be a Uint32array")));
    return;
  }
  if(args[1]->ToObject().As<v8::Uint32Array>()->Length() != numHeld) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "heldCoords must have length equal to numHeld")));
    return; 
  }
  uint32_t* heldCoords = reinterpret_cast<uint32_t*>(GET_CONTENTS(args[1]->ToObject().As<v8::Uint32Array>()));

  if(numHeld != 1) {
    for(uint32_t i=0; i<numHeld-1; i++) {
      if(heldCoords[i] > heldCoords[i+1]) {
        isolate->ThrowException(Exception::TypeError(
            String::NewFromUtf8(isolate, "heldCoords be sorted")));
        return;  
      }
    }
  }

  if(!args[2]->IsUint32Array()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "heldValues must be a Uint32array")));
    return;
  }
  if(args[2]->ToObject().As<v8::Uint32Array>()->Length() != numHeld) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "heldValues must have length equal to numHeld")));
    return; 
  }
  uint32_t* heldValues = reinterpret_cast<uint32_t*>(GET_CONTENTS(args[2]->ToObject().As<v8::Uint32Array>()));


  TensorError error = tensor::NoError;
  tensor::subTensor(source, heldCoords, heldValues, numHeld, dest, &error);
  if(error != tensor::NoError) {
    std::string errorString = std::string("Error in subTensor: ") + makeErrorString(error);
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, errorString.c_str()) ));
    return;
  }
}


void addScale(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  // Check the number of arguments passed.
  if (args.Length() < 5) {
    // Throw an Error that is passed back to JavaScript
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "Requires 4 arguments: source1, source2, scale1, scale2, dest")));
    return;
  }

  Tensor source1 = cTensorFromJSTensor(isolate, args[0]);
  Tensor source2 = cTensorFromJSTensor(isolate, args[1]);
  Tensor dest = cTensorFromJSTensor(isolate, args[4]);

  if(!args[2]->IsNumber()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "scale1 must be a number")));
    return;
  }
  double scale1 = args[2]->NumberValue();

  if(!args[3]->IsNumber()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "scale2 must be a number")));
    return;
  }
  double scale2 = args[3]->NumberValue();

  if(!source1.isValid() || !source2.isValid() || !dest.isValid()) {
    return;   
  }

  TensorError error = tensor::NoError;
  tensor::addScale(source1, source2, scale1, scale2, dest, &error);

  if(error != tensor::NoError) {
    std::string errorString = std::string("Error in addScale: ") + makeErrorString(error);
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, errorString.c_str()) ));
    return;
  }
}

void multiplyScale(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  // Check the number of arguments passed.
  if (args.Length() < 4) {
    // Throw an Error that is passed back to JavaScript
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "Requires 4 arguments: source1, source2, scale, dest")));
    return;
  }

  Tensor source1 = cTensorFromJSTensor(isolate, args[0]);
  Tensor source2 = cTensorFromJSTensor(isolate, args[1]);
  Tensor dest = cTensorFromJSTensor(isolate, args[3]);

  if(!args[2]->IsNumber()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "scale must be a number")));
    return;
  }
  double scale = args[2]->NumberValue();

  if(!source1.isValid() || !source2.isValid() || !dest.isValid()) {
    return;   
  }

  TensorError error = tensor::NoError;
  tensor::multiplyScale(source1, source2, scale, dest, &error);

  if(error != tensor::NoError) {
    std::string errorString = std::string("Error in multiplyScale: ") + makeErrorString(error);
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, errorString.c_str()) ));
    return;
  }
}

void divideScale(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  // Check the number of arguments passed.
  if (args.Length() < 4) {
    // Throw an Error that is passed back to JavaScript
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "Requires 4 arguments: source1, source2, scale, dest")));
    return;
  }

  Tensor source1 = cTensorFromJSTensor(isolate, args[0]);
  Tensor source2 = cTensorFromJSTensor(isolate, args[1]);
  Tensor dest = cTensorFromJSTensor(isolate, args[3]);

  if(!args[2]->IsNumber()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "scale must be a number")));
    return;
  }
  double scale = args[2]->NumberValue();

  if(!source1.isValid() || !source2.isValid() || !dest.isValid()) {
    return;   
  }

  TensorError error = tensor::NoError;
  tensor::divideScale(source1, source2, scale, dest, &error);

  if(error != tensor::NoError) {
    std::string errorString = std::string("Error in divideScale: ") + makeErrorString(error);
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, errorString.c_str()) ));
    return;
  }
}


void scale(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  // Check the number of arguments passed.
  if (args.Length() < 3) {
    // Throw an Error that is passed back to JavaScript
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "Requires 3 arguments: source, scale, dest")));
    return;
  }

  Tensor source = cTensorFromJSTensor(isolate, args[0]);
  Tensor dest = cTensorFromJSTensor(isolate, args[2]);

  if(!args[1]->IsNumber()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "scale must be a number")));
    return;
  }
  double scale = args[1]->NumberValue();

  if(!source.isValid() || !dest.isValid()) {
    return;   
  }

  TensorError error = tensor::NoError;
  tensor::scale(source, scale, dest, &error);

  if(error != tensor::NoError) {
    std::string errorString = std::string("Error in scale: ") + makeErrorString(error);
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, errorString.c_str()) ));
    return;
  }
}

void fillNormal(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  // Check the number of arguments passed.
  if (args.Length() < 3) {
    // Throw an Error that is passed back to JavaScript
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "Requires 3 arguments: mean, stdDev, dest")));
    return;
  }

  Tensor dest = cTensorFromJSTensor(isolate, args[2]);
  if(!dest.isValid())
    return;

  if(!args[0]->IsNumber()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "mean must be a number")));
    return;
  }
  double mean = args[0]->NumberValue();

  if(!args[1]->IsNumber()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "stdDev must be a number")));
    return;
  }
  double stdDev = args[1]->NumberValue();

  tensor::fillNormal(mean, stdDev, dest);

  return;
}

void fillUniform(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  // Check the number of arguments passed.
  if (args.Length() < 3) {
    // Throw an Error that is passed back to JavaScript
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "Requires 3 arguments: low, high, dest")));
    return;
  }

  Tensor dest = cTensorFromJSTensor(isolate, args[2]);
  if(!dest.isValid())
    return;

  if(!args[0]->IsNumber()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "low must be a number")));
    return;
  }
  double low = args[0]->NumberValue();

  if(!args[1]->IsNumber()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "high must be a number")));
    return;
  }
  double high = args[1]->NumberValue();

  tensor::fillUniform(low, high, dest);

  return;
}

void sum(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  // Check the number of arguments passed.
  if (args.Length() < 1) {
    // Throw an Error that is passed back to JavaScript
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "Requires 1 arguments: tensor")));
    return;
  }

  Tensor source = cTensorFromJSTensor(isolate, args[0]);
  if(!source.isValid())
    return;
  double answer = tensor::sum(source);
  args.GetReturnValue().Set(Number::New(isolate, answer));

  return;
}

CREATE_OP(exp)
CREATE_OP(abs)
CREATE_OP(sqrt)
CREATE_OP(sin)
CREATE_OP(cos)
CREATE_OP(tan)
CREATE_OP(sinh)
CREATE_OP(cosh)
CREATE_OP(tanh)
CREATE_OP(log)
CREATE_OP(atan)
CREATE_OP(acos)
CREATE_OP(asin)
CREATE_OP(atanh)
CREATE_OP(acosh)
CREATE_OP(asinh)
CREATE_OP(erf)
CREATE_OP(floor)
CREATE_OP(ceil)
CREATE_OP(round)
CREATE_OP(sign)

CREATE_BINARY_OP(max)
CREATE_BINARY_OP(min)
CREATE_BINARY_OP(pow)
CREATE_BINARY_OP(fmod)


void Method(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  Tensor T = cTensorFromJSTensor(isolate, args[0]);
  if(T.isValid()) {
    T.data[3] = 99;
    args.GetReturnValue().Set(Number::New(isolate, T.strides[1]));
  }
}

void init(Local<Object> exports) {
  tensor::seed_generator();

  NODE_SET_METHOD(exports, "hello", Method);
  NODE_SET_METHOD(exports, "contract", contract);
  NODE_SET_METHOD(exports, "scalarProduct", scalarProduct);
  NODE_SET_METHOD(exports, "subTensor", subTensor);
  NODE_SET_METHOD(exports, "addScale", addScale);
  NODE_SET_METHOD(exports, "multiplyScale", multiplyScale);
  NODE_SET_METHOD(exports, "divideScale", divideScale);
  NODE_SET_METHOD(exports, "scale", scale);
  NODE_SET_METHOD(exports, "fillNormal", fillNormal);
  NODE_SET_METHOD(exports, "fillUniform", fillUniform);
  NODE_SET_METHOD(exports, "sum", sum);

  DECLARE_OP(exp)
  DECLARE_OP(abs)
  DECLARE_OP(sqrt)
  DECLARE_OP(sin)
  DECLARE_OP(cos)
  DECLARE_OP(tan)
  DECLARE_OP(sinh)
  DECLARE_OP(cosh)
  DECLARE_OP(tanh)
  DECLARE_OP(log)
  DECLARE_OP(atan)
  DECLARE_OP(acos)
  DECLARE_OP(asin)
  DECLARE_OP(atanh)
  DECLARE_OP(acosh)
  DECLARE_OP(asinh)
  DECLARE_OP(erf)
  DECLARE_OP(floor)
  DECLARE_OP(ceil)
  DECLARE_OP(round)
  DECLARE_OP(sign)

  DECLARE_BINARY_OP(max)
  DECLARE_BINARY_OP(min)
  DECLARE_BINARY_OP(pow)
  DECLARE_BINARY_OP(fmod)
}

NODE_MODULE(NODE_GYP_MODULE_NAME, init)

}

#undef DECLARE_OP
#undef DECLARE_BINARY_OP
#undef CREATE_OP
