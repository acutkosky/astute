#include<node.h>
#include "tensor.h"
#include <iostream>
#include <string>

#define GET_CONTENTS(view) \
(static_cast<unsigned char*>(view->Buffer()->GetContents().Data()) + view->ByteOffset())

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

  for(uint32_t i=0; i<T.dimensions[0]; i++) {
    for(uint32_t j=0; j<T.dimensions[1]; j++) {
      cout<<T.at((uint32_t[]){i, j})<<" ";
    }
    cout<<endl;
  }
}

void print1DTensor(Tensor& T) {
  for(uint32_t i=0; i<T.dimensions[0]; i++) {
    cout<<T.at((uint32_t[]){i})<<" ";
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

  Tensor source1 = cTensorFromJSTensor(isolate, args[0]);
  Tensor source2 = cTensorFromJSTensor(isolate, args[1]);
  Tensor dest = cTensorFromJSTensor(isolate, args[2]);

  if(!args[3]->IsUint32()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "dimsToContract must be a positive integer")));
    return;
  }
  uint32_t dimsToContract = args[3]->Uint32Value();


  if(!source1.isValid() || !source2.isValid() || !dest.isValid()) {
     isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "invalid Tensor data")));
    return;   
  }

  TensorError error = tensor::NoError;
  tensor::contract(source1, source2, dest, dimsToContract, &error);

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
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "invalid Tensor data")));
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
    isolate->ThrowException(Exception::TypeError(String::NewFromUtf8(isolate, "invalid Tensor data")));
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
        String::NewFromUtf8(isolate, "Requires 4 arguments: source1, source2, dest, scale1, scale2")));
    return;
  }

  Tensor source1 = cTensorFromJSTensor(isolate, args[0]);
  Tensor source2 = cTensorFromJSTensor(isolate, args[1]);
  Tensor dest = cTensorFromJSTensor(isolate, args[2]);

  if(!args[3]->IsNumber()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "scale1 must be a number")));
    return;
  }
  double scale1 = args[3]->NumberValue();

  if(!args[4]->IsNumber()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "scale2 must be a number")));
    return;
  }
  double scale2 = args[4]->NumberValue();

  if(!source1.isValid() || !source2.isValid() || !dest.isValid()) {
     isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "invalid Tensor data")));
    return;   
  }

  TensorError error = tensor::NoError;
  tensor::addScale(source1, source2, dest, scale1, scale2, &error);

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
        String::NewFromUtf8(isolate, "Requires 4 arguments: source1, source2, dest, scale")));
    return;
  }

  Tensor source1 = cTensorFromJSTensor(isolate, args[0]);
  Tensor source2 = cTensorFromJSTensor(isolate, args[1]);
  Tensor dest = cTensorFromJSTensor(isolate, args[2]);

  if(!args[3]->IsNumber()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "scale must be a number")));
    return;
  }
  double scale = args[3]->NumberValue();

  if(!source1.isValid() || !source2.isValid() || !dest.isValid()) {
     isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "invalid Tensor data")));
    return;   
  }

  TensorError error = tensor::NoError;
  tensor::multiplyScale(source1, source2, dest, scale, &error);

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
        String::NewFromUtf8(isolate, "Requires 4 arguments: source1, source2, dest, scale")));
    return;
  }

  Tensor source1 = cTensorFromJSTensor(isolate, args[0]);
  Tensor source2 = cTensorFromJSTensor(isolate, args[1]);
  Tensor dest = cTensorFromJSTensor(isolate, args[2]);

  if(!args[3]->IsNumber()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "scale must be a number")));
    return;
  }
  double scale = args[3]->NumberValue();

  if(!source1.isValid() || !source2.isValid() || !dest.isValid()) {
     isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "invalid Tensor data")));
    return;   
  }

  TensorError error = tensor::NoError;
  tensor::divideScale(source1, source2, dest, scale, &error);

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
        String::NewFromUtf8(isolate, "Requires 4 arguments: source, dest, scale")));
    return;
  }

  Tensor source = cTensorFromJSTensor(isolate, args[0]);
  Tensor dest = cTensorFromJSTensor(isolate, args[1]);

  if(!args[2]->IsNumber()) {
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "scale must be a number")));
    return;
  }
  double scale = args[2]->NumberValue();

  if(!source.isValid() || !dest.isValid()) {
     isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, "invalid Tensor data")));
    return;   
  }

  TensorError error = tensor::NoError;
  tensor::scale(source, dest, scale, &error);

  if(error != tensor::NoError) {
    std::string errorString = std::string("Error in scale: ") + makeErrorString(error);
    isolate->ThrowException(Exception::TypeError(
        String::NewFromUtf8(isolate, errorString.c_str()) ));
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
  NODE_SET_METHOD(exports, "contract", contract);
  NODE_SET_METHOD(exports, "scalarProduct", scalarProduct);
  NODE_SET_METHOD(exports, "subTensor", subTensor);
  NODE_SET_METHOD(exports, "addScale", subTensor);
  NODE_SET_METHOD(exports, "multiplyScale", subTensor);
  NODE_SET_METHOD(exports, "divideScale", subTensor);
  NODE_SET_METHOD(exports, "scale", subTensor);
}

NODE_MODULE(NODE_GYP_MODULE_NAME, init)

}