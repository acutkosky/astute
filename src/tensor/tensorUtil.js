/* jshint esversion: 6 */


function compatibleForContraction(source1, source2, dimsToContract, dest) {

  if(source1.numDimensions < dimsToContract)
    return false;
  if(source2.numDimensions < dimsToContract)
    return false;


  for(let i=0; i<dimsToContract; i++) {
    let source1Offset = source1.numDimensions - 1 - i;
    let source2Offset = i;
    if(source1.shape[source1Offset] != source2.shape[source2Offset]) 
      return false;
  }

  for(let i=0; i<source1.numDimensions - dimsToContract; i++) {
    if(dest.shape[i] != source1.shape[i])
      return false;
  }
  for(let i=0; i<source2.numDimensions - dimsToContract; i++) {
    if(dest.shape[i + source1.numDimensions - dimsToContract] !=
        source2.shape[i + dimsToContract])
      return false;
  }

  return true;
}
exports.compatibleForContraction = compatibleForContraction;

function flattenArray(data) {
  if(!(data instanceof Array))
    return data;
  var flattened = [];
  for(let i=0; i<data.length; i++) {
    flattened = flattened.concat(flattenArray(data[i]));
  }
  return flattened;
}
exports.flattenArray = flattenArray;

class MultiIndexIterator {

  constructor(shape) {
    this.shape = shape;
    this.numDimensions = shape.length;
    this.currentCoords = [];
    this.ended = false;
    for(let i=0; i<numDimensions; i++) {
      currentCoords.push(0);
    }
  }

  next() {
    var i = 0;
    this.currentCoords[i] = (this.currentCoords[i] + 1) % this.shape[i];
    while(this.currentCoords[i] === 0) {
      i++;
      if(i>=this.numDimensions) {
        this.ended = true;
        break;
      }
      this.currentCoords[i] = (this.currentCoords[i] + 1) % this.shape[i];
     }
    return !ended;
  }

  get() {
    return this.currendCoords();
  }
}
exports.MultiIndexIterator = MultiIndexIterator;
