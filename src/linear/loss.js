/* jshint esversion: 6 */

function logisticLoss(pred, label) {
  return pred.scale(-label).exp().add(1.0).log();
}
exports.logisticLoss = logisticLoss;

function squaredLoss(pred, label) {
  return pred.sub(label).square().scale(0.5);
}
exports.squaredLoss = squaredLoss;