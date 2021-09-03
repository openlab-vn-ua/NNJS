// Simple console write trainging reporter

// Require nnjs.js
// Require nnjs.time.metter.js

(function () {

if (NN == null) { throw 'nnjs.js should be included first '; }

// Simple console write trainging reporter

var DEFAULT_REPORT_INTERVAL = 100;

function TrainingProgressReporterConsole(reportInterval, reportSamples)
{
  var that = this;
  var base = NN.TrainingProgressReporter; base.call(that);

  var maxEpochCount = 0;
  var samplesDone = 0;

  var lastSeenIndex = 0;

  var aggErrorSum = NaN;
  var aggValCount = 0;

  var beginTimeMetter = new NN.TimeMetter();

  // Constructor

  if (reportInterval == null) { reportInterval = DEFAULT_REPORT_INTERVAL; }
  if (reportSamples  == null) { reportSamples = false; }
  
  if (reportInterval < 0)
  {
    reportInterval = 0;
  }

  function STR(x) { return "" + x; }

  // methods/callbacks

  that.trainStart = function(NET, trainingParams, datasetInfo)
  //override
  { 
    maxEpochCount = trainingParams.maxEpochCount;
    samplesDone = 0;
    console.log("TRAINING Started", "speed:"+STR(trainingParams.speed), "fastVerify:"+STR(trainingParams.fastVerify));
    beginTimeMetter.start();
  }

  that.trainEposhStart = function(NET, epochIndex)
  //override
  {
    lastSeenIndex = epochIndex;
    aggErrorSum = NaN;
    aggValCount = 0;
  }

  that.trainSampleReportAndCheckContinue = function(NET, DATA, TARG, CALC, epochIndex, sampleIndex)
  //override
  {
    samplesDone++;

    var n = epochIndex + 1;
    var s = sampleIndex;

    if ((reportInterval > 0) && ((n % reportInterval) == 0))
    {
      if (isNaN(aggErrorSum)) { aggErrorSum = 0.0; }

      aggErrorSum += NN.NetworkStat.getResultSampleAggErrorSum(TARG, CALC);
      aggValCount += TARG.length;

      if (reportSamples)
      {
        console.log("TRAINING Result.N[n,s]", maxEpochCount, n, s, DATA, TARG, CALC);
      }
    }

    return true;
  }

  that.trainEposhEnd = function(NET, epochIndex)
  //override
  {
    var n = epochIndex + 1;
    var MAX_N = maxEpochCount;

    if ((reportInterval > 0) && ((n % reportInterval) == 0))
    {
      var variance = NN.NetworkStat.getResultAggErrorByAggErrorSum(aggErrorSum, aggValCount);
      console.log("TRAINING AggError[n,s]", MAX_N, n, variance);
    }
  }

  that.trainEnd = function(NET, isOk)
  //override
  {
    beginTimeMetter.stop();

    var n = lastSeenIndex + 1;

    var spentTime = beginTimeMetter.millisPassed(); // ms
    if (spentTime <= 0) { spentTime = 1; }

    var steps = samplesDone;
    var scale = NN.NetworkStat.getNetWeightsCount(NET) * steps;
    var speed = Math.round((1.0 * scale / spentTime));

    var stepTime = Math.round(((1.0 * spentTime) / steps) * 1000.0);

    if (isOk)
    {
      console.log("TRAINING OK", "iterations:" + STR(n), "time:" + STR(spentTime) + " ms", "speed:" + STR(speed) + "K w*s/s", "step:" + STR(stepTime) + " us", NET.layers);
    }
    else
    {
      console.log("TRAINING FAILED", "timeout:" + STR(n), NET.layers);
    }
  }

  return that;
}
// static items
(function ()
{
  var self = TrainingProgressReporterConsole;
  self.DEFAULT_REPORT_INTERVAL = DEFAULT_REPORT_INTERVAL; // export
})();

if (NN != null) {
NN.TrainingProgressReporterConsole = TrainingProgressReporterConsole;
}

})();