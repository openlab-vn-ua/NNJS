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
  var base = NN.TrainingProgressReporter;

  // Constructor

  base.call(this);

  if (reportInterval == null) { reportInterval = DEFAULT_REPORT_INTERVAL; }
  if (reportSamples  == null) { reportSamples = false; }
  
  if (reportInterval < 0)
  {
    reportInterval = 0;
  }

  var beginTimeMetter = new NN.TimeMetter();

  function STR(x) { return "" + x; }

  // methods/callbacks

  that.onTrainingBegin = function(args)
  {
    console.log("TRAINING Started", args.SPEED);
    beginTimeMetter.start();
  };

  var lastSeenIndex = 0;

  that.onTrainingStep  = function(args, step)
  { 
    lastSeenIndex = step.stepIndex;

    var n     = step.stepIndex + 1;
    var MAX_N = args.maxStepsCount;
    var DATAS = args.DATAS;
    var TARGS = args.TARGS;
    var CALCS = step.CALCS;

    if ((reportInterval > 0) && ((n % reportInterval) == 0))
    {
      var variance = NN.NetworkStat.getResultSetAggError(TARGS, CALCS);
      console.log("TRAINING AggError[n,s]", MAX_N, n, variance);
      if (reportSamples)
      {
        for (var s = 0; s < DATAS.length; s++)
        {
          console.log("TRAINING Result.N[n,s]", MAX_N, n, s, DATAS[s], TARGS[s], CALCS[s]);
        }
      }
    }

    return true; 
  };

  that.onTrainingEnd   = function(args, isOk)
  {
    beginTimeMetter.stop();

    var n = lastSeenIndex + 1;
    var NET = args.NET;

    var spentTime = beginTimeMetter.millisPassed(); // ms
    if (spentTime <= 0) { spentTime = 1; }

    var scale = NN.NetworkStat.getNetWeightsCount(NET) * args.DATAS.length * n;
    var speed = Math.round((1.0 * scale / spentTime));

    if (isOk)
    {
      console.log("TRAINING OK", "iterations:" + STR(n), "time:" + STR(spentTime) + " ms", "speed:" + STR(speed) + "K w*s/s", NET);
    }
    else
    {
      console.log("TRAINING FAILED", "timeout:"+STR(n), NET);
    }
  };

  return that;
}
TrainingProgressReporterConsole.DEFAULT_REPORT_INTERVAL = DEFAULT_REPORT_INTERVAL; // export

if (NN != null) {
NN.TrainingProgressReporterConsole = TrainingProgressReporterConsole;
}

})();