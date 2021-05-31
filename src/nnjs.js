// JavaScript Simple Neural Network toolkit
// [CORE]
// Initally Written by Serge Ageyev
// Open Source Software under MIT License

// Require prng.js

// Exports: NN namespace

var NN = new function() { var NCore = this;

// Public constants / params

NCore.DIV_IN_TRAIN = false; 

// Tools

//var PRNG = new Random(1363990714); // would not train on 2+1 -> 2+1 -> 1 configuration
var PRNG = new Random(new Date().getTime());

function getRandom (min, max)
{
  return (PRNG.randFloat(min, max));
}

// getRandomInt (from, limit) Return integer in range [from, limit) :: from-inclusive, limit-exclusive
// getRandomInt (limit)       Return integer in range [0, limit)    :: 0-inclusive, limit-exclusive
function getRandomInt (from, limit)
{
  if (limit == null) { limit = from; from = 0; }
  return Math.floor((PRNG.nextFloat() * (limit - from)) + from) % limit;
}

// Activation functions

function S(x)
{
  return(1.0/(1.0+Math.exp(-x)));
}

function SD(x)
{
  // e^x/(1 + e^x)^2
  return(Math.exp(x)/Math.pow(1.0+Math.exp(x),2));
}

// Neuron types

// Neuron have to have following functions:
// // Proc                                                        Input             Normal             Bias  
// .get()                        - to provide its current result  value of .set     result of .proc    1.0
// .proc()                       - to proccess data on its input  [ignored]         do proc inp links  [ignored]
// .set(inp)                     - to assing input value          assing inp value  [N/A]              [N/A]
// .inputs[] (used for train)    - current input Neurons links    [N/A]             [input link count] [] // empty
// .w[]      (used for train)    - current input Neurons weights  [N/A]             [input link count] [] // empty
// // Construction
// .addInput(Neuron,weight)      - add input link to Neuron       [N/A]             add input link     [ignored]
// .addInputAll(Neurons,weights) - add input link to Neurons      [N/A]             add input links    [ignored]
// // Train
// .getSum()                     - raw sum of all inputs before S [N/A]             sum of .proc       1.0
// .nw[]                         - new input Neurons weights      [N/A]             [input link count] [] // empty
// .initNewWeights()             - init new  weights (.nw) array  [N/A]             copy .w to .nw     [ignored]
// .addNewWeightsDelta(DW)       - adds DW to new  weights (.nw)  [N/A]             add dw to .nw      [ignored]
// .applyNewWeights()            - adds DW to new  weights (.nw)  [N/A]             copy .nw to .w     [ignored]

function InputNeuron()
{
  var that = this;

  that.out = 0.0;

  that.set = function(value)
  {
    that.out = value;
  }

  that.proc = function() { } // Nothing to do, action taken on "set"

  that.get = function()
  {
    return(that.out);
  }
}

function getRandomInitWeight()
{
  return(getRandom(-1, 1));
}

function ProcNeuron()
{
  var that = this;

  that.inputs = [];
  that.w = [];

  that.out = 0.0;

  that.sum = 0.0; // for train
  that.nw  = [];  // for train

  function calcOutputSum(ins)
  {
    var out = 0;

    for (var i = 0; i < that.w.length; i++)
    {
      out += ins[i] * that.w[i];
    }

    return(out);
  }

  that.addInput = function(neuron, w)
  {
    if (w == null) { w = getRandomInitWeight(); }
    that.inputs.push(neuron);
    that.w.push(w);
  }

  that.addInputAll = function(neurons, weights)
  {
    for (var i = 0; i < neurons.length; i++)
    {
      var w = null;

      if ((weights != null) && (weights[i] != null))
      {
        w = weights[i];
      }

      that.addInput(neurons[i], w);
    }
  }

  that.proc = function()
  {
    var ins = [];

    for (var i = 0; i < that.w.length; i++)
    {
      ins.push(that.inputs[i].get());
    }

    that.sum = calcOutputSum(ins);
    that.out = S(that.sum);
  }

  that.get = function()
  {
    return(that.out);
  }

  // for train

  that.getSum = function()
  {
    return(that.sum);
  }

  that.initNewWeights = function()
  {
    that.nw = that.w.slice();
  }

  that.addNewWeightsDelta = function (dw)
  {
    for (var i = 0; i < that.nw.length; i++)
    {
      that.nw[i] += dw[i];
    }
  }

  that.applyNewWeights = function()
  {
    that.w = that.nw.slice();
  }
}

function BiasNeuron()
{
  var that = this;

  var BIAS = 1.0;

  that.inputs = [];
  that.w = [];

  that.out = BIAS;
  that.sum = BIAS;

  that.sum = 0.0; // for train
  that.nw  = [];  // for train

  that.addInput = function(neuron, w)
  {
    // ignore
  }

  that.addInputAll = function(neurons, weights)
  {
    // ignore
  }

  that.proc = function()
  {
    // nothing to do
  }

  that.get = function()
  {
    return(that.out);
  }

  // for train

  that.getSum = function()
  {
    return(that.sum);
  }

  that.initNewWeights = function()
  {
    // ignore
  }

  that.addNewWeightsDelta = function (dw)
  {
    // ignore
  }

  that.applyNewWeights = function()
  {
    // ignore
  }
}

function Layer(N, maker)
{
  var that = this;

  if (N == null) { N = 0; }
  if (N < 0) { N = 0; }

  if (maker == null) { maker = ProcNeuron; }

  that.neurons = [];

  that.addNeuron = function(maker)
  {
    var theNeuron = new maker();
    that.neurons.push(theNeuron);
    return(theNeuron);
  }

  for (var i = 0; i < N; i++)
  {
    that.addNeuron(maker);
  }

  that.addInputAll = function(inputLayer)
  {
    for (var i = 0; i < that.neurons.length; i++)
    {
      that.neurons[i].addInputAll(inputLayer.neurons);
    }

    return(that);
  }
}

// Processing functions

function doProcNet(layers)
{
  // start from 1 to skip input layer
  for (var i = 1; i < layers.length; i++)
  {
    for (var ii = 0; ii < layers[i].neurons.length; ii++)
    {
      layers[i].neurons[ii].proc();
    }
  }
}

function doProc(NET, inputs)
{
  var LIN = NET[0];

  for (var i = 0; i < LIN.neurons.length; i++)
  {
    if (LIN.neurons[i].set == null)
    {
      // Bias Neuron (skip)
    }
    else
    {
      if (i < inputs.length)
      {
        LIN.neurons[i].set(inputs[i]);
      }
      else
      {
        LIN.neurons[i].set(0);
      }
    }
  }

  doProcNet(NET);

  var result = [];

  var LOUT   = NET[NET.length-1];

  for (var i = 0; i < LOUT.neurons.length; i++)
  {
    result.push(LOUT.neurons[i].get());
  }

  return(result);
}

// Training functions

function getDeltaOutputSum(outNeuron, OSME) // OSME = output sum margin of error (AKA Expected - Calculated)
{
  var OS = outNeuron.getSum();
  var DOS = SD(OS) * OSME; // FIX: DOS is local
  return(DOS);
}

function getDeltaWeights(theNeuron, DOS) // theNeuron in question, DOS = delta output sum
{
  var inputs = theNeuron.inputs;

  var DWS = [];

  var dw;
  for (var i = 0; i < inputs.length; i++)
  {
    if (NCore.DIV_IN_TRAIN) { dw = DOS / inputs[i].get(); } else { dw = DOS * inputs[i].get(); }
    DWS.push(dw);
  }

  return(DWS);
}

function getDeltaHiddenSums(theNeuron, DOS)
{
  var inputs = theNeuron.inputs;

  var DHS = [];

  var ds;
  for (var i = 0; i < inputs.length; i++)
  {
    if (NCore.DIV_IN_TRAIN) { ds = DOS / theNeuron.w[i] * SD(inputs[i].getSum()); } else { ds = DOS * theNeuron.w[i] * SD(inputs[i].getSum()); }
    DHS.push(ds);
  }

  return(DHS);
}

// Train function

function doTrainStep(NET, DATA, TARG, SPEED)
{
  // NET=network, DATA=input, TARG=expeted
  // CALC=calculated output (will be calculated)
  // Note: we re-run calculation here both to receive CALC AND update "sum" state of each neuron in NET

  if (SPEED == null) { SPEED = 1.0; }

  var CALC = doProc(NET, DATA); // we need this because sum has to be updated in NET for each neuron for THIS test case

  for (var i = 1; i < NET.length; i++) // skip input layer
  {
    for (var ii = 0; ii < NET[i].neurons.length; ii++)
    {
      NET[i].neurons[ii].initNewWeights(); // prepare
    }
  }

  // Output layer (special handling)

  var LOUT = NET[NET.length-1].neurons;

  var OSME = []; // output sum margin of error (AKA Expected - Calculated) for each output
  var DOS  = []; // delta output sum for each output neuron
  var DOIW = []; // delta output neuron input weights each output neuron

  for (var i = 0; i < LOUT.length; i++)
  {
    OSME.push((TARG[i] - CALC[i]) * SPEED);
    DOS.push(getDeltaOutputSum(LOUT[i], OSME[i]));
    DOIW.push(getDeltaWeights(LOUT[i], DOS[i]));
    LOUT[i].addNewWeightsDelta(DOIW[i]);
  }

  // proc prev layers

  function procPrevLayer(LOUT, DOS, layerIndex)
  {
    // Addjust previous layer(s)
    // LOUT[neurons count] = current level (its new weights already corrected)
    // DOS[neurons count]  = delta output sum for each neuron on in current level
    // layerIndex = current later index, where 0 = input layer

    if (layerIndex <= 1)
    {
      return; // previous layer is an input layer, so skip any action
    }

    for (var i = 0; i < LOUT.length; i++)
    {
      var LP = LOUT[i].inputs;
      var DOHS = getDeltaHiddenSums(LOUT[i], DOS[i]);

      for (var ii = 0; ii < LP.length; ii++)
      {
        var DW = getDeltaWeights(LP[ii], DOHS[ii]);
        LP[ii].addNewWeightsDelta(DW);
      }

      procPrevLayer(LP, DOHS, layerIndex-1);
    }
  }

  procPrevLayer(LOUT, DOS, NET.length-1);

  for (var i = 1; i < NET.length; i++) // skip input layer
  {
    for (var ii = 0; ii < NET[i].neurons.length; ii++)
    {
      NET[i].neurons[ii].applyNewWeights(); // adjust
    }
  }
}

function isTrainDoneDefaultFunc(DATAS, TARGS, CALCS, eps)
{
  if (eps == null) { eps = 0.1; } // > 0.0 and < 0.5

  function isResultItemMatch(t,c)
  {
    if (Math.abs(t-c) < eps) { return(true); }
    return(false);
  }

  isOK = true;

  for (var s = 0; s < TARGS.length; s++)
  {
    for (var ii = 0; ii < TARGS[s].length; ii++)
    {
      if (!isResultItemMatch(TARGS[s][ii], CALCS[s][ii]))
      {
        isOK = false;
      }
    }
  }

  return(isOK);
}

// Training progress reporter

function TrainProgress()
{
  var that = this;

  that.onTrainingBegin = function(args) { };
  that.onTrainingStep  = function(args, i, maxCount) { return true; }; // return false to abort training
  that.onTrainingEnd   = function(args, isOk) { };
};

TrainProgress.TrainingArgs = function(NET, DATAS, TARGS, SPEED, maxStepsCount) // args param to events
{
  this.NET   = NET;
  this.DATAS = DATAS;
  this.TARGS = TARGS;
  this.SPEED = SPEED;

  this.maxStepsCount = maxStepsCount;
};

TrainProgress.TrainingStep = function(CALCS, stepIndex) // args param to step
{
  this.CALCS = CALCS;
  this.stepIndex = stepIndex;
};

function ConsoleTrainProgress(reportInterval)
{
  var that = this;

  var DEFAULT_REPORT_INTERVAL = 100;

  if (reportInterval == null)
  {
    reportInterval = DEFAULT_REPORT_INTERVAL;
  }
  else if (reportInterval < 0)
  {
    reportInterval = 0;
  }

  TrainProgress.call(this);

  that.onTrainingBegin = function(args) { console.log("TRAINING Started", args.SPEED); };

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
      for (var s = 0; s < DATAS.length; s++)
      {
        console.log("TRAINING Result.N[n,s]", MAX_N, n, s, DATAS[s], TARGS[s], CALCS[s]);
      }
    }

    return true; 
  };

  that.onTrainingEnd   = function(args, isOk)
  {
    var n = lastSeenIndex + 1;
    var NET = args.NET;
    if (isOk)
    {
      console.log("TRAINING OK", "iterations:"+n, NET);
    }
    else
    {
      console.log("TRAINING FAILED", "timeout:"+n, NET);
    }
  };
}

var DEFAULT_TRAIN_COUNT    = 50000;
var DEFAULT_TRAINING_SPEED = 0.125;

/// Train the neural network
function doTrain(NET, DATAS, TARGS, SPEED, MAX_N, progressReporter, isTrainDoneFunc)
{
  if (MAX_N == null)       { MAX_N = DEFAULT_TRAIN_COUNT; }
  if (SPEED == null)       { SPEED = DEFAULT_TRAINING_SPEED; }

  if (isTrainDoneFunc == null) { isTrainDoneFunc = isTrainDoneDefaultFunc; }

  var trainArgs = new TrainProgress.TrainingArgs(NET, DATAS, TARGS, SPEED, MAX_N);

  if (progressReporter != null) { progressReporter.onTrainingBegin(trainArgs); }

  var isDone = false;
  for (var n = 0; (n < MAX_N) && (!isDone); n++)
  {
    var CALCS = [];
    for (var s = 0; s < DATAS.length; s++)
    {
      CALCS.push(doProc(NET, DATAS[s])); // Fill output
    }

    if (progressReporter != null)
    { 
      var trainStep = new TrainProgress.TrainingStep(CALCS, n);
      if (progressReporter.onTrainingStep(trainArgs, trainStep) === false)
      {
        // Abort training
        progressReporter.onTrainingEnd(trainArgs, false);
        return(false); 
      }
    }

    isDone = isTrainDoneFunc(DATAS, TARGS, CALCS);

    if (!isDone)
    {
      for (var s = 0; s < DATAS.length; s++)
      {
        doTrainStep(NET, DATAS[s], TARGS[s], SPEED);
      }
    }
  }

  if (progressReporter != null)
  { 
    progressReporter.onTrainingEnd(trainArgs, isDone);
  }

  return(isDone);
}

// Some internals

NCore.Internal = {};
NCore.Internal.PRNG = PRNG;
NCore.Internal.getRandom = getRandom;
NCore.Internal.getRandomInt = getRandomInt;
NCore.Internal.getDeltaOutputSum  = getDeltaOutputSum;
NCore.Internal.getDeltaWeights    = getDeltaWeights;
NCore.Internal.getDeltaHiddenSums = getDeltaHiddenSums;

// Exports

NCore.ProcNeuron = ProcNeuron;
NCore.InputNeuron = InputNeuron;
NCore.BiasNeuron = BiasNeuron;

NCore.Layer   = Layer;

NCore.doProc  = doProc;

NCore.TrainProgress = TrainProgress;
NCore.ConsoleTrainProgress = ConsoleTrainProgress;
NCore.doTrain = doTrain;

// Aux

NCore.isResultMatchSimpleFunc = function(TARG, CALC, eps) { return(isTrainDoneDefaultFunc(null, [ TARG ], [ CALC ], eps)); }
NCore.isResultBatchMatchSimpleFunc = function(TARGS, CALCS, eps) { return(isTrainDoneDefaultFunc(null, TARGS, CALCS, eps)); }

}()
