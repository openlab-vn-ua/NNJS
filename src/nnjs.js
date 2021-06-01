// JavaScript Simple Neural Network toolkit
// [CORE]
// Initally Written by Serge Ageyev
// Open Source Software under MIT License

// Require prng.js

// Exports: NN namespace

var NN = new function() { var NN = this;

// Public constants / params

// Use "/" instead of * during train. Used in unit tests only, should be false on production
NN.DIV_IN_TRAIN = false; 

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

function assert(condition) { if (!condition) { throw 'Assert failed'; } }
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
// // Base                                                        Input             Normal             Bias  
// .get()                        - to provide its current result  value of .set     result of .proc    1.0
// .proc()                       - to proccess data on its input  [ignored]         do proc inp links  [ignored]
// // Input
// .set(inp)                     - to assing input value          assing inp value  [N/A]              [N/A]
// // Proc
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

// InputNeuron
// Always return set value as its output

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

// Proc Neuron
// Neuron that proccess its input inside proc method

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

    var count = that.w.length;

    assert(ins.length == count);

    for (var i = 0; i < count; i++)
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

  // Core proccsing
  // Computes output based on input

  that.proc = function()
  {
    assert(that.inputs.length == that.w.length);

    var ins = [];

    var count = that.inputs.length;
    for (var i = 0; i < count; i++)
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

  // Replace current weights with new weights

  that.applyNewWeights = function()
  {
    that.w = that.nw.slice();
  }
}

// BiasNeuron
// Always return 1.0 as its output

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

/// Layer
/// Represent a layer of network
/// This class composes neuron network layer and acts as container for neurons
/// Layer Container "owns" Neuron(s)

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
  // potential optimization:
  // we may start from layer 1 (not 0) to skip input layer [on input/bias neurons proc func is empty]
  // but we will start from 0 because we want support "subnet" case here in future
  // start from 1 to skip input layer
  var layersCount = layers.length;
  for (var i = 1; i < layersCount; i++)
  {
    var neuronsCount = layers[i].neurons.length;
    for (var ii = 0; ii < neuronsCount; ii++)
    {
      layers[i].neurons[ii].proc();
    }
  }
}

/// Network
/// This class composes multiple neuron network layers and acts as container for layers
/// Network Container "owns" Layer(s)

// class Network // In JS NET just an array of layers (so far)

// Assign inputs
// The inputs assigned to first layer of the network
// inputs should be same saze as first (input) layer

function doProcAssignInput(NET, inputs)
{
  if ((NET.length <= 0) && (inputs.length <= 0))
  {
    return; // strange, but assume OK
  }

  assert(NET.length > 0);

  var LIN = NET[0];

  // assert(LIN->neurons.length == inputs.length); // input layer may have bias neuron(s)

  for (var i = 0; i < LIN.neurons.length; i++)
  {
    var input = LIN.neurons[i];
    
    if ((input == null) || (input.set == null))
    {
      // Bias Neuron (skip)
    }
    else
    {
      if (i < inputs.length)
      {
        input.set(inputs[i]);
      }
      else
      {
        input.set(0);
      }
    }
  }
}

// Get network output result
// resturns array of network's output

function doProcGetResult(NET)
{
  var result = [];

  if (NET.length <= 0)
  {
    return result; // strange, but assume empty response
  }

  assert(NET.length > 0);

  var LOUT   = NET[NET.length-1];

  for (var i = 0; i < LOUT.neurons.length; i++)
  {
    result.push(LOUT.neurons[i].get());
  }

  return(result);
}

function doProc(NET, inputs)
{
  doProcAssignInput(NET, inputs);

  doProcNet(NET);

  return doProcGetResult(NET);
}

// Training functions

function getDeltaOutputSum(outNeuron, OSME) // OSME = output sum margin of error (AKA Expected - Calculated)
{
  if ((outNeuron == null) || (outNeuron.getSum == null)) { return NaN; }
  var OS = outNeuron.getSum();
  var DOS = SD(OS) * OSME;
  return(DOS);
}

function getDeltaWeights(theNeuron, DOS) // theNeuron in question, DOS = delta output sum
{
  if ((theNeuron == null) || (theNeuron.inputs == null)) { return []; } // Empty

  var count = theNeuron.inputs.length;
  var DWS = [];

  var dw;
  for (var i = 0; i < count; i++)
  {
    if (NN.DIV_IN_TRAIN) { dw = DOS / theNeuron.inputs[i].get(); } else { dw = DOS * theNeuron.inputs[i].get(); }
    DWS.push(dw);
  }

  return(DWS);
}

function getDeltaHiddenSums(theNeuron, DOS)
{
  if ((theNeuron == null) || (theNeuron.inputs == null)) { return []; } // Empty

  var count = theNeuron.inputs.length;
  var DHS = [];

  var ds;
  for (var i = 0; i < count; i++)
  {
    var input = theNeuron.inputs[i];
    if ((input == null) || (input.getSum == null))
    {
      ds = NaN; // This neuron input is non-trainee neuron, ds is N/A since we do not know its getSum()
    }
    else // looks like SD here is SD for input neuron (?) use input->SD(input->getSum()) later
    {
      if (NN.DIV_IN_TRAIN) { ds = DOS / theNeuron.w[i] * SD(input.getSum()); } else { ds = DOS * theNeuron.w[i] * SD(input.getSum()); }
    }

    DHS.push(ds);
  }

  return(DHS);
}

// Train functions
// -----------------------------------------------
// Do network train

function doTrainStepProcPrevLayer(LOUT, DOS, layerIndex)
{
  // Addjust previous layer(s)
  // LOUT[neurons count] = current level (its new weights already corrected)
  // DOS[neurons count]  = delta output sum for each neuron on in current level
  // layerIndex = current later index, where 0 = input layer

  assert(LOUT.length == DOS.length);
  if (layerIndex <= 1)
  {
    return; // previous layer is an input layer, so skip any action
  }

  for (var i = 0; i < LOUT.length; i++)
  {
    var neuron = LOUT[i];

    if ((neuron == null) || (neuron.inputs == null)) { break; } // Non-trainable neuron

    var LP = neuron.inputs;
    var DOHS = getDeltaHiddenSums(neuron, DOS[i]);

    assert(LP.length == DOHS.length);

    for (var ii = 0; ii < LP.length; ii++)
    {
      var input = LP[ii];
      if ((input != null) && (input.getSum != null))
      {
        var DW = getDeltaWeights(input, DOHS[ii]);
        input.addNewWeightsDelta(DW);
      }
    }

    doTrainStepProcPrevLayer(LP, DOHS, layerIndex-1);
  }
}

function doTrainStep(NET, DATA, TARG, SPEED)
{
  // NET=network, DATA=input, TARG=expeted
  // CALC=calculated output (will be calculated)
  // Note: we re-run calculation here both to receive CALC AND update "sum" state of each neuron in NET

  if ((SPEED == null) || (isNaN(SPEED)) || (SPEED <= 0.0)) { SPEED = 0.1; } // 1.0 is max

  var CALC = doProc(NET, DATA); // we need this because sum has to be updated in NET for each neuron for THIS test case

  for (var i = 1; i < NET.length; i++) // skip input layer
  {
    var iicount = NET[i].neurons.length;
    for (var ii = 0; ii < iicount; ii++)
    {
      var neuron = NET[i].neurons[ii];
      if ((neuron != null) && (neuron.initNewWeights != null))
      {
        neuron.initNewWeights(); // prepare
      }
    }
  }

  // Output layer (special handling)

  var LOUT = NET[NET.length-1].neurons;

  var OSME = []; // output sum margin of error (AKA Expected - Calculated) for each output
  var DOS  = []; // delta output sum for each output neuron
  var DOIW = []; // delta output neuron input weights each output neuron

  // proc output layer

  for (var i = 0; i < LOUT.length; i++)
  {
    var neuron = LOUT[i];
    OSME.push((TARG[i] - CALC[i]) * SPEED);
    DOS.push(getDeltaOutputSum(neuron, OSME[i])); // will handle neuron=NULL case
    DOIW.push(getDeltaWeights(neuron, DOS[i])); // will handle neuron=NULL case
    if ((neuron != null) && (neuron.addNewWeightsDelta != null))
    {
      neuron.addNewWeightsDelta(DOIW[i]);
    }
  }

  // proc prev layers
  // will apply training back recursively
  // recursion controlled by laterIndex

  doTrainStepProcPrevLayer(LOUT, DOS, NET.length-1);

  for (var i = 1; i < NET.length; i++) // skip input layer
  {
    var iicount = NET[i].neurons.length;
    for (var ii = 0; ii < iicount; ii++)
    {
      var neuron = NET[i].neurons[ii];
      if ((neuron != null) && (neuron.applyNewWeights != null))
      {
        neuron.applyNewWeights(); // adjust
      }
    }
  }
}

/// Class checker for training is done

function TrainingDoneChecker()
{
  var that = this || {};
  var self = TrainingDoneChecker;

  /// Function checks if training is done
  /// DATAS is a list of source data sets
  /// TARGS is a list of target data sets
  /// CALCS is a list of result data sets
  function isTrainingDone(DATAS, TARGS, CALCS)
  {
    throw 'TrainingDoneChecker::isTrainingDone is abstract';
  }
  that.isTrainingDone = isTrainingDone;

  return that;
}


var DEFAULT_EPS = 0.1;

function TrainingDoneCheckerEps(eps)
{
  var that = this || {};
  var self = TrainingDoneCheckerEps;
  var base = TrainingDoneChecker;

  // Constructor

  base.call(this);

  if ((eps == null) || isNaN(eps) || (eps <= 0.0)) { eps = DEFAULT_EPS; } // > 0.0 and < 0.5

  // simple single vectors match

  function isResultSampleMatch(TARG, CALC, eps)
  {
    if ((eps == null) || isNaN(eps) || (eps <= 0.0)) { eps = DEFAULT_EPS; } // > 0.0 and < 0.5

    function isResultItemMatch(t,c)
    {
      if (Math.abs(t-c) < eps) { return(true); }
      return(false);
    }

    assert(TARG.length == CALC.length);

    for (var ii = 0; ii < TARG.length; ii++)
    {
      if (!isResultItemMatch(TARG[ii], CALC[ii]))
      {
        return(false);
      }
    }

    return(true);
  }
  self.isResultSampleMatch = isResultSampleMatch;

  function getResultSampleVarianceSum(TARG, CALC)
  {
    assert(TARG.length == CALC.length);

    var result = 0;

    for (var ii = 0; ii < TARG.length; ii++)
    {
      var diff = TARG[ii] - CALC[ii];

      result += diff * diff;
    }

    return(result);
  }

  function getResultSetVarianceSum(TARGS, CALCS)
  {
    assert(TARGS.length == CALCS.length);

    var result = 0;

    for (var s = 0; s < TARGS.length; s++)
    {
      result += getResultSampleVarianceSum(TARGS[s], CALCS[s]);
    }

    return(result);
  }

  function getResultSetVariance(TARGS, CALCS)
  {
    var result = getResultSetVarianceSum(TARGS, CALCS);
    var count = TARGS.length;
    if (count > 0) { count *= TARGS[0].length; }
    if (count <= 0) { return NaN; }
    return(result / count);
  }
  self.getResultSetVariance = getResultSetVariance;

  function isTrainingDoneSimple(TARGS, CALCS, eps)
  {
    assert(TARGS.length == CALCS.length);

    for (var s = 0; s < TARGS.length; s++)
    {
      if (!isResultSampleMatch(TARGS[s], CALCS[s], eps))
      {
        return(false);
      }
    }

    return(true);
  }

  function isTrainingDone(DATAS, TARGS, CALCS)
  {
    assert(DATAS.length == TARGS.length);
    assert(TARGS.length == CALCS.length);
    return(isTrainingDoneSimple(TARGS, CALCS, eps));
  }
  that.isTrainingDone = isTrainingDone;
  
  return that;
}

/// Training progress reporter
/// Will be called during traing to report progress
/// May rise traing abort event

function TrainingProgressReporter()
{
  var that = this || {};
  var self = TrainingProgressReporter;
  
  // TrainingArgs parameter

  function TrainingArgs(NET, DATAS, TARGS, SPEED, maxStepsCount) // args param to events
  {
    this.NET   = NET;
    this.DATAS = DATAS;
    this.TARGS = TARGS;
    this.SPEED = SPEED;

    this.maxStepsCount = maxStepsCount;
  }
  self.TrainingArgs = TrainingArgs;

  // TrainingStep parameter
  // for onTrainingStep

  function TrainingStep(CALCS, stepIndex) // args param to step
  {
    this.CALCS = CALCS;
    this.stepIndex = stepIndex;
  }
  self.TrainingStep = TrainingStep;

  // Report methods

  that.onTrainingBegin = function(args) { };
  that.onTrainingStep  = function(args, i, maxCount) { return true; }; // return false to abort training
  that.onTrainingEnd   = function(args, isOk) { };
  
  return that;
};

// Main training function

var DEFAULT_TRAIN_COUNT    = 50000;
var DEFAULT_TRAINING_SPEED = 0.125;

/// Train the neural network
function doTrain(NET, DATAS, TARGS, SPEED, MAX_N, progressReporter, isTrainingDoneChecker)
{
  if ((MAX_N == null) || (MAX_N < 0)) { MAX_N = DEFAULT_TRAIN_COUNT; }
  if ((SPEED == null) || (SPEED < 0)) { SPEED = DEFAULT_TRAINING_SPEED; }

  if (isTrainingDoneChecker == null) { isTrainingDoneChecker = new TrainingDoneCheckerEps(); }

  var trainArgs = new TrainingProgressReporter.TrainingArgs(NET, DATAS, TARGS, SPEED, MAX_N);

  if (progressReporter != null) { progressReporter.onTrainingBegin(trainArgs); }

  // steps
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
      var trainStep = new TrainingProgressReporter.TrainingStep(CALCS, n);
      if (progressReporter.onTrainingStep(trainArgs, trainStep) === false)
      {
        // Abort training
        progressReporter.onTrainingEnd(trainArgs, false);
        return(false); 
      }
    }

    isDone = isTrainingDoneChecker.isTrainingDone(DATAS, TARGS, CALCS);

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

// Exports

// Some internals

NN.Internal = {};
NN.Internal.PRNG = PRNG;
NN.Internal.getRandom = getRandom;
NN.Internal.getRandomInt = getRandomInt;
NN.Internal.getDeltaOutputSum  = getDeltaOutputSum;
NN.Internal.getDeltaWeights    = getDeltaWeights;
NN.Internal.getDeltaHiddenSums = getDeltaHiddenSums;

// Core

NN.ProcNeuron  = ProcNeuron;
NN.InputNeuron = InputNeuron;
NN.BiasNeuron  = BiasNeuron;

NN.Layer       = Layer;

NN.doProc      = doProc;

NN.TrainingDoneChecker = TrainingDoneChecker;
NN.TrainingDoneCheckerEps = TrainingDoneCheckerEps;
NN.TrainingProgressReporter = TrainingProgressReporter;
NN.doTrain = doTrain;

// Aux

NN.isResultSampleMatchSimpleFunc = function(TARG, CALC, eps)
{
  return TrainingDoneCheckerEps.isResultSampleMatch(TARG, CALC, eps);
}

NN.isResultBatchMatchSimpleFunc = function (TARGS, CALCS, eps)
{
  return TrainingDoneCheckerEps.isTrainingDoneSimple(TARGS, CALCS, eps);
}

}()
