// JavaScript Simple Neural Network toolkit
// [CORE]
// Initally Written by Serge Ageyev
// Open Source Software under MIT License

// Require prng.js

// Exports: NN namespace

var NN = new function() { var NN = this;

// Tools

//var PRNG_ = new Random(1363990714); // would not train on 2+1 -> 2+1 -> 1 configuration
var PRNG_ = new Random(new Date().getTime());

function getPRNG()
{
  return (PRNG_);
}

function getRandom (min, max)
{
  return (getPRNG().randFloat(min, max));
}

// getRandomInt (from, limit) Return integer in range [from, limit) :: from-inclusive, limit-exclusive
// getRandomInt (limit)       Return integer in range [0, limit)    :: 0-inclusive, limit-exclusive
function getRandomInt (from, limit)
{
  if (limit == null) { limit = from; from = 0; }
  return Math.floor((getPRNG().nextFloat() * (limit - from)) + from) % limit;
}

function assert(condition) { if (!condition) { 
  throw 'Assert failed';
} }

// Activation : Base

/// Activation function provider (Should be stateless)
//double ActFunc.S(double)

/// Activation function provider for training  (Should be stateless)
//double ActFuncTrainee.S(souble)
//double ActFuncTrainee.SD(souble)

// Activation : Sigmoid

var CalcMathSigmoid = (new function () // static provider
{
  var self = this;
  self.S   = function(x) { return (1.0 / (1.0 + Math.exp(-x))); }
  self.SD  = function(x) { return (Math.exp(x) / Math.pow(1.0 + Math.exp(x), 2)); } // e^x/(1 + e^x)^2
}());

function ActFuncSigmoid()
{
  var that = this || {};
  that.S = CalcMathSigmoid.S;
  return that;
}
(function () { var self = ActFuncSigmoid; var s = new self(); self.getInstance = function () { return s; } })();

function ActFuncSigmoidTrainee()
{
  var that = this || {};
  that.S  = CalcMathSigmoid.S;
  that.SD = CalcMathSigmoid.SD;
  return that;
}
(function () { var self = ActFuncSigmoidTrainee; var s = new self(); self.getInstance = function () { return s; } })();

// Activation : RELU

var CalcMathRELU = (new function () // static provider
{
  var self = this;
  self.S   = function(x) { return (x < 0) ? 0.0 : x; }
  self.SD  = function(x) { return (x < 0) ? 0.0 : 1.0; }
}());

function ActFuncRELU()
{
  var that = this || {};
  that.S = CalcMathRELU.S;
  return that;
}
(function () { var self = ActFuncRELU; var s = new self(); self.getInstance = function () { return s; } })();

function ActFuncRELUTrainee()
{
  var that = this || {};
  that.S  = CalcMathRELU.S;
  that.SD = CalcMathRELU.SD;
  return that;
}
(function () { var self = ActFuncRELUTrainee; var s = new self(); self.getInstance = function () { return s; } })();

// Activation : LRELU

var CoreMathLRELULeak = 0.001; // [0.0..1.0)

var CoreMathLRELU = function (leak) // core provider
{
  var that = this || {};
  if (leak == null) { leak = CoreMathLRELULeak; }
  that.S   = function(x) { return (x < 0) ? x * leak : x; }
  that.SD  = function(x) { return (x < 0) ? leak : 1.0; }
  return that;
};

function ActFuncLRELU(leak)
{
  var that = this || {};
  core = new CoreMathLRELU(leak);
  that.S = core.S;
  return that;
}
(function () { var self = ActFuncLRELU; var s = new self(); self.getInstance = function () { return s; } })();
(function () { var self = ActFuncLRELU; self.newInstance = function (leak) { return new self(leak); }; })();

function ActFuncLRELUTrainee(leak)
{
  var that = this || {};
  core = new CoreMathLRELU(leak);
  that.S  = core.S;
  that.SD = core.SD;
  return that;
}
(function () { var self = ActFuncLRELUTrainee; var s = new self(); self.getInstance = function () { return s; } })();
(function () { var self = ActFuncLRELUTrainee; self.newInstance = function (leak) { return new self(leak); }; })();

// Activation : LLRELU

var CoreMathLLRELULeak = 0.001; // [0.0..1.0)

var CoreMathLLRELU = function (nleak, pleak) // core provider
{
  var that = this || {};
  if (nleak == null) { nleak = CoreMathLLRELULeak; }
  if (pleak == null) { pleak = nleak; }
  that.S   = function(x) { return (x < 0) ? x * nleak : (x <= 1.0) ? x : 1.0 + (x-1.0) * pleak; }
  that.SD  = function(x) { return (x < 0) ? nleak : (x <= 1.0) ? 1.0 : pleak; }
  return that;
};

function ActFuncLLRELU(nleak, pleak)
{
  var that = this || {};
  core = new CoreMathLLRELU(nleak, pleak);
  that.S = core.S;
  return that;
}
(function () { var self = ActFuncLLRELU; var s = new self(); self.getInstance = function () { return s; } })();
(function () { var self = ActFuncLLRELU; self.newInstance = function (nleak, pleak) { return new self(nleak, pleak); } })();

function ActFuncLLRELUTrainee(nleak, pleak)
{
  var that = this || {};
  core = new CoreMathLLRELU(nleak, pleak);
  that.S  = core.S;
  that.SD = core.SD;
  return that;
}
(function () { var self = ActFuncLLRELUTrainee; var s = new self(); self.getInstance = function () { return s; } })();
(function () { var self = ActFuncLLRELUTrainee; self.newInstance = function (nleak, pleak) { return new self(nleak, pleak); } })();

// Activation : Tanh

var CalcMathTanh = (new function () // static provider
{
  var self = this;
  self.S   = function(x) { return Math.tanh(x); }
  self.SD  = function(x) { return 1.0-Math.pow(Math.tanh(x),2.0); }
}());

function ActFuncTanh()
{
  var that = this || {};
  that.S = CalcMathTanh.S;
  return that;
}
(function () { var self = ActFuncTanh; var s = new self(); self.getInstance = function () { return s; } })();

function ActFuncTanhTrainee()
{
  var that = this || {};
  that.S  = CalcMathTanh.S;
  that.SD = CalcMathTanh.SD;
  return that;
}
(function () { var self = ActFuncTanhTrainee; var s = new self(); self.getInstance = function () { return s; } })();

// Default Activation Functions

function  getDefActFunc()        { return ActFuncSigmoid.getInstance(); }
function  getDefActFuncTrainee() { return ActFuncSigmoidTrainee.getInstance(); }

// Neuron types

// Neuron have to have following functions:
// // Base                                                        Input             Normal               Bias  
// .get()                        - to provide its current result  value of .set     result of .proc      1.0
// .proc()                       - to proccess data on its input  [ignored]         do proc inp links    [ignored]
// // Input
// .set(inp)                     - to assing input value          assing inp value  [N/A]                [N/A]
// // Proc
// .inputs[] (used for train)    - current input Neurons links    [N/A]             [input link count]   [N/A]
// .w[]      (used for train)    - current input Neurons weights  [N/A]             [input link count]   [N/A]
// // Construction
// .addInput(Neuron,weight)      - add input link to Neuron       [N/A]             add input link       [N/A]
// .addInputAll(Neurons,weights) - add input link to Neurons      [N/A]             add input links      [N/A]
// // Trainee
// .getSum()                     - raw sum of all inputs before S [N/A]             sum of .proc         [N/A]
// .nw[]                         - new input Neurons weights      [N/A]             [input link count]   [N/A]
// .initTrainStep()              - init new  weights (.nw) array  [N/A]             .nw[]=.w[], .dos=0   [N/A]
// .addNewWeightsDelta(DW)       - adds DW to new  weights (.nw)  [N/A]             .nw[] += DW[]        [N/A]
// .applyNewWeights()            - adds DW to new  weights (.nw)  [N/A]             .w[]=.nw[]           [N/A]
// // Trainee // FAST
// .dos                          - accumulated delta out sum      [N/A]             accumulated dos      [N/A]
// .addDeltaOutputSum(ddos)      - increments dos for neuron      [N/A]             .dos+ddos            [N/A]
// .getDeltaOutputSum()          - return dos for neuron          [N/A]             return .dos          [N/A]

// Base neuron
// Base class for all neurons
function BaseNeuron()
{
  var that = this || {};
  that.proc = function () { throw "BaseNeuron.proc() is abstract!"; }
  that.get  = function () { throw "BaseNeuron.get() is abstract!"; }
  return that;
}

// InputNeuron
// Always return set value as its output
function InputNeuron()
{
  var that = new BaseNeuron();

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

  return that;
}

function dynamicCastInputNeuron(n) { if (n.set != null) { return n; } return null; }

// Proc Neuron
// Neuron that proccess its input inside proc method
function ProcNeuron(func)
{
  if (func == null) { func = getDefActFunc(); }

  var that = new BaseNeuron();

  function getRandomInitWeight()
  {
    return (getRandom(-1, 1));
  }

  function getRandomInitWeightForNNeurons(n)
  {
    // Select input range of w. for n neurons
    // Simple range is -1..1,
    // But we may try - 1/sqrt(n)..1/sqrt(n) to reduce range of w in case of many inputs (so act func will not saturate)
    var wrange = 1;

    if (n > 0)
    {
      wrange = 1.0/Math.sqrt(n);
    }

    return (getRandom(-wrange, wrange));
  }

  that.inputs = [];
  that.w = [];

  that.func = func;

  that.out = 0.0;

  that.sum = 0.0; // Used for for training, but kept here to simplify training implementation

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
      else
      {
        w = getRandomInitWeightForNNeurons(neurons.length);
      }

      that.addInput(neurons[i], w);
    }
  }

  // Core proccesing
  // Computes output based on input

  that.S = function (x) { assert(func != null); assert(func.S != null); return func.S(x); }

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
    that.out = that.S(that.sum);
  }

  that.get = function()
  {
    return(that.out);
  }

  return that;
}

function dynamicCastProcNeuron(n) { if (n.S != null) { return n; } return null; }

// Proc neuron "trainee"
// Regular proc neuron that extended with training data and functions
function ProcNeuronTrainee(func)
{
  // ProcNeurons extension used for training

  if (func == null) { func = getDefActFuncTrainee(); }

  var that = new ProcNeuron(func);

  // Main training

  that.SD = function (x) { assert(func != null); assert(func.SD != null); return func.SD(x); }

  that.nw = []; // new weights for training

  that.getSum = function() // Activation function argument AKA output sum AKA os 
  {
    return(that.sum);
  }

  that.addNewWeightsDelta = function (dw)
  {
    for (var i = 0; i < that.nw.length; i++)
    {
      that.nw[i] += dw[i];
    }
  }

  that.applyNewWeights = function () // Replace current weights with new weights
  {
    that.w = that.nw.slice();
  }

  function initTrainStepMain()
  {
    that.nw = that.w.slice(); // copy new from w
  }

  // Fast training

  that.dos = 0.0; // for train (delta output sum)

  that.addDeltaOutputSum = function (ddos)
  {
    that.dos += ddos;
  }

  that.getDeltaOutputSum = function (dos)
  {
    return that.dos;
  }

  function initTrainStepFast()
  {
    that.dos = 0.0;
  }

  // init

  that.initTrainStep = function()
  {
    initTrainStepMain();
    initTrainStepFast();
  }

  return that;
}

function dynamicCastProcNeuronTrainee(n) { if (n.SD != null) { return n; } return null; }

// BiasNeuron
// Always return 1.0 as its output

function BiasNeuron()
{
  var that = new BaseNeuron();

  var BIAS = 1.0;

  that.proc = function()
  {
    // nothing to do
  }

  that.get = function()
  {
    return(BIAS);
  }

  return that;
}

/// Neuron factory
/// Creates neurons when batch creation is used

function TheNeuronFactory(NeuronConstructor) // public NeuronFactory
{
  var that = this || {}; // work both as new or regular call

  that.makeNeuron = function ()
  {
    return new NeuronConstructor();
  };

  return that;
}

function ExtNeuronFactory(NeuronConstructor, constructorArg) // public NeuronFactory
{
  var that = this || {}; // work both as new or regular call

  that.makeNeuron = function ()
  {
    return new NeuronConstructor(constructorArg);
  };

  return that;
}

/// Layer
/// Represent a layer of network
/// This class composes neuron network layer and acts as container for neurons
/// Layer Container "owns" Neuron(s)

function Layer(N, maker)
{
  var that = this;

  // Constructor A

  if (N == null) { N = 0; }
  if (N < 0) { N = 0; }

  if (maker == null) { maker = ProcNeuron; }

  that.neurons = [];

  // Methods

  that.addNeuron = function(neuron)
  {
    that.neurons.push(neuron);
    return(neuron);
  };

  function makeNeuronByMaker(maker)
  {
    if (maker.makeNeuron != null) { return maker.makeNeuron(); }
    return new maker; // legacy mode, maker just a function
  }

  that.addNeurons = function(N, maker)
  {
    if (N <= 0) { return; }
    for (var i = 0; i < N; i++)
    {
      var theNeuron = makeNeuronByMaker(maker);
      that.addNeuron(theNeuron);
    }
  }

  that.addInputAll = function(inputLayer)
  {
    for (var i = 0; i < that.neurons.length; i++)
    {
      var neuron = dynamicCastProcNeuron(that.neurons[i]);
      if (neuron != null)
      {
        neuron.addInputAll(inputLayer.neurons);
      }
    }

    return(that);
  }

  // Constructor B

  that.addNeurons(N, maker);
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

function Network()
{
  var that = this || {};

  that.layers = [];

  that.addLayer = function(layer)
  {
    that.layers.push(layer);
    return (layer);
  };

  return that;
};


// Assign inputs
// The inputs assigned to first layer of the network
// inputs should be same saze as first (input) layer

function doProcAssignInput(NET, inputs)
{
  if ((NET.layers.length <= 0) && (inputs.length <= 0))
  {
    return; // strange, but assume OK
  }

  assert(NET.layers.length > 0);

  var LIN = NET.layers[0];

  // assert(LIN->neurons.length == inputs.length); // input layer may have bias neuron(s)

  for (var i = 0; i < LIN.neurons.length; i++)
  {
    var input = dynamicCastInputNeuron(LIN.neurons[i]);
    
    if (input == null)
    {
      // Bias or ProcNeuron (skip) // Not InputNeuron (skip)
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

  if (NET.layers.length <= 0)
  {
    return result; // strange, but assume empty response
  }

  assert(NET.layers.length > 0);

  var LOUT   = NET.layers[NET.layers.length-1];

  for (var i = 0; i < LOUT.neurons.length; i++)
  {
    result.push(LOUT.neurons[i].get());
  }

  return(result);
}

function doProc(NET, inputs)
{
  doProcAssignInput(NET, inputs);

  doProcNet(NET.layers);

  return doProcGetResult(NET);
}

// Network Stat and Math functionality
// -----------------------------------------------
// Mostly used for train

var NetworkStat = new function () // static class
{
  var self = this;

  // MatchEps

  var DEFAULT_EPS = 0.1;

  self.DEFAULT_EPS = DEFAULT_EPS;

  function isResultItemMatchEps(t, c, eps) // private
  {
    //if ((eps == null) || isNaN(eps) || (eps <= 0.0)) { eps = DEFAULT_EPS; } // > 0.0 and < 0.5
    if (Math.abs(t - c) < eps) { return (true); }
    return (false);
  }

  function isResultSampleMatchEps(TARG, CALC, eps)
  {
    if ((eps == null) || isNaN(eps) || (eps <= 0.0)) { eps = DEFAULT_EPS; } // > 0.0 and < 0.5

    var count = TARG.length;

    assert(count == CALC.length);

    for (var ii = 0; ii < count; ii++)
    {
      if (!isResultItemMatchEps(TARG[ii], CALC[ii], eps))
      {
        return(false);
      }
    }

    return(true);
  }
  self.isResultSampleMatchEps = isResultSampleMatchEps;

  // MatchArgMax

  function getMaximumIndexEps(R, eps)
  {
    // input:  R as vector of floats (usualy 0.0 .. 1.0), eps min comparison difference
    // result: index of maximum value, checking that next maximum is at least eps lower.
    // returns -1 if no such value found (maximums too close)

    if ((eps == null) || isNaN(eps) || (eps <= 0.0)) { eps = DEFAULT_EPS; } // > 0.0 and < 0.5

    var FAIL = -1;

    var RLen = R.length;

    if (RLen <= 0) { return(FAIL); }
    if (RLen <= 1) { return(0);    }

    // RLen >= 2

    var currMaxIndex;
    var prevMaxIndex;
    if (R[0] > R[1])
    {
      currMaxIndex = 0;
      prevMaxIndex = 1;
    }
    else
    {
      currMaxIndex = 1;
      prevMaxIndex = 0;
    }

    for (var i = 2; i < RLen; i++)
    {
      if (R[i] > R[currMaxIndex]) { prevMaxIndex = currMaxIndex; currMaxIndex = i; }
    }

    if (isNaN(eps))
    {
      // reserved for NAN = do not check (not used as for now)
    }
    else
    {
      if (R[currMaxIndex] < eps)
      {
        return(FAIL); // not ever greater than 0, no reason so check another max
      }

      if (Math.abs(R[currMaxIndex] - R[prevMaxIndex]) < eps)
      {
        return(FAIL); // maximums too close
      }
    }

    return (currMaxIndex);
  }
  self.getMaximumIndexEps = getMaximumIndexEps;

  function getMaximumIndex(R)
  {
    // input:  R as vector of floats (usualy 0.0 .. 1.0)
    // result: index of maximum value
    // returns -1 if no such value found (vector is empty)

    var FAIL = -1;

    var RLen = R.length;

    if (RLen <= 0) { return(FAIL); }
    if (RLen <= 1) { return(0);    }

    // RLen >= 2

    var currMaxIndex = 0;

    for (var i = 1; i < RLen; i++)
    {
      if (R[i] > R[currMaxIndex]) { currMaxIndex = i; }
    }

    return (currMaxIndex);
  }
  self.getMaximumIndex = getMaximumIndex;

  function isResultSampleMatchArgmaxEps(TARG, CALC, eps)
  {
    if ((eps == null) || isNaN(eps) || (eps <= 0.0)) { eps = DEFAULT_EPS; } // > 0.0 and < 0.5

    var maxIndex = getMaximumIndexEps(CALC, eps);

    if (maxIndex < 0) { return(false); }

    var count = TARG.length;

    assert(count == CALC.length);

    for (var ii = 0; ii < count; ii++)
    {
      if ((TARG[ii] > 0.0) && (ii != maxIndex))
      {
        return(false);
      }
      else if ((TARG[ii] <= 0.0) && (ii == maxIndex))
      {
        return(false);
      }
    }

    return(true);
  }
  self.isResultSampleMatchArgmaxEps = isResultSampleMatchArgmaxEps;

  function isResultSampleMatchArgmax(TARG, CALC)
  {
    var maxIndex = getMaximumIndex(CALC);

    if (maxIndex < 0) { return(false); }

    var count = TARG.length;

    assert(count == CALC.length);

    for (var ii = 0; ii < count; ii++)
    {
      if ((TARG[ii] > 0.0) && (ii != maxIndex))
      {
        return(false);
      }
      else if ((TARG[ii] <= 0.0) && (ii == maxIndex))
      {
        return(false);
      }
    }

    return(true);
  }
  self.isResultSampleMatchArgmax = isResultSampleMatchArgmax;

  // Aggregated error sum (AKA source for simple loss function)

  // Actually, simple loss function on sample is aggregated error sum divided by 2.0.
  // The divisor is need to have a "clean" partial derivative as simple difference
  // in many cases divisor ommited, as different anyway is mutiplied to small number (learning rate), but we define it here just in case

  function getResultSampleAggErrorSum(TARG, CALC)
  {
    var count = TARG.length;

    assert(count == CALC.length);

    var result = 0;

    for (var ii = 0; ii < count; ii++)
    {
      var diff = TARG[ii] - CALC[ii];

      result += diff * diff;
    }

    return(result);
  }
  self.getResultSampleAggErrorSum = getResultSampleAggErrorSum;

  // AggSum to SimpleLoss mutiplier: to be used as loss function (error function), should be mutiplied by 1/2 so derivative will not have 2x in front

  var AGG_ERROR_SUM_TO_SIMPLE_LOSS_MULTIPLY_BY = 0.5;
  self.AGG_ERROR_SUM_TO_SIMPLE_LOSS_MULTIPLY_BY = AGG_ERROR_SUM_TO_SIMPLE_LOSS_MULTIPLY_BY;

  function getResultSampleSimpleLoss(TARG, CALC)
  {
    return getResultSampleAggErrorSum(TARG, CALC) * AGG_ERROR_SUM_TO_SIMPLE_LOSS_MULTIPLY_BY;
  }
  self.getResultSampleSimpleLoss = getResultSampleSimpleLoss;

  // Mean Squared error

  function getResultSampleMSE(TARG, CALC)
  {
    var result = getResultSampleAggErrorSum(TARG, CALC);
    var count = TARG.length;
    assert(count == CALC.length);
    if (count <= 0) { return NaN; }
    return(result / count);
  }
  self.getResultSampleMSE = getResultSampleMSE;

  function getResultMSEByAggErrorSum(sum, sampleSize, samplesCount)
  {
    if (sampleSize == null) { return NaN; }
    if (samplesCount == null) { samplesCount = 1; }
    var count = samplesCount;
    if (count > 0) { count *= sampleSize; }
    if (count <= 0) { return NaN; }
    return(sum / count);
  }
  self.getResultMSEByAggErrorSum = getResultMSEByAggErrorSum;

  // Aggregated error (AKA MSE rooted)

  function getResultAggErrorByAggErrorSum(sum, sampleSize, samplesCount)
  {
    return Math.sqrt(getResultMSEByAggErrorSum(sum, sampleSize, samplesCount));
  }
  self.getResultAggErrorByAggErrorSum = getResultAggErrorByAggErrorSum;

  // Misc

  /// Retuns array with only one index of total item set to SET(=1) and all other as NOTSET(=0):
  /// Example: 0=[1, 0, 0 ...], 1=[0, 1, 0, ...], 2=[0, 0, 1, ...]
  function getR1Array(index, total, SET, NOTSET)
  {
    if (SET    == null) { SET    = 1; }
    if (NOTSET == null) { NOTSET = 0; }

    var R = [];

    for (var i = 0; i < total; i++)
    {
      R.push(i == index ? SET : NOTSET);
    }

    return(R);
  }
  self.getR1Array = getR1Array;

  function getNetWeightsCount(NET)
  {
    var result = 0;
    var layersCount = NET.layers.length;
    for (var i = 0; i < layersCount; i++) // TODO: may skip input layer later
    {
      var neuronsCount = NET.layers[i].neurons.length;
      for (var ii = 0; ii < neuronsCount; ii++)
      {
        var neuron = dynamicCastProcNeuron(NET.layers[i].neurons[ii]);
        if (neuron != null) // proc neuron
        {
          result += neuron.w.length;
        }
      }
    }
    return result;
  }
  self.getNetWeightsCount = getNetWeightsCount;

  function getNetNeuronsCount(NET)
  {
    var result = 0;
    var layersCount = NET.layers.length;
    for (var i = 0; i < layersCount; i++)
    {
      result += NET.layers[i].neurons.length;
    }
    return result;
  }
  self.getNetNeuronsCount = getNetNeuronsCount;

  return self;
}();

// Train functions
// -----------------------------------------------
// Do network train

var DEFAULT_MAX_EPOCH_COUNT = 50000;
var DEFAULT_TRAINING_SPEED  = 0.125;

/// Training parameters
function TrainingParams(speed, maxEpochCount, fastVerify)
{
  var that = this || {};

  that.speed = DEFAULT_TRAINING_SPEED;
  that.maxEpochCount = DEFAULT_MAX_EPOCH_COUNT;
  that.fastVerify = false;

  // ctor

  if ((speed == null) || (speed <= 0)) { speed = DEFAULT_TRAINING_SPEED; }
  if ((maxEpochCount == null) || (maxEpochCount <= 0)) { maxEpochCount = DEFAULT_MAX_EPOCH_COUNT; }
  if (fastVerify == null) { fastVerify = false; }

  that.speed = speed;
  that.maxEpochCount = maxEpochCount;
  that.fastVerify = fastVerify;

  return that;
}

/// Data information
function TrainingDatasetInfo()
{
  var that = this || {};

  that.trainDatasetSize = 0;

  return that;
}

/// Generic class for training lifecycle
function TrainingProccessor()
{
  var that = this || {};

  /// <summary> Start the whole training session </summary>
  that.trainStart = function(NET, trainingParams, datasetInfo) { }

  /// <summary> Start training eposh (each pass over the whole dataset) </summary>
  that.trainEposhStart = function(NET, eposhIndex) { }

  /// <summary> End training eposh (each pass over the whole dataset) </summary>
  that.trainEposhEnd = function(NET, eposhIndex) { }

  /// <summary> End the whole training session </summary>
  that.trainEnd = function(NET, isDone) { }

  return that;
};

/// Base class for training alogorith implementation
function NetworkTrainer()
{
  var that = this || {};
  var base = TrainingProccessor; base.call(that);

  /// <summary> Do single training step. Returs result of pre-training run with DATA </summary>
  that.trainBySample = function (NET, DATA, TARG, speed) { throw new Error("trainBySample not implemented"); }

  return that;
};

/// Class checker for training is done
function TrainingDoneChecker()
{
  var that = this || {};
  var base = TrainingProccessor; base.call(that);

  /// <summary> Check is sample is valid. If all samples are valid, assumed that training is complete </summary>
  that.trainSampleIsValid = function(NET, TARG, CALC) { return false; }

  /// <summary> Check that eposh result is valid. If this function returns true, training assumed complete </summary>
  that.trainEpochIsValid = function(NET, epochIndex) { return false; }

  return that;
};

/// Training progress reporter
/// Will be called during traing to report progress
/// May rise traing abort event

function TrainingProgressReporter()
{
  var that = this || {};
  var base = TrainingProccessor; base.call(that);

  /// Reports or updates stats by sample. if returns false, training will be aborted
  that.trainSampleReportAndCheckContinue = function(NET, DATA, TARG, CALC, epochIndex, sampleIndex)
  {
    return true;
  }

  return that;
};

// Training progress void (does nothing) implementation

function TrainingProgressReporterVoid()
{
  var that = this || {};
  var base = TrainingProgressReporter; base.call(that);

  // Does nothing

  return that;
};

// Back propagation training alogorithm implementation

function NetworkTrainerBackProp()
{
  var that = this || {};
  var base = NetworkTrainer; base.call(that);

  // Use "/" instead of * during train. Used in unit tests only, should be false on production
  that.DIV_IN_TRAIN = false;

  function getDeltaOutputSum(outNeuron, osme) // osme = output sum margin of error (AKA Expected - Calculated)
  {
    if ((outNeuron == null) || (outNeuron.getSum == null)) { return NaN; }
    var os = outNeuron.getSum();
    var dos = outNeuron.SD(os) * osme;
    return(dos);
  }

  function getDeltaWeights(theNeuron, dos) // theNeuron in question, dos = delta output sum
  {
    if ((theNeuron == null) || (theNeuron.inputs == null)) { return []; } // Empty

    var count = theNeuron.inputs.length;
    var DWS = [];

    var dw;
    for (var i = 0; i < count; i++)
    {
      if (that.DIV_IN_TRAIN) { dw = dos / theNeuron.inputs[i].get(); } else { dw = dos * theNeuron.inputs[i].get(); }
      DWS.push(dw);
    }

    return(DWS);
  }

  function getDeltaHiddenSums(theNeuron, dos)
  {
    if ((theNeuron == null) || (theNeuron.inputs == null)) { return []; } // Empty

    var count = theNeuron.inputs.length;
    var DHS = [];

    var ds;
    for (var i = 0; i < count; i++)
    {
      var input = dynamicCastProcNeuronTrainee(theNeuron.inputs[i]);
      if (input == null)
      {
        ds = NaN; // This neuron input is non-trainee neuron, ds is N/A since we do not know its getSum()
      }
      else // looks like SD here is SD for input neuron (?) use input->SD(input->getSum()) later
      {
        if (that.DIV_IN_TRAIN) { ds = dos / theNeuron.w[i] * input.SD(input.getSum()); } else { ds = dos * theNeuron.w[i] * input.SD(input.getSum()); }
      }

      DHS.push(ds);
    }

    return(DHS);
  }

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
      var neuron = dynamicCastProcNeuronTrainee(LOUT[i]);

      if (neuron == null) { break; } // Non-trainable neuron

      var LP = neuron.inputs;
      var DOHS = getDeltaHiddenSums(neuron, DOS[i]);

      assert(LP.length == DOHS.length);

      for (var ii = 0; ii < LP.length; ii++)
      {
        var input = dynamicCastProcNeuronTrainee(LP[ii]);
        if (input != null)
        {
          var DW = getDeltaWeights(input, DOHS[ii]);
          input.addNewWeightsDelta(DW);
        }
      }

      doTrainStepProcPrevLayer(LP, DOHS, layerIndex-1);
    }
  }

  function trainBySample(NET, DATA, TARG, speed)
  {
    // NET=network, DATA=input, TARG=expeted
    // CALC=calculated output (will be calculated)
    // Note: we re-run calculation here both to receive CALC AND update "sum" state of each neuron in NET

    if ((speed == null) || (isNaN(speed)) || (speed <= 0.0)) { speed = 0.1; } // 1.0 is max

    var CALC = doProc(NET, DATA); // we need this because sum has to be updated in NET for each neuron for THIS test case

    for (var i = 1; i < NET.layers.length; i++) // skip input layer
    {
      var iicount = NET.layers[i].neurons.length;
      for (var ii = 0; ii < iicount; ii++)
      {
        var neuron = dynamicCastProcNeuronTrainee(NET.layers[i].neurons[ii]);
        if (neuron != null)
        {
          neuron.initTrainStep(); // prepare
        }
      }
    }

    // Output layer (special handling)

    var LOUT = NET.layers[NET.layers.length-1].neurons;

    var OSME = []; // output sum margin of error (AKA Expected - Calculated) for each output
    var DOS  = []; // delta output sum for each output neuron
    var DOIW = []; // delta output neuron input weights each output neuron

    // proc output layer

    for (var i = 0; i < LOUT.length; i++)
    {
      var neuron = dynamicCastProcNeuronTrainee(LOUT[i]);
      OSME.push((TARG[i] - CALC[i]) * speed);
      DOS.push(getDeltaOutputSum(neuron, OSME[i])); // will handle neuron=NULL case
      DOIW.push(getDeltaWeights(neuron, DOS[i])); // will handle neuron=NULL case
      if (neuron != null)
      {
        neuron.addNewWeightsDelta(DOIW[i]);
      }
    }

    // proc prev layers
    // will apply training back recursively
    // recursion controlled by laterIndex

    doTrainStepProcPrevLayer(LOUT, DOS, NET.layers.length-1);

    for (var i = 1; i < NET.layers.length; i++) // skip input layer
    {
      var iicount = NET.layers[i].neurons.length;
      for (var ii = 0; ii < iicount; ii++)
      {
        var neuron = dynamicCastProcNeuronTrainee(NET.layers[i].neurons[ii]);
        if (neuron != null)
        {
          neuron.applyNewWeights(); // adjust
        }
      }
    }

    return CALC;
  }

  // Export

  that.trainBySample = trainBySample;

  // Extra export (for unit tests)

  that.getDeltaOutputSum  = getDeltaOutputSum;
  that.getDeltaWeights    = getDeltaWeights;
  that.getDeltaHiddenSums = getDeltaHiddenSums;

  return that;
}

// Back propagation training fast alogorithm implemenation

function NetworkTrainerBackPropFast()
{
  var that = this || {};
  var base = NetworkTrainer; base.call(that);

  // Use "/" instead of * during train. Used in unit tests only, should be false on production
  that.DIV_IN_TRAIN = false;
  
  function addDeltaOutputSum(outNeuron, osme) // osme = output sum margin of error (AKA Expected - Calculated) // FAST
  {
    if ((outNeuron == null) || (outNeuron.getSum == null)) { return; }
    var dos = outNeuron.SD(outNeuron.getSum()) * osme;
    outNeuron.addDeltaOutputSum(dos);
  }

  function addDeltaWeights(theNeuron, dos) // theNeuron in question, dos = delta output sum // FAST
  {
    if ((theNeuron == null) || (theNeuron.inputs == null)) { return; } // Empty

    var count = theNeuron.inputs.length;

    var dw;
    for (var i = 0; i < count; i++)
    {
      if (that.DIV_IN_TRAIN) { dw = dos / theNeuron.inputs[i].get(); } else { dw = dos * theNeuron.inputs[i].get(); }
      theNeuron.nw[i] += dw;
    }
  }

  function addDeltaHiddenSums(theNeuron, dos) // FAST
  {
    if ((theNeuron == null) || (theNeuron.inputs == null)) { return; } // Empty

    var count = theNeuron.inputs.length;

    var ds;
    for (var i = 0; i < count; i++)
    {
      var input = dynamicCastProcNeuronTrainee(theNeuron.inputs[i]);
      if (input == null)
      {
        // This neuron input is non-trainee neuron, ds is N/A since we do not know its getSum()
      }
      else // looks like SD here is SD for input neuron (?) use input->SD(input->getSum()) later
      {
        if (that.DIV_IN_TRAIN) { ds = dos / theNeuron.w[i] * input.SD(input.getSum()); } else { ds = dos * theNeuron.w[i] * input.SD(input.getSum()); }
        input.addDeltaOutputSum(ds);
      }
    }
  }

  function trainBySample(NET, DATA, TARG, speed)
  {
    // NET=network, DATA=input, TARG=expeted
    // CALC=calculated output (will be calculated)
    // Note: we re-run calculation here both to receive CALC AND update "sum" state of each neuron in NET

    if ((speed == null) || (isNaN(speed)) || (speed <= 0.0)) { speed = 0.1; } // 1.0 is max

    var CALC = doProc(NET, DATA); // we need this because sum has to be updated in NET for each neuron for THIS test case

    for (var i = 1; i < NET.layers.length; i++) // skip input layer
    {
      var iicount = NET.layers[i].neurons.length;
      for (var ii = 0; ii < iicount; ii++)
      {
        var neuron = dynamicCastProcNeuronTrainee(NET.layers[i].neurons[ii]);
        if (neuron != null)
        {
          neuron.initTrainStep(); // prepare
        }
      }
    }

    // proc output layer (special handling)

    if (NET.layers.length > 0)
    {
      var LOUT = NET.layers[NET.layers.length - 1].neurons;
      for (var i = 0; i < LOUT.length; i++)
      {
        var neuron = dynamicCastProcNeuronTrainee(LOUT[i]);
        if (neuron != null)
        {
          var osme = (TARG[i] - CALC[i]) * speed;
          addDeltaOutputSum(neuron, osme);
          addDeltaWeights(neuron, neuron.getDeltaOutputSum());
          addDeltaHiddenSums(neuron, neuron.getDeltaOutputSum());
        }
      }
    }

    // proc hidden layers, skip input layer

    for (var li = NET.layers.length - 2; li > 0; li--)
    {
      var LOUT = NET.layers[li].neurons;
      for (var i = 0; i < LOUT.length; i++)
      {
        var neuron = dynamicCastProcNeuronTrainee(LOUT[i]);
        if (neuron != null)
        {
          addDeltaWeights(neuron, neuron.getDeltaOutputSum());
          addDeltaHiddenSums(neuron, neuron.getDeltaOutputSum());
        }
      }
    }

    for (var i = 1; i < NET.layers.length; i++) // skip input layer
    {
      var iicount = NET.layers[i].neurons.length;
      for (var ii = 0; ii < iicount; ii++)
      {
        var neuron = dynamicCastProcNeuronTrainee(NET.layers[i].neurons[ii]);
        if (neuron != null)
        {
          neuron.applyNewWeights(); // adjust
        }
      }
    }
  }

  // Export

  that.trainBySample = trainBySample;

  return that;
}

var defTrainer_ = new NetworkTrainerBackPropFast();
function getDefTrainer() { return defTrainer_; }

// Class checker for training is done (by results differ from groud truth no more than eps) implemenation

function TrainingDoneCheckerEps(eps)
{
  var that = this || {};
  var base = TrainingDoneChecker; base.call(that);

  var DEFAULT_EPS = NetworkStat.DEFAULT_EPS;

  // Constructor

  if ((eps == null) || isNaN(eps) || (eps <= 0.0)) { eps = DEFAULT_EPS; } // > 0.0 and < 0.5

  that.trainSampleIsValid = function(NET, TARG, CALC)
  //override
  {
    return NetworkStat.isResultSampleMatchEps(TARG, CALC, eps);
  }
  
  return that;
}

// Main training function

/// Train the neural network
function doTrain(NET, DATAS, TARGS, trainingParams, trainingProgressReporter, trainingDoneChecker, trainer)
{
  if (trainingParams == null) { trainingParams = new TrainingParams; }

  if (trainingProgressReporter == null) { trainingProgressReporter = new TrainingProgressReporterVoid(); }

  if (trainingDoneChecker == null) { trainingDoneChecker = new TrainingDoneCheckerEps(); }

  if (trainer == null) { trainer = getDefTrainer(); }

  var trainingDatasetInfo = new TrainingDatasetInfo();
  trainingDatasetInfo.trainDatasetSize = DATAS.length;

  trainer.trainStart(NET, trainingParams, trainingDatasetInfo);
  trainingDoneChecker.trainStart(NET, trainingParams, trainingDatasetInfo);
  trainingProgressReporter.trainStart(NET, trainingParams, trainingDatasetInfo);

  var MaxN = trainingParams.maxEpochCount;
  var speed = trainingParams.speed;

  // steps
  var isDone = false;
  var isAbort = false;
  for (var n = 0; (n < MaxN) && (!isDone) && (!isAbort); n++)
  {
    trainer.trainEposhStart(NET, n);
    trainingDoneChecker.trainEposhStart(NET, n);
    trainingProgressReporter.trainEposhStart(NET, n);

    if (!trainingParams.fastVerify)
    {
      // strict verify
      isDone = true;
      for (var s = 0; (s < DATAS.length) && (!isAbort); s++)
      {
        var CALC = doProc(NET, DATAS[s]);

        if (!trainingDoneChecker.trainSampleIsValid(NET, TARGS[s], CALC))
        {
          isDone = false;
        }

        if (!trainingProgressReporter.trainSampleReportAndCheckContinue(NET, DATAS[s], TARGS[s], CALC, n, s))
        {
          isDone = false;
          isAbort = true;
        }
      }

      if ((!isDone) && (!isAbort))
      {
        for (var s = 0; s < DATAS.length; s++)
        {
          trainer.trainBySample(NET, DATAS[s], TARGS[s], speed);
        }
      }
    }
    else
    {
      // fast verify
      isDone = true;
      for (var s = 0; (s < DATAS.length) && (!isAbort); s++)
      {
        var CALC_BEFORE_TRAIN = trainer.trainBySample(NET, DATAS[s], TARGS[s], speed);

        // we use calc before train as input
        // strictly speaking this is incorrect, 
        // but we assume that single training step will not affect stats very much and will only improve results

        var CALC = CALC_BEFORE_TRAIN;

        if (!trainingDoneChecker.trainSampleIsValid(NET, TARGS[s], CALC))
        {
          isDone = false;
        }

        if (!trainingProgressReporter.trainSampleReportAndCheckContinue(NET, DATAS[s], TARGS[s], CALC, n, s))
        {
          isDone = false;
          isAbort = true;
        }
      }
    }

    trainer.trainEposhEnd(NET, n);
    trainingDoneChecker.trainEposhEnd(NET, n);
    trainingProgressReporter.trainEposhEnd(NET, n);
  }

  trainer.trainEnd(NET, isDone);
  trainingDoneChecker.trainEnd(NET, isDone);
  trainingProgressReporter.trainEnd(NET, isDone);

  return (isDone);
}

// Exports

// Some internals

NN.Internal = {};
NN.Internal.getPRNG = getPRNG;
NN.Internal.getRandom = getRandom;
NN.Internal.getRandomInt = getRandomInt;

// Activation

NN.ActFuncSigmoid = ActFuncSigmoid;
NN.ActFuncSigmoidTrainee = ActFuncSigmoidTrainee;
NN.ActFuncRELU = ActFuncRELU;
NN.ActFuncRELUTrainee = ActFuncRELUTrainee;
NN.ActFuncLRELU = ActFuncLRELU;
NN.ActFuncLRELUTrainee = ActFuncLRELUTrainee;
NN.ActFuncTanh = ActFuncTanh;
NN.ActFuncTanhTrainee = ActFuncTanhTrainee;

// Core

NN.BaseNeuron = BaseNeuron;
NN.InputNeuron = InputNeuron; NN.dynamicCastInputNeuron = dynamicCastInputNeuron;
NN.ProcNeuron = ProcNeuron; NN.dynamicCastProcNeuron = dynamicCastProcNeuron;
NN.ProcNeuronTrainee = ProcNeuronTrainee; NN.dynamicCastProcNeuronTrainee = dynamicCastProcNeuronTrainee;
NN.BiasNeuron = BiasNeuron;

NN.Layer       = Layer;
NN.Network     = Network;
NN.doProc      = doProc;

NN.TheNeuronFactory = TheNeuronFactory;
NN.ExtNeuronFactory = ExtNeuronFactory;

// Math

NN.NetworkStat = NetworkStat;

// Training
NN.TrainingParams = TrainingParams;
NN.TrainingDatasetInfo = TrainingDatasetInfo;
NN.TrainingProccessor = TrainingProccessor;
NN.TrainingDoneChecker = TrainingDoneChecker;
NN.TrainingProgressReporter = TrainingProgressReporter;
NN.TrainingProgressReporterVoid = TrainingProgressReporterVoid;
NN.NetworkTrainer = NetworkTrainer;
NN.NetworkTrainerBackProp = NetworkTrainerBackProp;
NN.NetworkTrainerBackPropFast = NetworkTrainerBackPropFast;
NN.getDefTrainer = getDefTrainer;
NN.TrainingDoneCheckerEps = TrainingDoneCheckerEps;
NN.TrainingProgressReporter = TrainingProgressReporter;
NN.doTrain = doTrain;

}()
