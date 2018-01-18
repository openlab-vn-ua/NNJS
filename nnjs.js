// JavaScript Simple Neural Network toolkit
// [CORE]
// Initally Written by Serge Ageyev
// Open Source Software under BSD License

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
  return (PRNG.nextFloat() * (max - min)) + min;
}

function getRandomInt (min, max)
{
  return Math.ceil((PRNG.nextFloat() * (max - min)) + min) % max;
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

function Neuron()
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

  if (maker == null) { maker = Neuron; }

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
  DOS = SD(OS) * OSME;
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

function doTrainStep(NET, DATA, TARG, CALC, SPEED)
{
  // NET=network, DATA=input, TARG=expeted, CALC=calculated

  if (SPEED == null) { SPEED = 1.0; }

  var CALC = doProc(NET, DATA); // not we need this because sum has to be updated for each neuron

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

function doTrain(NET, DATAS, TARGS, SPEED, MAX_N, REP_N, isTrainDoneFunc)
{
  function isTrainDoneDefaultFunc(DATAS, TARGS, CALCS)
  {
    function isResultItemMatch(t,c)
    {
      if (Math.abs(t-c) < 0.1) { return(true); }
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

  if (MAX_N == null)       { MAX_N = 50000; }
  if (REP_N == null)       { REP_N =   100; } // report interval
  if (SPEED == null)       { SPEED = 0.125; }

  if (isTrainDoneFunc == null) { isTrainDoneFunc = isTrainDoneDefaultFunc; }

  var isDone = false;
  for (var n = 0; (n < MAX_N) && (!isDone); n++)
  {
    var CALCS = [];
    for (var s = 0; s < DATAS.length; s++)
    {
      CALCS.push(doProc(NET, DATAS[s])); // Fill output
    }

    isDone = isTrainDoneFunc(DATAS, TARGS, CALCS);

    if ((REP_N != null) && (REP_N != 0))
    {
      if (((n % REP_N) == 0) || (isDone))
      {
        for (var s = 0; s < DATAS.length; s++)
        {
          console.log('Result.N[n,s]', MAX_N, n, s, DATAS[s], TARGS[s], CALCS[s]);
        }
      }
    }

    if (!isDone)
    {
      for (var s = 0; s < DATAS.length; s++)
      {
        doTrainStep(NET, DATAS[s], TARGS[s], SPEED);
      }
    }
  }

  if (isDone)
  {
    if ((REP_N != null) && (REP_N != 0)) { console.log('Training OK', 'iterations:'+n, NET); }
    return(true);
  }
  else
  {
    if ((REP_N != null) && (REP_N != 0)) { console.log('Training FAILED', 'timeout:'+MAX_N, NET ); }
    return(false);
  }
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

NCore.Neuron = Neuron;
NCore.InputNeuron = InputNeuron;
NCore.BiasNeuron = BiasNeuron;

NCore.Layer   = Layer;

NCore.doProc  = doProc;
NCore.doTrain = doTrain;

}()
