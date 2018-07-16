// JavaScript Simple Neural Network toolkit
// [Xor demo network]

// Require nnjs.js

function sampleXorNetwork()
{
  var IN  = new NN.Layer(2, NN.InputNeuron); IN.addNeuron(NN.BiasNeuron);
  var L1  = new NN.Layer(2, NN.Neuron); L1.addNeuron(NN.BiasNeuron);
  L1.addInputAll(IN);
  var OUT = new NN.Layer(1, NN.Neuron); 
  OUT.addInputAll(L1);
  var NET = [IN, L1, OUT];

  var DATAS = [ [1, 1], [1, 0], [0, 1], [0, 0]];
  var TARGS = [    [0],    [1],    [1],    [0]];

  NN.doTrain(NET, DATAS, TARGS);
}

//sampleXorNetwork();

function sampleXorNetwork2()
{
  var IN  = new NN.Layer(2, NN.InputNeuron); IN.addNeuron(NN.BiasNeuron);
  var L1  = new NN.Layer(3, NN.Neuron); L1.addNeuron(NN.BiasNeuron);
  L1.addInputAll(IN);
  var L2  = new NN.Layer(3, NN.Neuron); L2.addNeuron(NN.BiasNeuron);
  L2.addInputAll(L1);
  var OUT = new NN.Layer(1, NN.Neuron); 
  OUT.addInputAll(L2);
  var NET = [IN, L1, L2, OUT];

  var DATAS = [ [1, 1], [1, 0], [0, 1], [0, 0]];
  var TARGS = [    [0],    [1],    [1],    [0]];

  NN.doTrain(NET, DATAS, TARGS);
}

//doXorNetwork2();
