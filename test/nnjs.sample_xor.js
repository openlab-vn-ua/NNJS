// JavaScript Simple Neural Network toolkit
// Open Source Software under MIT License
// [Xor demo network]

// Require nnjs.js
// Require nnjs.console.training.stat.js

function sampleXorNetwork()
{
  if (true)
  {
    var seed = new Date().getTime() % 0x7FFF0000 + 1;
    NN.Internal.PRNG.setSeed(seed);
    console.log("sampleXorNetwork","seed=",seed);
  }

  var IN  = new NN.Layer(2, NN.TheNeuronFactory(NN.InputNeuron)); IN.addNeuron(new NN.BiasNeuron());
  var L1  = new NN.Layer(2, NN.TheNeuronFactory(NN.ProcNeuronTrainee)); L1.addNeuron(new NN.BiasNeuron());
  L1.addInputAll(IN);
  var OUT = new NN.Layer(1, NN.TheNeuronFactory(NN.ProcNeuronTrainee)); 
  OUT.addInputAll(L1);
  var NET = new NN.Network(); NET.addLayer(IN); NET.addLayer(L1); NET.addLayer(OUT);

  var DATAS = [ [1, 1], [1, 0], [0, 1], [0, 0]];
  var TARGS = [    [0],    [1],    [1],    [0]];

  return NN.doTrain(NET, DATAS, TARGS, -1, -1, new NN.TrainingProgressReporterConsole(1000));
}

function sampleXorNetwork2()
{
  if (true)
  {
    var seed = new Date().getTime() % 0x7FFF0000 + 1;
    NN.Internal.PRNG.setSeed(seed);
    console.log("sampleXorNetwork2","seed=",seed);
  }

  var IN  = new NN.Layer(2, NN.TheNeuronFactory(NN.InputNeuron)); IN.addNeuron(new NN.BiasNeuron());
  var L1  = new NN.Layer(3, NN.TheNeuronFactory(NN.ProcNeuronTrainee)); L1.addNeuron(new NN.BiasNeuron());
  L1.addInputAll(IN);
  var L2  = new NN.Layer(3, NN.TheNeuronFactory(NN.ProcNeuronTrainee)); L2.addNeuron(new NN.BiasNeuron());
  L2.addInputAll(L1);
  var OUT = new NN.Layer(1, NN.TheNeuronFactory(NN.ProcNeuronTrainee)); 
  OUT.addInputAll(L2);
  var NET = new NN.Network(); NET.addLayer(IN); NET.addLayer(L1); NET.addLayer(L2); NET.addLayer(OUT);

  var DATAS = [ [1, 1], [1, 0], [0, 1], [0, 0]];
  var TARGS = [    [0],    [1],    [1],    [0]];

  return NN.doTrain(NET, DATAS, TARGS, -1, -1, new NN.TrainingProgressReporterConsole(1000));
}
