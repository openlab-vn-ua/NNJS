// JavaScript Simple Neural Network toolkit
// [Unit test]

// Require nnjs.js

// Test case(s)
// ---------------------------------

function doUnitTest1()
{
  var ODT = NN.DIV_IN_TRAIN;

  NN.DIV_IN_TRAIN = true;

  var IN  = new NN.Layer(2, NN.InputNeuron);

  var L1  = new NN.Layer(3, NN.Neuron); 
  //L1.addInputAll(IN);
  L1.neurons[0].addInput(IN.neurons[0], 0.8);
  L1.neurons[0].addInput(IN.neurons[1], 0.2);
  L1.neurons[1].addInput(IN.neurons[0], 0.4);
  L1.neurons[1].addInput(IN.neurons[1], 0.9);
  L1.neurons[2].addInput(IN.neurons[0], 0.3);
  L1.neurons[2].addInput(IN.neurons[1], 0.5);

  var OUT = new NN.Layer(1, NN.Neuron); 
  //OUT.addInputAll(L1);
  OUT.neurons[0].addInput(L1.neurons[0], 0.3);
  OUT.neurons[0].addInput(L1.neurons[1], 0.5);
  OUT.neurons[0].addInput(L1.neurons[2], 0.9);

  var NET = [IN, L1, OUT];

  var DATA = [1, 1]; // Input
  var TARG = [0]; // Expected output

  var CALC = NN.doProc(NET, DATA)[0]; // Actual output
  console.log('Result', CALC);

  // Adjust Output layer

  var OSME = TARG - CALC;
  console.log('output sum margin of error', OSME);

  var DOS = NN.Internal.getDeltaOutputSum(OUT.neurons[0], OSME);
  console.log('delta output sum', DOS); // How much sum have to be adjusted

  var pOut = OUT.neurons[0].inputs; // Pre-output layer (L1)
  var DWS = NN.Internal.getDeltaWeights(OUT.neurons[0], DOS);
  console.log('delta weights', DWS); // How much w of prev neurons have to be adjusted

  OUT.neurons[0].initNewWeights();
  OUT.neurons[0].addNewWeightsDelta(DWS);

  var NWS = OUT.neurons[0].nw;
  console.log('new weights', NWS); // New w of output

  // calclulate how to change outputs of prev layer (DOS for each neuton of prev layer)
  // DOS is delta output sum for this neuron

  var DHS = NN.Internal.getDeltaHiddenSums(OUT.neurons[0], DOS);

  console.log('delta hidden sums', DHS); // array of DOS for prev layer

  // Proc the hidden layer

  var DWSL1 = [];
  var NWSL1 = [];

  for (var i = 0; i < pOut.length; i++)
  {
    DWSL1.push(NN.Internal.getDeltaWeights(L1.neurons[i], DHS[i]));
    L1.neurons[i].initNewWeights(); // would work this way since only one output neuron (so will be called once for each hidden neuron)
    L1.neurons[i].addNewWeightsDelta(DWSL1[i]);
    NWSL1 = L1.neurons[i].nw;
  }

  console.log('delta weights L1', [DWSL1]); // array of DOS for prev layer

  // assign

  OUT.neurons[0].applyNewWeights();

  for (var i = 0; i < pOut.length; i++)
  {
    L1.neurons[i].applyNewWeights();
  }

  var CALC2 = NN.doProc(NET, DATA); // Actual output
  console.log('Result after adjust', CALC2); // should be 0.6917258326007417

  if (CALC2[0] == 0.6917258326007417)
  {
    console.log('UNIT TEST OK', NET);
  }

  NN.DIV_IN_TRAIN = ODT;
}

function runUnitTests()
{
  doUnitTest1();
}
