// JavaScript Simple Neural Network toolkit
// Open Source Software under MIT License
// [Unit test]

// Utils
// ----------------------------------------------------

function STR(x) { return "" + x; }

function isFloatAlmostEqual(a,b,eps)
{
  if (eps == null) { eps = 0.0001; }
  var dif = a-b;
  if (dif < 0) { dif = -dif; }
  return (dif <= eps);
}

function isFloatListAlmostEqual(a,b,eps)
{
  if (a.length != b.length) { return false; }
  var count = a.length;
  for (var i = 0; i < count; i++)
  {
    if (!isFloatAlmostEqual(a[i],b[i],eps)) { return false; }
  }
  return true;
}

// Test case(s) [NN]
// ----------------------------------------------------
// require nnjs.js

function doUnitTest1()
{
  // Test case based on
  // http://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
  // Note: For some reason original test uses division by S' instead of S' multiplication during train calculation 
  // That is why we use DIV_IN_TRAIN=true here to keep clculation consistent with the original good explaied test (basides the trick with "/")
  // Mode DIV_IN_TRAIN=true intended to use for this test only, during production work we use is as false

  var isOk = true;

  var ODT = NN.DIV_IN_TRAIN;

  NN.DIV_IN_TRAIN = true;

  var IN  = new NN.Layer(2, NN.InputNeuron);

  var L1  = new NN.Layer(3, NN.ProcNeuron); 
  //L1.addInputAll(IN);
  L1.neurons[0].addInput(IN.neurons[0], 0.8);
  L1.neurons[0].addInput(IN.neurons[1], 0.2);
  L1.neurons[1].addInput(IN.neurons[0], 0.4);
  L1.neurons[1].addInput(IN.neurons[1], 0.9);
  L1.neurons[2].addInput(IN.neurons[0], 0.3);
  L1.neurons[2].addInput(IN.neurons[1], 0.5);

  var OUT = new NN.Layer(1, NN.ProcNeuron); 
  //OUT.addInputAll(L1);
  OUT.neurons[0].addInput(L1.neurons[0], 0.3);
  OUT.neurons[0].addInput(L1.neurons[1], 0.5);
  OUT.neurons[0].addInput(L1.neurons[2], 0.9);

  var NET = [IN, L1, OUT];

  var DATA = [1, 1]; // Input
  var TARG = [0]; // Expected output

  var CALC = NN.doProc(NET, DATA)[0]; // Actual output

  if (!isFloatAlmostEqual(CALC,0.7743802720529458))
  {
    isOk = false;
    console.log("FAIL: Result", CALC);
  }
 
  // Adjust Output layer

  var OSME = TARG[0] - CALC;

  if (!isFloatAlmostEqual(OSME,-0.7743802720529458))
  {
    isOk = false;
    console.log("FAIL: output sum margin of error", OSME);
  }

  var DOS = NN.Internal.getDeltaOutputSum(OUT.neurons[0], OSME);
  if (!isFloatAlmostEqual(DOS, -0.13529621033156358))
  {
    console.log("FAIL: delta output sum", DOS); // How much sum have to be adjusted
  }

  var pOut = OUT.neurons[0].inputs; // Pre-output layer (L1)
  var DWS = NN.Internal.getDeltaWeights(OUT.neurons[0], DOS);

  //console.log("INFO: delta weights", DWS);

  if (!isFloatListAlmostEqual(DWS, [ -0.1850689045809531, -0.1721687291239315, -0.19608871636883077 ]))
  {
    console.log("FAIL: delta weights", DWS); // How much w of prev neurons have to be adjusted
  }

  OUT.neurons[0].initNewWeights();
  OUT.neurons[0].addNewWeightsDelta(DWS);

  var NWS = OUT.neurons[0].nw;

  if (!isFloatListAlmostEqual(NWS, [ 0.11493109541904689, 0.3278312708760685, 0.7039112836311693 ]))
  {
    console.log("FAIL: new weights", NWS); // New w of output
  }

  // calclulate how to change outputs of prev layer (DOS for each neuton of prev layer)
  // DOS is delta output sum for this neuron

  var DHS = NN.Internal.getDeltaHiddenSums(OUT.neurons[0], DOS);

  if (!isFloatListAlmostEqual(DHS, [ -0.08866949824511623, -0.045540261294143396, -0.032156856991522986 ]))
  {
    console.log("FAIL: delta hidden sums", DHS); // array of DOS for prev layer
  }

  // Proc the hidden layer

  var DWSL1 = [];
  var NWSL1 = [];

  for (var i = 0; i < pOut.length; i++)
  {
    DWSL1.push(NN.Internal.getDeltaWeights(L1.neurons[i], DHS[i]));
    L1.neurons[i].initNewWeights(); // would work this way since only one output neuron (so will be called once for each hidden neuron)
    L1.neurons[i].addNewWeightsDelta(DWSL1[i]);
    NWSL1.push(L1.neurons[i].nw);
  }

  //console.log("INFO: delta weights L1", DWSL1);

  if (!isFloatListAlmostEqual(DWSL1[0], [-0.08866949824511623 , -0.08866949824511623 ]) ||
      !isFloatListAlmostEqual(DWSL1[1], [-0.045540261294143396, -0.045540261294143396]) ||
      !isFloatListAlmostEqual(DWSL1[2], [-0.032156856991522986, -0.032156856991522986]))
  {
    console.log("FAIL: delta weights L1", DWSL1); // [] array of DOS for prev layer
  }

  //console.log("INFO: new weights L1", NWSL1);

  if (!isFloatListAlmostEqual(NWSL1[0], [0.7113305017548838, 0.11133050175488378]) ||
      !isFloatListAlmostEqual(NWSL1[1], [0.3544597387058566, 0.8544597387058567 ]) ||
      !isFloatListAlmostEqual(NWSL1[2], [0.267843143008477 , 0.467843143008477  ]))
  {
    console.log("FAIL: new weights L1", NWSL1); // [] array of NW for prev layer
  }

  // assign

  OUT.neurons[0].applyNewWeights();

  for (var i = 0; i < pOut.length; i++)
  {
    L1.neurons[i].applyNewWeights();
  }

  var CALC2 = NN.doProc(NET, DATA)[0]; // Actual output

  if (!isFloatAlmostEqual(CALC2,0.6917258326007417))
  {
    isOk = false;
    console.log("FAIL: Result after adjust", CALC2); // should be 0.6917258326007417
  }

  NN.DIV_IN_TRAIN = ODT;

  return isOk;
}

// Test case(s) [PRNG]
// ----------------------------------------------------
// require prng.js

function doUnitTestRNG0()
{
  var isOk = true;
  var i = 0;
  var r;
  var TRNG = new Random(1);
  while (isOk)
  {
    i++;
    r = TRNG.next();

    if (i == 1) { isOk = (16807 == r); }
    if (i == 2) { isOk = (282475249 == r); }
    if (i == 3) { isOk = (1622650073 == r); }
    if (i == 4) { isOk = (984943658 == r); }
    if (i == 5) { isOk = (1144108930 == r); }
    if (i == 6) { isOk = (470211272 == r); }
    if (i == 7) { isOk = (101027544 == r); }
    if (i == 8) { isOk = (1457850878 == r); }
    if (i == 9) { isOk = (1458777923 == r); }
    if (i == 10) { isOk = (2007237709 == r); }

    if (i == 9998) { isOk = (925166085 == r); }
    if (i == 9999) { isOk = (1484786315 == r); }
    if (i == 10000) { isOk = (1043618065 == r); }
    if (i == 10001) { isOk = (1589873406 == r); }
    if (i == 10002) { isOk = (2010798668 == r); }

    if (i == 1000000) { isOk = (1227283347 == r); }
    if (i == 2000000) { isOk = (1808217256 == r); }
    if (i == 3000000) { isOk = (1140279430 == r); }
    if (i == 4000000) { isOk = (851767375 == r); }
    if (i == 5000000) { isOk = (1885818104 == r); }

    if (i == 99000000) { isOk = (168075678 == r); }
    if (i == 100000000) { isOk = (1209575029 == r); }
    if (i == 101000000) { isOk = (941596188 == r); }

    if (i == 2147483643) { isOk = (1207672015 == r); }
    if (i == 2147483644) { isOk = (1475608308 == r); }
    if (i == 2147483645) { isOk = (1407677000 == r); }

    // Starting the sequence again with the original seed

    if (i == 2147483646) { isOk = (1 == TRNG.next()); }
    if (i == 2147483647) { isOk = (16807 == TRNG.next()); }

    if (i > 2000000) { break; } // if you no not want to wait too long
  }
  return(isOk);
}

function doUnitTestRNG1()
{
  var isOk = true;
  var TRNG = new Random(42);
  if (isOk) { isOk = (705894     == TRNG.next()); }
  if (isOk) { isOk = (1126542223 == TRNG.next()); }
  if (isOk) { isOk = (1579310009 == TRNG.next()); }
  if (isOk) { isOk = (565444343  == TRNG.next()); }
  if (isOk) { isOk = (807934826  == TRNG.next()); }
  return(isOk);
}

function doUnitTestRNG2()
{
  var isOk = true;
  var TRNG = new Random(42);
  if (isOk) { isOk = isFloatAlmostEqual(0.0003287070433876543 , TRNG.nextFloat()); }
  if (isOk) { isOk = isFloatAlmostEqual(0.5245871017916008    , TRNG.nextFloat()); }
  if (isOk) { isOk = isFloatAlmostEqual(0.7354235320681926    , TRNG.nextFloat()); }
  if (isOk) { isOk = isFloatAlmostEqual(0.26330554044182      , TRNG.nextFloat()); }
  if (isOk) { isOk = isFloatAlmostEqual(0.3762239710206389    , TRNG.nextFloat()); }
  return(isOk);
}

var TEST_RNG_MAX_COUNT = 1000000;

function getTestRNGCountSeed()
{ 
  return(42); 
}

function doUnitTestRNG3()
{
  var NAME = STR("RNG3:");
  var isOk = true;
  var TRNG = new Random(getTestRNGCountSeed());
  var r;
  var cmin = 0;
  for (var i = 0; i < TEST_RNG_MAX_COUNT; i++)
  {
    r = TRNG.nextFloat();
    if (r < 0) { isOk = false; break; }
    if (r == 0) { cmin++; }
    if (r >= 1.0) { isOk = false; break; } // 1.0 not inclusive
  }
  //if (isOk) { if (cmin <= 0) { console.log(NAME+"WARN: no min found"); } }
  if (!isOk) { console.log(NAME+"FAIL", r); }
  return(isOk);
}

function doUnitTestRNG4()
{
  var NAME = STR("RNG4:");
  var isOk = true;
  var TRNG = new Random(getTestRNGCountSeed());
  var r;
  var cmin = 0;
  var cmax = 0;
  for (var i = 0; i < TEST_RNG_MAX_COUNT; i++)
  {
    r = TRNG.randFloat();
    if (r < 0) { isOk = false; break; }
    if (r == 0) { cmin++; }
    if (r > 1.0) { isOk = false; break; }
    if (r == 1.0) { cmax++; }
  }
  //if (isOk) { if (cmax <= 0) { console.log(NAME+"WARN: no max found"); } }
  //if (isOk) { if (cmin <= 0) { console.log(NAME+"WARN: no min found"); } }
  if (!isOk) { console.log(NAME+"FAIL", r); }
  return(isOk);
}

function doUnitTestRNG5()
{
  var NAME = STR("RNG5:");
  var isOk = true;
  var TRNG = new Random(getTestRNGCountSeed());
  var r;
  var TMAX = TRNG.RAND_MAX_VALUE / 64.0;
  var cmax = 0;
  for (var i = 0; i < TEST_RNG_MAX_COUNT; i++)
  {
    r = TRNG.randFloat(TMAX);
    if (r < 0) { isOk = false; break; }
    if (r > TMAX) { isOk = false; break; }
    if (r == TMAX) { cmax++; }
  }
  //if (isOk) { if (cmax <= 0) { console.log(NAME+"WARN: no max found"); } }
  if (!isOk) { console.log(NAME+"FAIL", r); }
  return(isOk);
}

function doUnitTestRNG6()
{
  var NAME = STR("RNG6:");
  var isOk = true;
  var TRNG = new Random(getTestRNGCountSeed());
  var r;
  var TMIN = 3333;
  var TMAX = 5555;
  var cmin = 0;
  var cmax = 0;
  for (var i = 0; i < TEST_RNG_MAX_COUNT; i++)
  {
    r = TRNG.randFloat(TMIN, TMAX);
    if (r < TMIN) { isOk = false; break; }
    if (r > TMAX) { isOk = false; break; }
    if (r == TMIN) { cmin++; }
    if (r == TMAX) { cmax++; }
  }
  //if (isOk) { if (cmax <= 0) { console.log(NAME+"WARN: no max found"); } }
  //if (isOk) { if (cmin <= 0) { console.log(NAME+"WARN: no min found"); } }
  if (!isOk) { console.log(NAME+"FAIL", r); }
  return(isOk);
}

// Runner
// ----------------------------------------------------

function runUnitTests()
{
  var TESTS = 
  [ 
    doUnitTest1, 
    doUnitTestRNG0, 
    doUnitTestRNG1, 
    doUnitTestRNG2, 
    doUnitTestRNG3, 
    doUnitTestRNG4, 
    doUnitTestRNG5, 
    doUnitTestRNG6, 
  ];

  var count = TESTS.length;
  var failed = 0;
  for (var i = 0; i < count; i++)
  {
    var test = TESTS[i];
    if (!test())
    {
      failed++;
      console.log("UNIT "+STR(i)+" failed");
    }
  }

  if (failed == 0)
  {
    console.log("UNIT TESTS OK "+STR(count)+"");
  }
  else
  {
    console.log("UNIT TESTS FAILED "+STR(failed)+" of "+STR(count)+"");
  }

  return(failed == 0);
}

