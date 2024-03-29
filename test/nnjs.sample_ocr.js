﻿// JavaScript Simple Neural Network toolkit
// Open Source Software under MIT License
// [OCR demo network]

// Require nnjs.js
// Require nnjs.console.training.stat.js

// Ocr sample
// ----------------------------------------------------

var FIXED_SEED = 1; // 0=use random seed

var SAMPLE_OCR_SX = 9;
var SAMPLE_OCR_SY = 8;

var SAMPLE_OCR_INP_VOID  = 0.0;
var SAMPLE_OCR_INP_FILL  = 1.0;

var SAMPLE_OCR_OUT_NONE  = 0.0;
var SAMPLE_OCR_OUT_FOUND = 1.0;

function emptyStr() { return ""; }

function sampleOcrGetSamples()
{
  // Return array of array of input samples of each letter [ [IA0, IA1, IA2 ...], [IB0, ...], [IC0, ...] ]

  var IA0 = emptyStr()
           +"         "
           +"    *    "
           +"   * *   "
           +"  *   *  "
           +"  *****  "
           +"  *   *  "
           +"  *   *  "
           +"         "
           +"";

  var IA1 = emptyStr()
           +"         "
           +"    *    "
           +"  ** **  "
           +" **   ** "
           +" ******* "
           +" **   ** "
           +" **   ** "
           +"         "
           +"";

  var IA2 = emptyStr()
           +"         "
           +"   ***   "
           +"  *   *  "
           +"  *   *  "
           +"  *****  "
           +"  *   *  "
           +" *** *** "
           +"         "
           +"";

  var IB0 = emptyStr()
           +"         "
           +"  ****   "
           +"  *   *  "
           +"  ****   "
           +"  *   *  "
           +"  *   *  "
           +"  ****   "
           +"         "
           +"";

  var IB1 = emptyStr()
           +"         "
           +" *****   "
           +" **  **  "
           +" *****   "
           +" **   ** "
           +" **   ** "
           +" *****   "
           +"         "
           +"";

  var IC0 = ""
           +"         "
           +"   ***   "
           +"  *   *  "
           +"  *      "
           +"  *      "
           +"  *   *  "
           +"   ***   "
           +"         "
           +"";

  var IC1 = ""
           +"         "
           +"   ***   "
           +" **   ** "
           +" **      "
           +" **      "
           +" **   ** "
           +"   ***   "
           +"         "
           +"";

  var ID0 = ""
           +"         "
           +"  ****   "
           +"  *   *  "
           +"  *   *  "
           +"  *   *  "
           +"  *   *  "
           +"  ****   "
           +"         "
           +"";

  var ID1 = ""
           +"         "
           +" *****   "
           +" **   ** "
           +" **   ** "
           +" **   ** "
           +" **   ** "
           +" *****   "
           +"         "
           +"";

  var ID2 = ""
           +"         "
           +" *****   "
           +"  *   *  "
           +"  *   *  "
           +"  *   *  "
           +"  *   *  "
           +" *****   "
           +"         "
           +"";

  function getLNArray(L)
  {
    // Convert letter from fancy text to plain array of 1 and 0
    var R = [];
    for (var y = 0; y < SAMPLE_OCR_SY; y++)
    {
      for (var x = 0; x < SAMPLE_OCR_SX; x++)
      {
        R.push(L[y*SAMPLE_OCR_SX+x] == " " ? SAMPLE_OCR_INP_VOID : SAMPLE_OCR_INP_FILL);
      }
    }

    return(R);
  }

  // letters to recognize, each SX * SY size in many samples
  var I0 = 
  [
    [ getLNArray(IA0), getLNArray(IA1), getLNArray(IA2), ], 
    [ getLNArray(IB0), getLNArray(IB1) ], 
    [ getLNArray(IC0), getLNArray(IC1) ], 
    [ getLNArray(ID0), getLNArray(ID1), getLNArray(ID2) ]
  ];

  return(I0);
}

// Train input preparation

var NOISE_TYPE_PIXEL_FLIP = 0;
var NOISE_TYPE_PIXEL_DARKER_LIGHTER = 1;
var NOISE_TYPE_PIXEL_RANDOM = 2;

function getNoisedInput(L, noiseCount, noiseType)
{
  // type: 0=flip pixel, 1=drarker/lighter
  if (noiseType == null) { noiseType = NOISE_TYPE_PIXEL_FLIP; }

  function makeNoise(value)
  {
    if (noiseType == NOISE_TYPE_PIXEL_DARKER_LIGHTER)
    {
      var HALF = (SAMPLE_OCR_INP_VOID + SAMPLE_OCR_INP_FILL) / 2.0;
      if (value <  HALF) { return(NN.Internal.getRandom(Math.min(SAMPLE_OCR_INP_VOID, SAMPLE_OCR_INP_FILL), HALF - 0.1)); }
      if (value >= HALF) { return(NN.Internal.getRandom(HALF + 0.1, Math.max(SAMPLE_OCR_INP_VOID, SAMPLE_OCR_INP_FILL))); }
      return(value);
    }

    if (noiseType == NOISE_TYPE_PIXEL_RANDOM)
    {
      return(NN.Internal.getRandom(Math.min(SAMPLE_OCR_INP_VOID, SAMPLE_OCR_INP_FILL), Math.max(SAMPLE_OCR_INP_VOID, SAMPLE_OCR_INP_FILL)));
    }

    return(1-value); // flip pixel
  }

  if (noiseCount == null) { noiseCount = 0; }

  var R = L.slice(); // copy

  for (var i = 0; i < noiseCount; i++)
  {
    var noiseIndex = NN.Internal.getRandomInt(R.length);
    R[noiseIndex] = makeNoise(R[noiseIndex]);
  }

  return(R);
}

function getShiftedImg(L, sx, sy)
{
  if (sx == null) { sx = 0; }
  if (sy == null) { sy = 0; }

  var R = [];
  for (var y = 0; y < SAMPLE_OCR_SY; y++)
  {
    for (var x = 0; x < SAMPLE_OCR_SX; x++)
    {
      var ox = (x + -sx); ox = (ox < 0) ? SAMPLE_OCR_SX+ox : ox; ox %= SAMPLE_OCR_SX;
      var oy = (y + -sy); oy = (oy < 0) ? SAMPLE_OCR_SY+oy : oy; oy %= SAMPLE_OCR_SY;
      R.push(L[oy*SAMPLE_OCR_SX+ox]);
    }
  }

  return(R);
}

function sampleAddLetTexts(L,inT,addTopSep,addLeftSep,addBottomSep,addRightSep)
{
  var USE_ASCII = false; // #define

  if (addTopSep == null)    { addTopSep = true; }
  if (addLeftSep == null)   { addLeftSep = true; }
  if (addBottomSep == null) { addBottomSep = false; }
  if (addRightSep == null)  { addRightSep = false; }

  function inText(i) { if (inT == null) { return(""); } else { return(inT[i]); } }

  // T will be SAMPLE_OCR_SY+1+1 height

  var T = [];

  // !00000000000! top Sep
  // !<text Y[0]>
  // !<text Y[SAMPLE_OCR_SY-1]>
  // !00000000000! bottom Sep

  var ty;
  
  ty = 0;
  if (addTopSep) { T.push(inText(ty++)); }
  for (var y = 0; y < SAMPLE_OCR_SY; y++)
  {
    T.push(inText(ty++));
  }
  if (addBottomSep) { T.push(inText(ty++)); }

  var t = ""; 
  if (addLeftSep) { t += "!"; }
  for (var x = 0; x < SAMPLE_OCR_SX; x++)
  {
    t += "-";
  }
  if (addRightSep) { t += "!"; }

  var SEP = t; // sep line

  ty = 0;
  if (addTopSep) { T[ty++] += SEP; }
  for (var y = 0; y < SAMPLE_OCR_SY; y++)
  {
    var t = ""; 
    if (addLeftSep) { t += "!"; }
    for (var x = 0; x < SAMPLE_OCR_SX; x++)
    {
      var v = L[y*SAMPLE_OCR_SX+x];
      var c = "";

      if (v <= 0)
      {
        c = " ";
      }
      else if (v >= 1)
      {
        if (USE_ASCII) {
          c = "*"; // "*";
        } else {
          c = "\u2588"; // "█";
        }
      }
      else
      {
        // v = Math.floor(v * 10); c = v.toString()[0];
        v = Math.floor(v * 10);
        var F;
        if (USE_ASCII) {
          F=["0",      "1",      "2",      "3",      "4",      "5",      "6",      "7",      "8",      "9"     ]; // "0123456789";
        } else {
          F=["\u2591", "\u2591", "\u2591", "\u2592", "\u2592", "\u2592", "\u2593", "\u2593", "\u2593", "\u2593"]; // "░░░▒▒▒▒▓▓▓";
        }
        c = F[v];
      }

      t += c;
    }
    if (addRightSep) { t += "!"; }
    T[ty++] += t;
  }
  if (addBottomSep) { T[ty++] += SEP; }

  //USE_ASCII = null; // #undef
  return(T);
}

function sampleOcrNetwork()
{
  var testResult = true;

  // The Dataset

  if (true)
  {
    var seed = FIXED_SEED > 0 ? FIXED_SEED : Random.getRandomSeed(new Date().getTime());
    NN.Internal.getPRNG().setSeed(seed);
    console.log("sampleOcrNetwork", "(samples)", "seed=", seed);
  }

  var SAMPLES = sampleOcrGetSamples(); // [letter][sample] = data[]
  var SAMPLES_COUNT = SAMPLES.length;
  var RESULTS = []; // [letter] = R1 array
  for (var dataIndex = 0; dataIndex < SAMPLES_COUNT; dataIndex++)
  {
    var RESULT = NN.NetworkStat.getR1Array(dataIndex, SAMPLES_COUNT, SAMPLE_OCR_OUT_FOUND, SAMPLE_OCR_OUT_NONE); // target result for this letter
    RESULTS.push(RESULT);
  }

  // The Net

  var LAYERS = 3;
  var NET = new NN.Network();

  if (true)
  {
    var seed = FIXED_SEED > 0 ? FIXED_SEED : Random.getRandomSeed(new Date().getTime());
    NN.Internal.getPRNG().setSeed(seed);
    console.log("sampleOcrNetwork", "(net)", "seed=", seed, "layers=", LAYERS);
  }

  //Not working     : seed=1 no train // TODO: Check why RELU does not work here (error is 0.5 all they way during training)
  //var actFunc = NN.ActFuncRELUTrainee.getInstance();
  //var outFunc = NN.ActFuncRELUTrainee.getInstance();

  //Works very good : seed=1 train in  41 iterations // Stat: 100.0%/100.0%/100.0%/ 97.1% // InfTime:71us(65us?) // INP/OUT 0.0..1.0 // Leak:0.1
  //Works good      : seed=1 train in  84 iterations // Stat: 100.0%/100.0%/100.0%/ 95.7% // InfTime:71us(65us?) // INP/OUT 0.0..1.0 // Leak:0.01
  //Works EXCELLENT : seed=1 train in  85 iterations // Stat: 100.0%/100.0%/100.0%/100.0% // InfTime:71us(65us?) // INP/OUT 0.0..1.0 // Leak:0.001 * [def]
  //Works good      : seed=1 train in 356 iterations // Stat: 100.0%/100.0%/100.0%/ 98.6% // InfTime:71us(65us?) // INP/OUT 0.0..1.0 // Leak:0.0001
  var actFunc = NN.ActFuncLRELUTrainee.newInstance(0.001);
  var outFunc = NN.ActFuncLRELUTrainee.newInstance(0.001);

  //Works good      : seed=1 train in  71 iterations // Stat: 100.0%/100.0%/100.0%/ 94.4% // InfTime:71us(65us?) // INP/OUT 0.0..1.0 // Leak:0.001 * [def]
  //var actFunc = NN.ActFuncLLRELUTrainee.newInstance(0.001);
  //var outFunc = NN.ActFuncLLRELUTrainee.newInstance(0.001);

  //Works good      : seed=1 train in 255 iterations // Stat: 100.0%/ 98.6%/100.0%/ 88.6% // InfTime:71us
  //var actFunc = NN.ActFuncSigmoidTrainee.getInstance();
  //var outFunc = NN.ActFuncSigmoidTrainee.getInstance();

  //Works norm      : seed=1 train in 442 iterations // Stat: 100.0%/ 98.6%/ 92.9%/ 84.3% // InfTime:71us
  //var actFunc = NN.ActFuncTanhTrainee.getInstance();
  //var outFunc = NN.ActFuncTanhTrainee.getInstance();

  if (LAYERS == 3)
  {
    var IN  = new NN.Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY, NN.TheNeuronFactory(NN.InputNeuron)); IN.addNeurons(1, NN.TheNeuronFactory(NN.BiasNeuron));
    var L1  = new NN.Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY*1, NN.ExtNeuronFactory(NN.ProcNeuronTrainee, actFunc)); L1.addNeurons(1, NN.TheNeuronFactory(NN.BiasNeuron)); L1.addInputAll(IN);
    var OUT = new NN.Layer(SAMPLES_COUNT, NN.ExtNeuronFactory(NN.ProcNeuronTrainee, outFunc)); OUT.addInputAll(L1); // Outputs: 0=A, 1=B, 2=C, ...
    NET.addLayer(IN); NET.addLayer(L1); NET.addLayer(OUT);
  }
  else
  {
    var IN  = new NN.Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY, NN.TheNeuronFactory(NN.InputNeuron)); IN.addNeurons(1, NN.TheNeuronFactory(NN.BiasNeuron));
    var L1  = new NN.Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY*1, NN.ExtNeuronFactory(NN.ProcNeuronTrainee, actFunc)); L1.addNeurons(1, NN.TheNeuronFactory(NN.BiasNeuron)); L1.addInputAll(IN);
    var L2  = new NN.Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY, NN.ExtNeuronFactory(NN.ProcNeuronTrainee, actFunc)); L2.addNeurons(1, NN.TheNeuronFactory(NN.BiasNeuron)); L2.addInputAll(L1);
    var OUT = new NN.Layer(SAMPLES_COUNT, NN.ExtNeuronFactory(NN.ProcNeuronTrainee, outFunc)); OUT.addInputAll(L2); // Outputs: 0=A, 1=B, 2=C, ...
    NET.addLayer(IN); NET.addLayer(L1); NET.addLayer(L2); NET.addLayer(OUT);
  }

  console.log("Network created: Layers=" + STR(NET.layers.length) + " Neurons=" + STR(NN.NetworkStat.getNetNeuronsCount(NET)) + " Weights=" + STR(NN.NetworkStat.getNetWeightsCount(NET)));

  // Dataset prepration

  // SAMPLES // 2D array [letter][sample] = data[]
  // RESULTS // 1D array [letter] = result[] expected

  // Prepare DATAS and TARGS as source and expected results to train

  //var DATASE = [ getLArray(LA0), getLArray(LB0), getLArray(LC0), getLArray(LD0) ];
  //var TARGSE = [ getR1Out(0,4),  getR1Out(1,4),  getR1Out(2,4),  getR1Out(3,4)  ];

  var DATASE = [ ]; // data source etalon (no noise) : source samples as plain array (inputs)
  var TARGSE = [ ]; // data target etalon (no noise) : results expected as plain array (outputs)

  for (var dataIndex = 0; dataIndex < SAMPLES.length; dataIndex++)
  {
    for (var ii = 0; ii < SAMPLES[dataIndex].length; ii++)
    {
      DATASE.push(SAMPLES[dataIndex][ii]);
      TARGSE.push(RESULTS[dataIndex]); // for all samples of same input result should be the same
    }
  }

  // Augmented data (original + shifted)

  var DATAS  = []; // work samples (may be noised)
  var TARGS  = []; // work targets (for in noised)

  for (var dataIndex = 0; dataIndex < DATASE.length; dataIndex++)
  {
    DATAS.push(DATASE[dataIndex]);
    TARGS.push(TARGSE[dataIndex]);
    DATAS.push(getShiftedImg(DATASE[dataIndex],0,1));
    DATAS.push(getShiftedImg(DATASE[dataIndex],1,0));
    DATAS.push(getShiftedImg(DATASE[dataIndex],1,1));
    TARGS.push(TARGSE[dataIndex]);
    TARGS.push(TARGSE[dataIndex]);
    TARGS.push(TARGSE[dataIndex]);
    DATAS.push(getShiftedImg(DATASE[dataIndex],0,-1));
    DATAS.push(getShiftedImg(DATASE[dataIndex],-1,0));
    DATAS.push(getShiftedImg(DATASE[dataIndex],-1,-1));
    TARGS.push(TARGSE[dataIndex]);
    TARGS.push(TARGSE[dataIndex]);
    TARGS.push(TARGSE[dataIndex]);
  }

  //var DATA_AUGMENTATION_MUTIPLIER = DATAS.length / DATASE.length; // there will be 7 DATAS images per 1 source DATASE image

  // Dump dataset before train

  var DUMP_DATASET = false;

  function dumpSamples(DATAS, imagesPerSample)
  {
    var sampleCount = DATAS.length / imagesPerSample; // sumber of samples

    for (var sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
      var T = null;
      for (var imageIndex = 0; imageIndex < imagesPerSample; imageIndex++)
      {
        T = sampleAddLetTexts(DATAS[sampleIndex*imagesPerSample+imageIndex], T);
      }

      for (var lineIndex = 0; lineIndex < T.length; lineIndex++)
      {
        console.log(T[lineIndex], lineIndex, sampleIndex);
      }
    }
  }

  if (DUMP_DATASET) { dumpSamples(DATAS, DATAS.length / DATASE.length); }

  // Training

  console.log("Training, please wait ...");
  if (!NN.doTrain(NET, DATAS, TARGS, new NN.TrainingParams(0.125, 1000), new NN.TrainingProgressReporterConsole(10), new NN.TrainingDoneCheckerEps()))
  {
    console.log("Training failed (does not to achieve loss error margin?)", NET);
    testResult = false;
  }

  console.log("Training finsihed", NET);

  // Verification

  function verifyProc(NET, DATAS, TARGS, stepName, imagesPerSample, maxFailRate = 0.0)
  {
    if (maxFailRate == null) { maxFailRate = 0.0; }

    var DUMP_FAILED_IMAGES = false;

    var startTimeMetter = new NN.TimeMetter();
    startTimeMetter.start();

    var CHKRS = [];
    for (var dataIndex = 0; dataIndex < DATAS.length; dataIndex++)
    {
      CHKRS.push(NN.doProc(NET, DATAS[dataIndex]));
    }

    startTimeMetter.stop();

    var stepTime = Math.round((1000.0 * startTimeMetter.millisPassed() / DATAS.length)); // microseconds (us)

    var vdif = 0.15; // max diff for smart verification
    var veps = 0.4; // epsilon for strict verification

    var statGood = 0;
    var statFail = 0;
    var statWarn = 0;

    var isOK = true;
    for (var dataIndex = 0; dataIndex < DATAS.length; dataIndex++)
    {
      var imageIndex = dataIndex % imagesPerSample;
      var sampleIndex = (dataIndex-imageIndex) / imagesPerSample;

      var isSimpleMatchOK = NN.NetworkStat.isResultSampleMatchEps(TARGS[dataIndex], CHKRS[dataIndex], veps);

      var status = "";

      if (isSimpleMatchOK)
      {
        //status = "OK.OK.OK.OK.OK.OK.OK.OK"; // uncomment to dump all
        statGood++;
      }
      else
      {
        var smartMatchSampleIndex = NN.NetworkStat.getMaximumIndexEps(CHKRS[dataIndex], vdif);
        if (smartMatchSampleIndex < 0)
        {
          status = "FAIL*";
          statFail++;
          isOK = false;
        }
        else
        {
          var smartMatchExpectIndex = NN.NetworkStat.getMaximumIndex(TARGS[dataIndex], vdif);
          if (smartMatchSampleIndex != smartMatchExpectIndex)
          {
            status = "FAIL";
            statFail++;
            isOK = false;
          }
          else // match, but not simple match
          {
            status = "WARN";
            statWarn++;
          }
        }
      }

      if ((status != null) && (status != ""))
      {
        console.log("Verification step " + STR(stepName) + "[" + STR(dataIndex) + "]" + ":" + STR(status) + "", [DATAS[dataIndex], TARGS[dataIndex], CHKRS[dataIndex]], smartMatchSampleIndex, [veps, vdif]);
        if (DUMP_FAILED_IMAGES)
        {
          var T = sampleAddLetTexts(DATAS[dataIndex], null, true, true, true, true);
          for (var lineIndex = 0; lineIndex < T.length; lineIndex++)
          {
            console.log(T[lineIndex], lineIndex, sampleIndex, imageIndex);
          }
        }
      }
    }

    function showPerc(val) { return STR("") + STR(Math.round(val * 1000.0) / 10.0) + "%"; }

    var statFull = statGood + statFail + statWarn;
    var rateNorm = statFull > 0 ? (1.0 * statGood + statWarn) / statFull : 0.0;

    var statText = "";

    if (isOK)
    {
      statText = "Good";
    }
    else
    {
      statText = "Fail";
      if (maxFailRate > 0)
      {
        if ((1.0 * statFail / statFull) <= maxFailRate)
        {
          isOK = true; // failed, but withing the allowed range - assume ok
          statText = "Warn";
        }
      }
    }

    var msg;

    msg  = "";
    msg += "Verification step " + STR(stepName) + STR(" ");
    msg += "Status:" + STR(statText) + STR(" ");
    msg += "Rate:" + showPerc(rateNorm) + STR(" ");
    msg += "InfTime:" + STR(stepTime) + "us" + " ";

    if (statWarn > 0)
    {
      msg += "Warn:" + showPerc(1.0 * statWarn / statFull) + STR(" ");
    }

    if (statFail > 0)
    {
      msg += "Fail:" + showPerc(1.0 * statFail / statFull) + STR(" ");
    }

    console.log(msg);

    return(isOK);
  }

  // Verify on source dataset (dry run)

  testResult = verifyProc(NET, DATAS, TARGS, "Source", DATAS.length / DATASE.length) && testResult;

  // Create noised data

  var DATASN; // would be the same size, as DATAS, so imagesPerSample will be the same

  DATASN = [];
  for (var dataIndex = 0; dataIndex < DATAS.length; dataIndex++)
  {
    DATASN.push(getNoisedInput(DATAS[dataIndex],1));
  }
  testResult = verifyProc(NET, DATASN, TARGS, "Noised.F1", DATASN.length / DATASE.length, 0.1) && testResult;

  DATASN = [];
  for (var dataIndex = 0; dataIndex < DATAS.length; dataIndex++)
  {
    DATASN.push(getNoisedInput(DATAS[dataIndex],30,NOISE_TYPE_PIXEL_DARKER_LIGHTER));
  }
  testResult = verifyProc(NET, DATASN, TARGS, "Noised.DL30", DATASN.length / DATASE.length, 0.1) && testResult;

  DATASN = [];
  for (var dataIndex = 0; dataIndex < DATAS.length; dataIndex++)
  {
    DATASN.push(getNoisedInput(DATAS[dataIndex],10,NOISE_TYPE_PIXEL_RANDOM));
  }
  testResult = verifyProc(NET, DATASN, TARGS, "Noised.R10", DATASN.length / DATASE.length, 0.2) && testResult;
  
  return(testResult);
}

//sampleOcrNetwork();
