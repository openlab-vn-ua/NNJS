﻿// JavaScript Simple Neural Network toolkit
// Open Source Software under MIT License
// [OCR demo network]

// Require nnjs.js
// Require nnjs.console.training.stat.js

// Ocr sample
// ----------------------------------------------------

var SAMPLE_OCR_SX = 9;
var SAMPLE_OCR_SY = 8;

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
        R.push(L[y*SAMPLE_OCR_SX+x] == "*" ? 1 : 0);
      }
    }

    return(R);
  }

  // letters to recognize, each SX * SY size in many samples
  var I0 = [ 
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
      if (value <= 0) { return(NN.Internal.getRandom(0.0 , 0.49)); }
      if (value >= 1) { return(NN.Internal.getRandom(0.51, 1.0 )); }
      return(value);
    }

    if (noiseType == NOISE_TYPE_PIXEL_RANDOM)
    {
      return(NN.Internal.getRandom(0.0,1.0));
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
  if (true)
  {
    var seed = new Date().getTime() % 0x7FFF0000 + 1;
    NN.Internal.PRNG.setSeed(seed);
    console.log("sampleOcrNetwork", "(samples)", "seed=", seed);
  }

  var SAMPLES = sampleOcrGetSamples(); // [letter][sample] = data[]

  // The Net

  var LAYERS = 3;
  var NET;

  if (true)
  {
    var seed = new Date().getTime() % 0x7FFF0000 + 1;
    NN.Internal.PRNG.setSeed(seed);
    console.log("sampleOcrNetwork", "(net)", "seed=", seed, "layers=", LAYERS);
  }

  if (LAYERS == 3)
  {
    var IN  = new NN.Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY, NN.InputNeuron); IN.addNeuron(NN.BiasNeuron);
    var L1  = new NN.Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY*1, NN.ProcNeuron); L1.addNeuron(NN.BiasNeuron); L1.addInputAll(IN);
    var OUT = new NN.Layer(SAMPLES.length, NN.ProcNeuron); OUT.addInputAll(L1); // Outputs: 0=A, 1=B, 2=C, ...
    NET = [IN, L1, OUT];
  }
  else
  {
    var IN  = new NN.Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY, NN.InputNeuron); IN.addNeuron(NN.BiasNeuron);
    var L1  = new NN.Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY*1, NN.ProcNeuron); L1.addNeuron(NN.BiasNeuron); L1.addInputAll(IN);
    var L2  = new NN.Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY, NN.ProcNeuron); L2.addNeuron(NN.BiasNeuron); L2.addInputAll(L1);
    var OUT = new NN.Layer(SAMPLES.length, NN.ProcNeuron); OUT.addInputAll(L2); // Outputs: 0=A, 1=B, 2=C, ...
    NET = [IN, L1, L2, OUT];
  }

  function getR1Array(index, total, SET, NOTSET)
  {
    if (SET    == null) { SET    = 1; }
    if (NOTSET == null) { NOTSET = 0; }

    // Retuns array with only one index of total item set to SET(=1) and all other as NOTSET(=0): 0=[1, 0, 0 ...], 1=[0, 1, 0, ...], 2=[0, 0, 1, ...]
    var R = [];

    for (var i = 0; i < total; i++)
    {
      R.push(i == index ? SET : NOTSET);
    }

    return(R);
  }

  // Prepare DATAS and TARGS as source and expected results to train

  var INPSS = SAMPLES; // 2D array [letter][sample] = data[]
  var OUTRS = [ ]; // 1D array [letter] = result[] expected

  for (var dataIndex = 0; dataIndex < INPSS.length; dataIndex++)
  {
    var OUTR = getR1Array(dataIndex,INPSS.length); // target result for this letter
    OUTRS.push(OUTR);
  }

  //var DATASE = [ getLArray(LA0), getLArray(LB0), getLArray(LC0), getLArray(LD0) ];
  //var TARGSE = [ getR1Out(0,4),  getR1Out(1,4),  getR1Out(2,4),  getR1Out(3,4) ];

  var DATASE = [ ]; // data source etalon (no noise) : source samples as plain array (inputs)
  var TARGSE = [ ]; // data target etalon (no noise) : results expected as plain array (outputs)

  for (var dataIndex = 0; dataIndex < INPSS.length; dataIndex++)
  {
    var INPS = INPSS[dataIndex]; // letter samples (may be many)
    var OUTR = OUTRS[dataIndex]; // target result for this letter

    for (var ii = 0; ii < INPS.length; ii++)
    {
      DATASE.push(INPS[ii]);
      TARGSE.push(OUTR); // for all samples of same input result should be the same
    }
  }

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

  dumpSamples(DATAS, DATAS.length / DATASE.length);

  console.log("Training, please wait ...");
  if (!NN.doTrain(NET, DATAS, TARGS, -1, -1, new NN.TrainingProgressReporterConsole(10)))
  {
    console.log("Training failed!", NET);
    return(false);
  }

  console.log("Training complete", NET);

  // Verification

  function getMaximumIndex(R, minDiff)
  {
    // input:  R as vector of floats (usualy 0.0 .. 1.0)
    // result: index of maximum value, checking that next maximum is at least eps lower.
    // returns -1 if no such value found (maximums too close)

    var FAIL = -1;

    if (R.length <= 0) { return(FAIL); }

    var currMaxIndex = 0;
    for (var i = 1; i < R.length; i++)
    {
      if (R[i] > R[currMaxIndex]) { currMaxIndex = i; }
    }

    if (R[currMaxIndex] < minDiff)
    {
      return(FAIL); // not ever greater than 0, no reason so check another max
    }

    if (R.length <= 1) { return(currMaxIndex); } // actually, 0

    var nextMaxIndex = (currMaxIndex + 1) % R.length; // actually, any other value

    for (var i = 0; i < R.length; i++)
    {
      if (i == currMaxIndex)
      {
        // skip, this is current max
      }
      else if (i == nextMaxIndex)
      {
        // skip, this is current next max
      }
      else
      {
        if (R[i] > R[nextMaxIndex]) { nextMaxIndex = i; }
      }
    }

    var nextMaxValue = R[nextMaxIndex];

    if (nextMaxValue < 0) { nextMaxValue = 0; } // bug trap

    if ((R[currMaxIndex] - nextMaxValue) >= minDiff)
    {
      return(currMaxIndex);
    }

    return(FAIL);
  }

  function verifyProc(NET, DATAS, TARGS, stepName, imagesPerSample)
  {
    var CHKRS = [];
    for (var dataIndex = 0; dataIndex < DATAS.length; dataIndex++)
    {
      CHKRS.push(NN.doProc(NET, DATAS[dataIndex]));
    }

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

      var isSimpleMatchOK = NN.isResultSampleMatchSimpleFunc(TARGS[dataIndex], CHKRS[dataIndex], veps);
      var smartMatchSampleIndex = getMaximumIndex(CHKRS[dataIndex], vdif);
      var smartMatchExpectIndex = getMaximumIndex(TARGS[dataIndex], vdif);

      //if (true) // all
      //if (!isSimpleMatchOK) // warn
      if ((smartMatchSampleIndex < 0) || (smartMatchSampleIndex != smartMatchExpectIndex)) // fail
      {
        var status = "";

        if ((smartMatchSampleIndex < 0) || (smartMatchSampleIndex != smartMatchExpectIndex))
        {
          status = "FAIL";
          statFail++;
        }
        else if (!isSimpleMatchOK)
        {
          status = "WARN";
          statWarn++;
        }
        else
        {
          status = "OK.OK.OK.OK.OK.OK.OK.OK";
          statGood++;
        }

        console.log("Verification step " + STR(stepName) + "[" + STR(dataIndex) + "]" + ":" + STR(status) + "", [DATAS[dataIndex], TARGS[dataIndex], CHKRS[dataIndex]], smartMatchSampleIndex, [ veps, vdif ]);
        var T = sampleAddLetTexts(DATAS[dataIndex], null, true, true, true, true);
        for (var lineIndex = 0; lineIndex < T.length; lineIndex++)
        {
          console.log(T[lineIndex], lineIndex, sampleIndex, imageIndex);
        }
        isOK = false;
      }
      else
      {
        statGood++;
      }
    }

    if (isOK)
    {
      console.log("Verification step " + STR(stepName) + ":OK [100%]");
    }
    else
    {
      var statFull = 0.0 + statGood + statFail + statWarn;
      function showPerc(val) { return STR("") + STR(Math.round(val * 1000.0) / 10.0); }
      console.log("Verification step " + STR(stepName) + ":Done:" + (" GOOD=" + showPerc(statGood / statFull)) + (" WARN=" + showPerc(statWarn / statFull)) + (" FAIL=" + showPerc(statFail / statFull)));
    }

    return(isOK);
  }

  verifyProc(NET, DATAS, TARGS, "Source", DATAS.length / DATASE.length);

  // Create noised data

  var DATASN;

  DATASN = [];
  for (var dataIndex = 0; dataIndex < DATAS.length; dataIndex++)
  {
    DATASN.push(getNoisedInput(DATAS[dataIndex],1));
  }

  verifyProc(NET, DATASN, TARGS, "Noised.F1", DATASN.length / DATASE.length);

  DATASN = [];
  for (var dataIndex = 0; dataIndex < DATAS.length; dataIndex++)
  {
    DATASN.push(getNoisedInput(DATAS[dataIndex],30,NOISE_TYPE_PIXEL_DARKER_LIGHTER));
  }

  verifyProc(NET, DATASN, TARGS, "Noised.DL30", DATASN.length / DATASE.length);

  DATASN = [];
  for (var dataIndex = 0; dataIndex < DATAS.length; dataIndex++)
  {
    DATASN.push(getNoisedInput(DATAS[dataIndex],10,NOISE_TYPE_PIXEL_RANDOM));
  }

  verifyProc(NET, DATASN, TARGS, "Noised.R10", DATASN.length / DATASE.length);
  return(true);
}

//sampleOcrNetwork();
