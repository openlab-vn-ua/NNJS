// JavaScript Simple Neural Network toolkit
// [OCR demo network]

// Require nnjs.js

var SAMPLE_OCR_SX = 9;
var SAMPLE_OCR_SY = 8;

function sampleOcrGetSamples()
{
  // Return array of array of input samples of each letter [ [IA0, IA1, IA2 ...], [IB0, ...], [IC0, ...] ]

  var IA0 = ''
           +'         '
           +'    *    '
           +'   * *   '
           +'  *   *  '
           +'  *****  '
           +'  *   *  '
           +'  *   *  '
           +'         '
           +'';

  var IA1 = ''
           +'         '
           +'    *    '
           +'  ** **  '
           +' **   ** '
           +' ******* '
           +' **   ** '
           +' **   ** '
           +'         '
           +'';

  var IA2 = ''
           +'         '
           +'   ***   '
           +'  *   *  '
           +'  *   *  '
           +'  *****  '
           +'  *   *  '
           +' *** *** '
           +'         '
           +'';

  var IB0 = ''
           +'         '
           +'  ****   '
           +'  *   *  '
           +'  ****   '
           +'  *   *  '
           +'  *   *  '
           +'  ****   '
           +'         '
           +'';

  var IB1 = ''
           +'         '
           +' *****   '
           +' **  **  '
           +' *****   '
           +' **   ** '
           +' **   ** '
           +' *****   '
           +'         '
           +'';

  var IC0 = ''
           +'         '
           +'   ***   '
           +'  *   *  '
           +'  *      '
           +'  *      '
           +'  *   *  '
           +'   ***   '
           +'         '
           +'';

  var IC1 = ''
           +'         '
           +'   ***   '
           +' **   ** '
           +' **      '
           +' **      '
           +' **   ** '
           +'   ***   '
           +'         '
           +'';

  var ID0 = ''
           +'         '
           +'  ****   '
           +'  *   *  '
           +'  *   *  '
           +'  *   *  '
           +'  *   *  '
           +'  ****   '
           +'         '
           +'';

  var ID1 = ''
           +'         '
           +' *****   '
           +' **   ** '
           +' **   ** '
           +' **   ** '
           +' **   ** '
           +' *****   '
           +'         '
           +'';

  var ID2 = ''
           +'         '
           +' *****   '
           +'  *   *  '
           +'  *   *  '
           +'  *   *  '
           +'  *   *  '
           +' *****   '
           +'         '
           +'';

  function getLNArray(L)
  {
    // Convert letter from fancy text to plain array of 1 and 0
    var R = [];
    for (var y = 0; y < SAMPLE_OCR_SY; y++)
    {
      for (var x = 0; x < SAMPLE_OCR_SX; x++)
      {
        R.push(L[y*SAMPLE_OCR_SX+x] == '*' ? 1 : 0);
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

function sampleAddLetTexts(L,inT,addTopSep,addLeftSep,addBottomSep,addRightSep)
{
  if (addTopSep == null)    { addTopSep = true; }
  if (addLeftSep == null)   { addLeftSep = true; }
  if (addBottomSep == null) { addBottomSep = false; }
  if (addRightSep == null)  { addRightSep = false; }

  function inText(i) { if (inT == null) { return(''); } else { return(inT[i]); } }

  // T will be SAMPLE_OCR_SY+1+1 height

  var T = [];

  // !00000000000! top Sep
  // !<text Y[0]>
  // !<text Y[SAMPLE_OCR_SY-1]>
  // !00000000000! bottom Sep

  var ty = 0;
  if (addTopSep) { T.push(inText(ty++)); }
  for (var y = 0; y < SAMPLE_OCR_SY; y++)
  {
    T.push(inText(ty++));
  }
  if (addBottomSep) { T.push(inText(ty++)); }

  var t = ''; 
  if (addLeftSep) { t += '!'; }
  for (var x = 0; x < SAMPLE_OCR_SX; x++)
  {
    t += '-';
  }
  if (addRightSep) { t += '!'; }

  var SEP = t; // sep line

  var ty = 0;
  if (addTopSep) { T[ty++] += SEP; }
  for (var y = 0; y < SAMPLE_OCR_SY; y++)
  {
    var t = ''; 
    if (addLeftSep) { t += '!'; }
    for (var x = 0; x < SAMPLE_OCR_SX; x++)
    {
      t += '' + ((L[y*SAMPLE_OCR_SX+x] == 0) ? ' ' : '1');
    }
    if (addRightSep) { t += '!'; }
    T[ty++] += t;
  }
  if (addBottomSep) { T[ty++] += SEP; }

  return(T);
}

function sampleOcrNetwork()
{
  var SAMPLES = sampleOcrGetSamples();

  // The Net

  var LAYERS = 3;
  var NET;

  if (LAYERS == 3)
  {
    var IN  = new NN.Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY, NN.InputNeuron); IN.addNeuron(NN.BiasNeuron);
    var L1  = new NN.Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY, NN.Neuron); L1.addNeuron(NN.BiasNeuron); L1.addInputAll(IN);
    var OUT = new NN.Layer(SAMPLES.length, NN.Neuron); OUT.addInputAll(L1); // Outputs: 0=A, 1=B, 2=C, ...
    NET = [IN, L1, OUT];
  }
  else
  {
    var IN  = new NN.Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY, NN.InputNeuron); IN.addNeuron(NN.BiasNeuron);
    var L1  = new NN.Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY, NN.Neuron); L1.addNeuron(NN.BiasNeuron); L1.addInputAll(IN);
    var L2  = new NN.Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY, NN.Neuron); L2.addNeuron(NN.BiasNeuron); L2.addInputAll(L1);
    var OUT = new NN.Layer(SAMPLES.length, NN.Neuron); OUT.addInputAll(L2); // Outputs: 0=A, 1=B, 2=C, ...
    NET = [IN, L1, L2, OUT];
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
        return(value == 0 ? NN.Internal.getRandom(0.0,0.49) : NN.Internal.getRandom(0.51,1.0));
      }

      if (noiseType == NOISE_TYPE_PIXEL_RANDOM)
      {
        return(NN.Internal.getRandom(0.0,1.0));
      }

      return(1-value); // flip pixel
    }

    if (noiseCount == null) { noiseCount = 0; }

    var R = L.slice();

    for (var i = 0; i < noiseCount; i++)
    {
      var noiseIndex = NN.Internal.getRandomInt(0,R.length);
      R[noiseIndex] = 1-R[noiseIndex];
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

  var INPSS = SAMPLES; // 2D array [letter][sample] = data
  var OUTRS = [ ]; // 1D array [letter] = result expected

  for (var dataIndex = 0; dataIndex < INPSS.length; dataIndex++)
  {
    var OUTR = getR1Array(dataIndex,INPSS.length); // target result for this letter
    OUTRS.push(OUTR);
  }

  //var DATASE = [ getLArray(LA0), getLArray(LB0), getLArray(LC0), getLArray(LD0)];
  //var TARGSE = [ getR1Out(0,4),  getR1Out(1,4),  getR1Out(2,4),  getR1Out(3,4)];

  var DATASE = [ ]; // data source etalon (no noise) : source samples as plain array
  var TARGSE = [ ]; // data source etalon (no noise) : results expected as plain array

  for (var dataIndex = 0; dataIndex < INPSS.length; dataIndex++)
  {
    var INPS = INPSS[dataIndex]; // letter samples (may be many)
    var OUTR = OUTRS[dataIndex]; // target result for this letter

    for (var ii = 0; ii < INPS.length; ii++)
    {
      DATASE.push(INPS[ii]);
      TARGSE.push(OUTR);
    }
  }

  var DATAS  = []; // work samples (may be noised)
  var TARGS  = []; // work samples (may be noised)

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

  console.log('Training, please wait ...');
  if (!NN.doTrain(NET, DATAS, TARGS, null, null, 0))
  {
    console.log('Training failed!', NET);
    return(false);
  }

  console.log('Training complete', NET);

  // Verification

  function verifyProc(NET, DATAS, TARGS, stepName, imagesPerSample)
  {
    var CHKRS = [];
    for (var dataIndex = 0; dataIndex < DATAS.length; dataIndex++)
    {
      CHKRS.push(NN.doProc(NET, DATAS[dataIndex]));
    }

    var veps = 0.4; // epsilon for verification

    var isOK = true;
    for (var dataIndex = 0; dataIndex < DATAS.length; dataIndex++)
    {
      if (!NN.isResultMatchSimpleFunc(TARGS[dataIndex], CHKRS[dataIndex], veps))
      {
        console.log('Verification step '+stepName+'['+dataIndex+']'+':FAILED', [TARGS[dataIndex], CHKRS[dataIndex], veps]);
        var T = sampleAddLetTexts(DATAS[dataIndex], null, true, true, true, true);
        for (var lineIndex = 0; lineIndex < T.length; lineIndex++)
        {
          var imageIndex = dataIndex % imagesPerSample;
          var sampleIndex = (dataIndex-imageIndex) / imagesPerSample;
          console.log(T[lineIndex], lineIndex, sampleIndex, imageIndex);
        }
        isOK = false;
      }
    }

    if (isOK) { console.log('Verification step '+stepName+':OK'); }

    return(isOK);
  }

  verifyProc(NET, DATAS, TARGS, 'Source', DATAS.length / DATASE.length);

  // Create noised data

  var DATASN = [];
  for (var dataIndex = 0; dataIndex < DATAS.length; dataIndex++)
  {
    DATASN.push(getNoisedInput(DATAS[dataIndex],1));
  }

  verifyProc(NET, DATASN, TARGS, 'Noised.1', DATASN.length / DATASE.length);

  var DATASN = [];
  for (var dataIndex = 0; dataIndex < DATAS.length; dataIndex++)
  {
    DATASN.push(getNoisedInput(DATAS[dataIndex],2));
  }

  verifyProc(NET, DATASN, TARGS, 'Noised.3', DATASN.length / DATASE.length);
}

//sampleOcrNetwork();
