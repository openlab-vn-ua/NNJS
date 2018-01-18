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

function sampleOcrNetwork()
{
  var SAMPLES = sampleOcrGetSamples();

  // The Net

  var IN  = new NN.Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY, NN.InputNeuron); IN.addNeuron(NN.BiasNeuron);
  var L1  = new NN.Layer(SAMPLE_OCR_SX*SAMPLE_OCR_SY, NN.Neuron); L1.addNeuron(NN.BiasNeuron); L1.addInputAll(IN);
  var OUT = new NN.Layer(SAMPLES.length, NN.Neuron); OUT.addInputAll(L1); // Outputs: 0=A, 1=B, 2=C, ...
  
  var NET = [IN, L1, OUT];

  // Train input preparation

  function getNoisedInput(L, noise)
  {
    if (noise == null) { noise = 0; }

    var R = L.slice();

    for (var i = 0; i < noise; i++)
    {
      var ii = NN.Internal.getRandomInt(0,R.length);
      R[ii] = 1-R[ii];
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
        ox = (x + -sx); ox = (ox < 0) ? SAMPLE_OCR_SX+ox : ox; ox %= SAMPLE_OCR_SX;
        oy = (y + -sy); oy = (oy < 0) ? SAMPLE_OCR_SY+oy : oy; oy %= SAMPLE_OCR_SY;
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

  var INPSS = SAMPLES; // 2D array [letter][sample] = data
  var OUTRS = [ ]; // 1D array [letter] = result expected

  for (var i = 0; i < INPSS.length; i++)
  {
    var OUTR = getR1Array(i,INPSS.length); // target result for this letter
    OUTRS.push(OUTR);
  }

  //var DATASE = [ getLArray(LA0), getLArray(LB0), getLArray(LC0), getLArray(LD0)];
  //var TARGSE = [ getR1Out(0,4),  getR1Out(1,4),  getR1Out(2,4),  getR1Out(3,4)];

  var DATASE = [ ]; // data source etalon (no noise) : source samples as plain array
  var TARGSE = [ ]; // data source etalon (no noise) : results expected as plain array

  for (var i = 0; i < INPSS.length; i++)
  {
    var INPS = INPSS[i]; // letter samples (may be many)
    var OUTR = OUTRS[i]; // target result for this letter

    for (var ii = 0; ii < INPS.length; ii++)
    {
      DATASE.push(INPS[ii]);
      TARGSE.push(OUTR);
    }
  }

  var DATAS  = []; // work samples (may be noised)
  var TARGS  = []; // work samples (may be noised)

  for (var i = 0; i < DATASE.length; i++)
  {
    DATAS.push(DATASE[i]);
    TARGS.push(TARGSE[i]);
    DATAS.push(getShiftedImg(DATASE[i],0,1));
    DATAS.push(getShiftedImg(DATASE[i],1,0));
    DATAS.push(getShiftedImg(DATASE[i],1,1));
    TARGS.push(TARGSE[i]);
    TARGS.push(TARGSE[i]);
    TARGS.push(TARGSE[i]);
    DATAS.push(getShiftedImg(DATASE[i],0,-1));
    DATAS.push(getShiftedImg(DATASE[i],-1,0));
    DATAS.push(getShiftedImg(DATASE[i],-1,-1));
    TARGS.push(TARGSE[i]);
    TARGS.push(TARGSE[i]);
    TARGS.push(TARGSE[i]);
  }

  function dumpSamples(DATAS)
  {
    var scount = DATASE.length; // sumber of samples
    var icount = DATAS.length / scount; // number of images per sample

    for (var i = 0; i < scount; i++)
    {
      var t = '!';
      for (var s = 0; s < icount; s++)
      {
        for (var x = 0; x < SAMPLE_OCR_SX; x++)
        {
          t += '0';
        }
        t += '!';
      }
      console.log(t,i);

      for (var y = 0; y < SAMPLE_OCR_SY; y++)
      {
        var t = '|';
        for (var s = 0; s < icount; s++)
        {
          for (var x = 0; x < SAMPLE_OCR_SX; x++)
          {
            t += '' + ((DATAS[i*icount+s][y*SAMPLE_OCR_SX+x] == 0) ? ' ' : '1');
          }
          t += '|';
        }
        console.log(t,i,y);
      }
    }
  }

  dumpSamples(DATAS);

  if (NN.doTrain(NET, DATAS, TARGS, null, null, 0))
  {
    console.log('Training complete, verification', NET);
  }
}

//sampleOcrNetwork();
