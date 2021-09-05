# NNJS
Simple Self-Contained Neural Network toolkit written on JavaScript (usefull as sample or tutorial)

## Example

### Create network 
```
// Create NN.Layers
var IN  = new NN.Layer(28*28, NN.TheNeuronFactory(NN.InputNeuron)); IN.addNeuron(new NN.BiasNeuron());
var L1  = new NN.Layer(28*28, NN.TheNeuronFactory(NN.ProcNeuron)); L1.addNeuron(new NN.BiasNeuron()); 
var OUT = new NN.Layer(10, NN.TheNeuronFactory(NN.ProcNeuron)); // Outputs: 0="0", 1="1", 2="2", ...
// Connect layers
L1.addInputAll(IN);
OUT.addInputAll(L1); 
// Create NN.Network by NN.Layers
var NET = new NN.Network(); NET.addLayer(IN); NET.addLayer(L1); NET.addLayer(OUT);
```

### Train
```
// NN.doTrain(Net,DATAS, TARGS) // run training session
//var DATAS = [ [...], [...], ... ]; // source data samples (each sample holds values for inputs)
//var TARGS = [ [...], [...], ... ]; // expected result vectors (each result holds expected outputs)
console.log("Training, please wait ...");
if (!NN.doTrain(NET, DATAS, TARGS))
{
  console.log("Training failed!", NET);
}
console.log("Training complete", NET);
```

### Run inference
```
// NN.doProc(Net,DATA) // run single inference calculation
// Input DATA (must have same count as number of inputs)
// Returns result vector (will have same count as number of outputs)
var CALC = NN.doProc(NET, DATA);
```

## Cousin C++ project: NNCPP
There is a NN engine implementation on c++ with (almost) the same API:
https://github.com/openlab-vn-ua/NNCPP

## Useful links (and thanks to!)
* Steven Miller, "Mind: How to Build a Neural Network (Part One)"<br/>
http://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
* Matt Mazur, "A Step by Step Backpropagation Example"<br/>
https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
* Arnis71, "Introduction to neural networks" (on Russian)<br/>
https://habrahabr.ru/post/312450/ <br/>
https://habrahabr.ru/post/313216/ <br/>
https://habrahabr.ru/post/307004/
