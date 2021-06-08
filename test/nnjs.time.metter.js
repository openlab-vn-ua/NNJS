// Simple operation timer checker

// Require nnjs.js

(function () {

if (NN == null) { throw 'nnjs.js should be included first '; }

// Simple console write trainging reporter

function TimeMetter()
{
  var that = this;
  var self = TimeMetter;

  // static reimport

  var millisGlobal = self.millisGlobal;

  // const

  var VOID_TIME = -1;

  // fields

  var timeStart = VOID_TIME;
  var timeStop  = VOID_TIME;

  // Methods

  function start()
  {
    timeStart = millisGlobal();
    timeStop  = VOID_TIME;
  }
  that.start = start;

  function stop()
  {
    if (timeStart == VOID_TIME) { return -1; }
    timeStop = millisGlobal();
    return timeStop - timeStart;
  }
  that.stop = stop;

  function isStarted()
  {
    return (timeStart != VOID_TIME);
  }
  that.isStarted = isStarted;

  function isStoped()
  {
    return (timeStop != VOID_TIME);
  }
  that.isStoped = isStoped;

  function millisPassed()
  {
    if (timeStart == VOID_TIME) { return -1; }
    if (timeStop == VOID_TIME) { return millisGlobal() - timeStart; }
    return timeStop - timeStart;
  }
  that.millisPassed = millisPassed;

  // Constructor

  // start();

  // export

  return that;
}
// static
(function()
{
  var self = TimeMetter;
  self.millisGlobal = function millisGlobal()
  {
    return Date.now();
  }
})();

if (NN != null) {
NN.TimeMetter = TimeMetter;
}

})();