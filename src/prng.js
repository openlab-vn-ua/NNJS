// Simple Random value generator
// Uses an optimized version of the Park-Miller PRNG.
// Inspired by 
// http://www.firstpr.com.au/dsp/rand31/
// https://gist.github.com/blixt/f17b47c62508be59987b
// Open Source Software under MIT License

function Random(seed)
{
  var that = this || {}; // works with both new Random(...) or just Random(...)

  // seed (state)

  that._seed = null; // not initialized

  that.srand = function(seed) // C-like API
  {
    if (seed < 0) { seed = -seed; }
    if (seed == 0) { seed = 1; }
    that._seed = seed % 2147483647;
  }

  that.setSeed = that.srand; // Synonym

  // Constructor

  if (seed == null) { seed = 42; }
  that.srand(seed);

  // Constants                        

  /// Min output value for next() == 1
  that.NEXT_MIN = 1;
          
  /// Max output value for next() == 2^31-2
  that.NEXT_MAX = 2147483646; 

  /// Returns a pseudo-random value between NEXT_MIN (1) and NEXT_MAX (2^32 - 2) [NEXT_MIN .. NEXT_MAX] (inclusive)
  that.next = function()
  {
    return that._seed = that._seed * 16807 % 2147483647;
  };

  /// Returns a pseudo-random floating point number in range [0.0 .. 1.0) (upper bound exclsive)
  that.nextFloat = function()
  {
    // We know that result of next() will be 1 to 2147483646 (inclusive).
    return (that.next() - that.NEXT_MIN) / (that.NEXT_MAX - that.NEXT_MIN + 1);
  };

  // Like C Random

  /// Maximum output value for rand() == [0..RAND_MAX_VALUE]
  that.RAND_MAX_VALUE = (that.NEXT_MAX - that.NEXT_MIN);

  that.RAND_MAX = that.RAND_MAX_VALUE; // Synonym

  /// Returns next random in range [0..RAND_MAX_VALUE] (inclusive)
  that.rand = function()
  {
    return that.next() - that.NEXT_MIN;
  }

  /// randFloat()        Returns random in range [0.0 .. 1.0] (inclusive)
  /// randFloat(max)     Returns random in range [0.0 .. max] (inclusive)
  /// randFloat(min,max) Returns random in range [min .. max] (inclusive)
  that.randFloat = function(minOrMax, max)
  {
    if (max == null)
    {
      max = minOrMax;
      if (max == null)
      {
        return that.rand() * 1.0 / that.RAND_MAX_VALUE; 
      }
      else
      {
        return (that.rand() * 1.0 / that.RAND_MAX_VALUE) * max;
      }
    }
    else
    {
      var min = minOrMax;
      if (min == null) { min = 0.0; }
      var diff = max - min;
      return (that.rand() * 1.0 / that.RAND_MAX_VALUE) * diff + min;
    }
  }

  return that;
}
// static items
(function ()
{
  var self = Random;
  self.getRandomSeed = function (timeVal)
  {
    return timeVal % 0x7FFF0000 + 1;
  }
})();


