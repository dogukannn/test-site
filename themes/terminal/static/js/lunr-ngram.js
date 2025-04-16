/**
 *
 * lunr-ngram.js
 * Improved version with precise prefix matching
 */

(function (root, factory) {
  if (typeof define === 'function' && define.amd) {
    define(['lunr'], factory);
  } else if (typeof exports === 'object') {
    module.exports = factory(require('lunr'));
  } else {
    root.lunrNgram = factory(root.lunr);
  }
}(this, function (lunr) {
  // Helper function to generate word prefixes
  function generatePrefixes(str) {
    const prefixes = new Set();
    str = str.toLowerCase().trim();
    
    // Add all prefixes of the word
    for (let i = 1; i <= str.length; i++) {
      prefixes.add(str.substring(0, i));
    }
    
    return Array.from(prefixes);
  }

  const customTokenizer = function (input, metadata) {
    if (!input) return [];
    
    // Split input into words
    const words = input.toString()
                      .toLowerCase()
                      .trim()
                      .split(/[\s-]+/)
                      .filter(word => word.length > 0);
    
    const tokens = [];
    
    words.forEach(word => {
      // Add the complete word
      tokens.push(new lunr.Token(
        word,
        lunr.utils.clone(metadata)
      ));
      
      // Add all prefixes
      generatePrefixes(word).forEach(prefix => {
        tokens.push(new lunr.Token(
          prefix,
          { ...lunr.utils.clone(metadata), position: [prefix.length] }
        ));
      });
    });
    
    return tokens;
  };

  return function (builder) {
    // Override the default tokenizer
    builder.tokenizer = customTokenizer;
    
    // Remove stemming since we're using prefix matching
    builder.pipeline.remove(lunr.stemmer);
    builder.searchPipeline.remove(lunr.stemmer);
  };
}));
