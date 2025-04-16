(function() {
  let data = [];
  let searchIndex = null;

  function fetchSearchIndex() {
    return fetch('/index.json')
      .then(response => response.json())
      .then(indexData => {
        data = indexData;

        searchIndex = lunr(function() {
          this.pipeline.reset();
          this.pipeline.add(lunr.trimmer);
          
          this.searchPipeline.reset();
          this.searchPipeline.add(lunr.trimmer);

          this.ref('href');
          this.field('text');
          
          // Add documents to the index
          indexData.forEach(doc => {
            // Get all unique words from title and content
            const allText = `${doc.title} ${doc.content}`;
            const words = new Set(allText.toLowerCase().split(/\s+/));
            
            this.add({
              href: doc.href,
              // Join unique words with spaces to ensure word boundaries
              text: Array.from(words).join(' ')
            });
          });
        });

        window.searchIndex = searchIndex;
        console.log('Search index initialized');
      })
      .catch(error => {
        console.error('Error initializing search index:', error);
      });
  }

  function search(query) {
    if (!searchIndex || !query) {
      return [];
    }

    query = query.toLowerCase().trim();

    try {
      // Get unique words from query
      const words = [...new Set(query.split(/\s+/).filter(w => w))];
      
      if (words.length === 0) {
        return [];
      }

      // Build search query with unique terms only
      const searchQuery = words.map(word => `+text:${word}*`).join(' ');
      
      const searchResults = searchIndex.search(searchQuery);
      
      // Filter results and ensure good relevance
      return searchResults
        .filter(result => result.score > 0.1)  // Increased threshold
        .map(result => data.find(doc => doc.href === result.ref))
        .filter(Boolean);

    } catch (e) {
      console.error('Search error:', e);
      return [];
    }
  }

  window.fetchSearchIndex = fetchSearchIndex;
  window.search = search;
})();
