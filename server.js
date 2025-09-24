const express = require('express');
const path = require('path');

const app = express();
const port = parseInt(process.env.PORT, 10) || 8080;

// Serve static files (index.html at root)
app.use(express.static(path.join(__dirname)));

// Simple health endpoint
app.get('/healthz', (req, res) => {
  res.type('text/plain').send('ok');
});

// Fallback to index.html for other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(port, '0.0.0.0', () => {
  console.log(`Project-1 listening on http://0.0.0.0:${port}`);
});
