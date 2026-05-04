// Ambient declaration for plotly.js-dist-min — the package ships JS only.
// react-plotly.js/factory accepts whatever Plotly object we pass in; we don't
// need the full @types/plotly.js (heavy, version-bound) just to satisfy tsc.
declare module 'plotly.js-dist-min';
declare module 'react-plotly.js/factory';
