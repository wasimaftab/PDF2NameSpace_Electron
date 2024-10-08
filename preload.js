// preload.js
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  selectPdfFolder: (namespace) => ipcRenderer.invoke('select-pdf-folder', namespace),
  onPdfFolderSelected: (callback) => ipcRenderer.on('pdf-folder-selected', callback),
  onProcessingUpdate: (callback) => ipcRenderer.on('processing-update', callback),
  onProcessingComplete: (callback) => ipcRenderer.on('processing-complete', callback),
  onGrobidNotRunning: (callback) => ipcRenderer.on('grobid-not-running', callback),
});
