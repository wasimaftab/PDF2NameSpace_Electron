// preload.js
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  selectPdfFolder: () => ipcRenderer.send('select-pdf-folder'),
  onPdfFolderSelected: (callback) => ipcRenderer.on('pdf-folder-selected', callback),
  onProcessingUpdate: (callback) => ipcRenderer.on('processing-update', callback),
  onProcessingComplete: (callback) => ipcRenderer.on('processing-complete', callback),
});
