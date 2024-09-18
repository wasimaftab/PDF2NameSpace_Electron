// renderer.js
document.getElementById('select-folder').addEventListener('click', () => {
    window.electronAPI.selectPdfFolder();
  });
  
  window.electronAPI.onPdfFolderSelected((event, folderPath) => {
    const output = document.getElementById('output');
    output.textContent = `Selected folder: ${folderPath}\nProcessing...`;
  });
  
  window.electronAPI.onProcessingUpdate((event, message) => {
    const output = document.getElementById('output');
    output.textContent += `\n${message}`;
    output.scrollTop = output.scrollHeight; // Auto-scroll
  });
  
  window.electronAPI.onProcessingComplete((event, message) => {
    const output = document.getElementById('output');
    output.textContent += `\n${message}\nProcessing complete.`;
  });
  